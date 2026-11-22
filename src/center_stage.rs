use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use log::{info, warn};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

const DET_WIDTH: usize = 640;
const DET_HEIGHT: usize = 640;
/// Active 16:9 image area letterboxed into the top of the 640x640 YuNet input (preserves exact 1:1 aspect ratio).
const DET_ACTIVE_H: usize = 360;

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    /// Normalized [0..1] top-left X, Y, width, height
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

/// Active Center Stage viewport transform applied on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct CropRect {
    /// Normalized viewport center X in [0..1]
    pub cx: f32,
    /// Normalized viewport center Y in [0..1]
    pub cy: f32,
    /// Zoom factor >= 1.0 (1.0 = full wide frame, 1.8 = zoomed in 1.8x)
    pub zoom: f32,
}

impl Default for CropRect {
    fn default() -> Self {
        Self {
            cx: 0.5,
            cy: 0.5,
            zoom: 1.0,
        }
    }
}

/// Lightweight YuNet ONNX face detector (~230 KB, ~1.5 ms on ARM64 CPU).
pub struct YuNetDetector {
    session: Session,
    input_buf: Vec<f32>,
}

impl YuNetDetector {
    pub fn new(env: &Environment, model_path: &Path, threads: usize) -> Result<Self> {
        let session = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("failed to create YuNet SessionBuilder: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("failed to set YuNet threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to load YuNet ONNX model from {}: {e}",
                    model_path.display()
                )
            })?;

        info!(
            "Loaded YuNet face detector {} (inputs={:?}, run={}x{}, threads={})",
            model_path.display(),
            session.inputs(),
            DET_WIDTH,
            DET_HEIGHT,
            threads
        );

        Ok(Self {
            session,
            input_buf: vec![0.0f32; 3 * DET_WIDTH * DET_HEIGHT],
        })
    }

    /// Downsample NV12 (Rec. 709 limited) into top 640x360 letterbox of BGR NCHW f32 [0..255] for YuNet.
    fn fill_bgr_from_nv12(&mut self, nv12: &[u8], src_w: u32, src_h: u32) {
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = DET_WIDTH;
        let dh = DET_ACTIVE_H;
        let plane = DET_WIDTH * DET_HEIGHT;
        let uv_offset = sw * sh;

        let (b_plane, rest) = self.input_buf.split_at_mut(plane);
        let (g_plane, r_plane) = rest.split_at_mut(plane);

        for dy in 0..dh {
            let sy = ((dy * sh + sh / 2) / dh).min(sh - 1);
            let y_row = sy * sw;
            let uv_row = uv_offset + (sy / 2) * sw;
            let dst_row = dy * dw;

            for dx in 0..dw {
                let sx = ((dx * sw + sw / 2) / dw).min(sw - 1);
                let y_raw = nv12[y_row + sx] as f32;
                let uv_idx = uv_row + (sx & !1);
                let u_raw = nv12[uv_idx] as f32;
                let v_raw = nv12[uv_idx + 1] as f32;

                let y = (y_raw - 16.0) * (1.0 / 219.0);
                let cb = (u_raw - 128.0) * (1.0 / 224.0);
                let cr = (v_raw - 128.0) * (1.0 / 224.0);

                let r = (y + 1.5748 * cr).clamp(0.0, 1.0) * 255.0;
                let g = (y - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0) * 255.0;
                let b = (y + 1.8556 * cb).clamp(0.0, 1.0) * 255.0;

                let idx = dst_row + dx;
                b_plane[idx] = b;
                g_plane[idx] = g;
                r_plane[idx] = r;
            }
        }
    }

    pub fn detect(&mut self, nv12: &[u8], src_w: u32, src_h: u32) -> Result<Vec<FaceBox>> {
        self.fill_bgr_from_nv12(nv12, src_w, src_h);

        let tensor = TensorRef::from_array_view((
            [1usize, 3, DET_HEIGHT, DET_WIDTH],
            self.input_buf.as_slice(),
        ))?;

        let outputs = self.session.run(ort::inputs!["input" => tensor])?;

        let mut candidates = Vec::new();
        let score_thresh = 0.72f32;
        let inv_w = 1.0 / DET_WIDTH as f32;
        let inv_h = 1.0 / DET_ACTIVE_H as f32;

        for (stride, cls_k, obj_k, bbox_k) in [
            (8usize, "cls_8", "obj_8", "bbox_8"),
            (16usize, "cls_16", "obj_16", "bbox_16"),
            (32usize, "cls_32", "obj_32", "bbox_32"),
        ] {
            let (_, cls) = outputs[cls_k].try_extract_tensor::<f32>()?;
            let (_, obj) = outputs[obj_k].try_extract_tensor::<f32>()?;
            let (_, bbox) = outputs[bbox_k].try_extract_tensor::<f32>()?;

            let cols = DET_WIDTH / stride;
            let active_rows = DET_ACTIVE_H.div_ceil(stride);
            let s = stride as f32;

            for r in 0..active_rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    if idx >= cls.len() || idx >= obj.len() || idx * 4 + 3 >= bbox.len() {
                        continue;
                    }
                    let cs = cls[idx].clamp(0.0, 1.0);
                    let os = obj[idx].clamp(0.0, 1.0);
                    let score = (cs * os).sqrt();
                    if score < score_thresh {
                        continue;
                    }

                    let dx = bbox[idx * 4];
                    let dy = bbox[idx * 4 + 1];
                    let dw = bbox[idx * 4 + 2];
                    let dh = bbox[idx * 4 + 3];

                    let cx = (c as f32 + dx) * s;
                    let cy = (r as f32 + dy) * s;
                    let bw = dw.exp() * s;
                    let bh = dh.exp() * s;

                    let x = ((cx - bw * 0.5) * inv_w).clamp(0.0, 1.0);
                    let y = ((cy - bh * 0.5) * inv_h).clamp(0.0, 1.0);
                    let w = (bw * inv_w).clamp(0.01, 1.0);
                    let h = (bh * inv_h).clamp(0.01, 1.0);

                    // Ignore tiny background/poster faces (< 10% of frame height)
                    if h < 0.10 {
                        continue;
                    }

                    candidates.push(FaceBox { x, y, w, h, score });
                }
            }
        }

        Ok(nms(candidates, 0.35))
    }
}

fn iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 1e-6 { 0.0 } else { inter / union }
}

fn nms(mut boxes: Vec<FaceBox>, iou_thresh: f32) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<FaceBox> = Vec::new();
    for b in boxes {
        if keep.iter().all(|k| iou(k, &b) < iou_thresh) {
            keep.push(b);
        }
    }
    keep
}

/// Apple-style Center Stage controller:
/// - Runs YuNet at 15 Hz (every 2nd frame) for minimal CPU overhead (~0.8 ms avg).
/// - Frames 1 face in a balanced portrait (rule-of-thirds vertical placement) OR smoothly zooms out
///   when multiple people enter the shot.
/// - Uses deadzone hysteresis + critically-damped glide so micro head movements never cause jitter.
pub struct CenterStageController {
    detector: YuNetDetector,
    frame_idx: u64,
    current: CropRect,
    target: CropRect,
    last_face_seen: Option<Instant>,
    last_face_center: Option<[f32; 2]>,
    last_face_count: usize,
    last_tick: Instant,
}

impl CenterStageController {
    pub fn new(detector: YuNetDetector) -> Self {
        Self {
            detector,
            frame_idx: 0,
            current: CropRect::default(),
            target: CropRect::default(),
            last_face_seen: None,
            last_face_center: None,
            last_face_count: 0,
            last_tick: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.frame_idx = 0;
        self.current = CropRect::default();
        self.target = CropRect::default();
        self.last_face_seen = None;
        self.last_face_center = None;
        self.last_face_count = 0;
        self.last_tick = Instant::now();
    }

    /// Returns current smoothed `CropRect` and optional detected face center (for Studio Light).
    /// When `alpha_mask` is provided, faces located in the background (e.g. posters/books on a shelf)
    /// are automatically ignored!
    pub fn update(
        &mut self,
        nv12: &[u8],
        src_w: u32,
        src_h: u32,
        alpha_mask: Option<(&[u8], u32, u32)>,
        enabled: bool,
        zoom_pref: f32,
    ) -> (CropRect, Option<[f32; 2]>) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().clamp(0.005, 0.1);
        self.last_tick = now;

        if !enabled {
            self.target = CropRect::default();
            self.step_smoothing(dt, 4.5);
            return (self.current, None);
        }

        // Run YuNet every 6th frame (5 Hz detection, 30 Hz smooth spring interpolation -> ~2-4 ms avg)
        if self.frame_idx.is_multiple_of(6) {
            match self.detector.detect(nv12, src_w, src_h) {
                Ok(mut faces) => {
                    // Reject faces located in the background (posters, book covers, paintings)
                    if let Some((mask_data, mw, mh)) = alpha_mask {
                        let fg_faces: Vec<FaceBox> = faces
                            .iter()
                            .copied()
                            .filter(|f| {
                                let fx = ((f.x + f.w * 0.5) * mw as f32).clamp(0.0, (mw - 1) as f32) as usize;
                                let fy = ((f.y + f.h * 0.5) * mh as f32).clamp(0.0, (mh - 1) as f32) as usize;
                                let idx = fy * (mw as usize) + fx;
                                mask_data.get(idx).copied().unwrap_or(255) >= 100
                            })
                            .collect();
                        if !fg_faces.is_empty() {
                            faces = fg_faces;
                        }
                    }

                    // Only count secondary faces if they are real people of comparable size (>= 55% of primary face)
                    if faces.len() > 1 {
                        let max_h = faces.iter().map(|f| f.h).fold(0.0f32, f32::max);
                        faces.retain(|f| f.h >= max_h * 0.55);
                    }

                    if !faces.is_empty() {
                        self.last_face_seen = Some(now);
                        self.update_target_from_faces(&faces, zoom_pref);
                    } else if let Some(seen) = self.last_face_seen
                        && now.duration_since(seen).as_secs_f32() > 2.0
                    {
                        self.target = CropRect::default();
                        self.last_face_center = None;
                        self.last_face_count = 0;
                    }
                }
                Err(err) => {
                    warn!("YuNet detection error: {err:#}");
                }
            }
        }
        self.frame_idx = self.frame_idx.wrapping_add(1);

        self.step_smoothing(dt, 3.6);
        (self.current, self.last_face_center)
    }

    fn update_target_from_faces(&mut self, faces: &[FaceBox], zoom_pref: f32) {
        // Compute union bounding box of all detected faces (supports multi-person zoom-out!)
        let mut min_x = 1.0f32;
        let mut min_y = 1.0f32;
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;

        for f in faces {
            min_x = min_x.min(f.x);
            min_y = min_y.min(f.y);
            max_x = max_x.max(f.x + f.w);
            max_y = max_y.max(f.y + f.h);
        }

        let union_w = (max_x - min_x).max(0.05);
        let union_h = (max_y - min_y).max(0.08);
        let face_cx = (min_x + max_x) * 0.5;
        let face_cy = (min_y + max_y) * 0.5;

        self.last_face_center = Some([face_cx, face_cy]);

        let zp = zoom_pref.clamp(0.0, 1.0);
        // Always keep a minimum tracking zoom floor for a single person so Center Stage has
        // genuine pan margin even when sitting close to a MacBook Pro (1.18x .. 1.60x floor),
        // while zooming smoothly out to 1.0x whenever 2+ people are in frame.
        let min_zoom = if faces.len() == 1 {
            1.12 + zp * 0.38
        } else {
            1.0
        };
        let max_zoom = 1.35 + zp * 0.65;

        // Target face height scales with zoom preference (~36%..50% for single portrait)
        let desired_face_h = if faces.len() == 1 {
            0.36 + zp * 0.14
        } else {
            0.28
        };
        let zoom_by_h = desired_face_h / union_h;
        let zoom_by_w = 1.0 / (union_w * 2.2).max(0.35);
        let desired_zoom = zoom_by_h.min(zoom_by_w).clamp(min_zoom, max_zoom);

        // Place face slightly above vertical center (rule of thirds) so shoulders fit cleanly
        let vertical_offset = 0.075 / desired_zoom;
        let clamped = clamp_crop(CropRect {
            cx: face_cx,
            cy: face_cy + vertical_offset,
            zoom: desired_zoom,
        });

        if faces.len() != self.last_face_count {
            info!(
                "Center Stage tracking {} face(s) at ({:.2}, {:.2}), target zoom {:.2}x",
                faces.len(),
                face_cx,
                face_cy,
                clamped.zoom
            );
            self.last_face_count = faces.len();
        }

        // Gentle deadzone hysteresis so micro head twitches don't jitter, while real shifts pan smoothly
        let pos_deadzone = 0.020 / self.current.zoom.max(1.0);
        let zoom_deadzone = 0.04;

        let dx = (clamped.cx - self.target.cx).abs();
        let dy = (clamped.cy - self.target.cy).abs();
        let dz = (clamped.zoom - self.target.zoom).abs();

        if dx > pos_deadzone || dy > pos_deadzone {
            self.target.cx = clamped.cx;
            self.target.cy = clamped.cy;
        }
        if dz > zoom_deadzone {
            self.target.zoom = clamped.zoom;
        }
        self.target = clamp_crop(self.target);
    }

    fn step_smoothing(&mut self, dt: f32, speed: f32) {
        let k_pos = 1.0 - (-dt * speed).exp();
        let k_zoom = 1.0 - (-dt * (speed * 0.75)).exp();

        self.current.cx += (self.target.cx - self.current.cx) * k_pos;
        self.current.cy += (self.target.cy - self.current.cy) * k_pos;
        self.current.zoom += (self.target.zoom - self.current.zoom) * k_zoom;
        self.current = clamp_crop(self.current);
    }
}

fn clamp_crop(mut c: CropRect) -> CropRect {
    c.zoom = c.zoom.clamp(1.0, 2.2);
    let half = 0.5 / c.zoom;
    c.cx = c.cx.clamp(half, 1.0 - half);
    c.cy = c.cy.clamp(half, 1.0 - half);
    c
}
