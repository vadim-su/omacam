use std::path::Path;

use anyhow::Result;
use log::info;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

/// Resolution fed into Robust Video Matting (16:9 to match 1920x1080 FaceTime HD Camera).
pub const MATTING_WIDTH: u32 = 512;
pub const MATTING_HEIGHT: u32 = 288;

/// Alpha matte + color-decontaminated foreground produced by a segmenter.
pub struct AlphaMask {
    pub width: u32,
    pub height: u32,
    /// Single-channel alpha matte in 0..=255 (`width * height` bytes).
    pub data: Vec<u8>,
    /// RVM color-decontaminated foreground RGBA8 (`width * height * 4` bytes) for halo-free hair edges.
    pub fgr_rgba: Vec<u8>,
    /// Smoothed foreground focal point in normalized [0..1, 0..1] coordinates (for Studio Light).
    pub centroid: [f32; 2],
}

/// Abstract person-matting interface so the underlying ML runtime/model can be swapped cleanly.
pub trait Segmenter: Send {
    fn mask_dims(&self) -> (u32, u32);
    fn infer_alpha(&mut self, nv12: &[u8], src_w: u32, src_h: u32) -> Result<&AlphaMask>;
    fn reset(&mut self);
}

struct RecurrentState {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl RecurrentState {
    fn initial() -> Self {
        Self {
            shape: vec![1, 1, 1, 1],
            data: vec![0.0],
        }
    }
}

/// Robust Video Matting (MobileNetV3) running via ONNX Runtime (`ort`).
/// Maintains temporal recurrent states (r1..r4) across frames for flicker-free hair matting.
pub struct RvmSegmenter {
    session: Session,
    downsample_ratio: f32,
    src_buf: Vec<f32>,
    r1: RecurrentState,
    r2: RecurrentState,
    r3: RecurrentState,
    r4: RecurrentState,
    mask: AlphaMask,
    centroid_initialized: bool,
}

impl RvmSegmenter {
    pub fn new(
        env: &Environment,
        model_path: &Path,
        downsample_ratio: f32,
        threads: usize,
    ) -> Result<Self> {
        let session = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("failed to create ONNX SessionBuilder: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("failed to configure ONNX intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to load RVM ONNX model from {}: {e}",
                    model_path.display()
                )
            })?;

        info!(
            "Loaded RVM matting model {} ({}x{}, downsample_ratio={:.2}, threads={})",
            model_path.display(),
            MATTING_WIDTH,
            MATTING_HEIGHT,
            downsample_ratio,
            threads
        );

        let pixels = (MATTING_WIDTH * MATTING_HEIGHT) as usize;
        Ok(Self {
            session,
            downsample_ratio,
            src_buf: vec![0.0f32; 3 * pixels],
            r1: RecurrentState::initial(),
            r2: RecurrentState::initial(),
            r3: RecurrentState::initial(),
            r4: RecurrentState::initial(),
            mask: AlphaMask {
                width: MATTING_WIDTH,
                height: MATTING_HEIGHT,
                data: vec![255u8; pixels],
                fgr_rgba: vec![0u8; pixels * 4],
                centroid: [0.5, 0.42],
            },
            centroid_initialized: false,
        })
    }

    /// Downsample NV12 (Rec. 709 limited range) directly into NCHW f32 [1, 3, H, W] in 0..=1.
    fn fill_input_from_nv12(&mut self, nv12: &[u8], src_w: u32, src_h: u32) {
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = MATTING_WIDTH as usize;
        let dh = MATTING_HEIGHT as usize;
        let plane_len = dw * dh;
        let uv_offset = sw * sh;

        let (r_plane, rest) = self.src_buf.split_at_mut(plane_len);
        let (g_plane, b_plane) = rest.split_at_mut(plane_len);

        for dy in 0..dh {
            let sy = (dy * sh + sh / 2) / dh;
            let sy = sy.min(sh - 1);
            let y_row = sy * sw;
            let uv_row = uv_offset + (sy / 2) * sw;
            let dst_row = dy * dw;

            for dx in 0..dw {
                let sx = (dx * sw + sw / 2) / dw;
                let sx = sx.min(sw - 1);

                let y_raw = nv12[y_row + sx] as f32;
                let uv_idx = uv_row + (sx & !1);
                let u_raw = nv12[uv_idx] as f32;
                let v_raw = nv12[uv_idx + 1] as f32;

                // Rec. 709 limited-range YCbCr -> sRGB [0..1]
                let y = (y_raw - 16.0) * (1.0 / 219.0);
                let cb = (u_raw - 128.0) * (1.0 / 224.0);
                let cr = (v_raw - 128.0) * (1.0 / 224.0);

                let r = (y + 1.5748 * cr).clamp(0.0, 1.0);
                let g = (y - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0);
                let b = (y + 1.8556 * cb).clamp(0.0, 1.0);

                let idx = dst_row + dx;
                r_plane[idx] = r;
                g_plane[idx] = g;
                b_plane[idx] = b;
            }
        }
    }
}

impl Segmenter for RvmSegmenter {
    fn mask_dims(&self) -> (u32, u32) {
        (MATTING_WIDTH, MATTING_HEIGHT)
    }

    fn reset(&mut self) {
        self.r1 = RecurrentState::initial();
        self.r2 = RecurrentState::initial();
        self.r3 = RecurrentState::initial();
        self.r4 = RecurrentState::initial();
        self.centroid_initialized = false;
    }

    fn infer_alpha(&mut self, nv12: &[u8], src_w: u32, src_h: u32) -> Result<&AlphaMask> {
        self.fill_input_from_nv12(nv12, src_w, src_h);

        let src_tensor = TensorRef::from_array_view((
            [1usize, 3, MATTING_HEIGHT as usize, MATTING_WIDTH as usize],
            self.src_buf.as_slice(),
        ))?;
        let r1_tensor = TensorRef::from_array_view((self.r1.shape.as_slice(), self.r1.data.as_slice()))?;
        let r2_tensor = TensorRef::from_array_view((self.r2.shape.as_slice(), self.r2.data.as_slice()))?;
        let r3_tensor = TensorRef::from_array_view((self.r3.shape.as_slice(), self.r3.data.as_slice()))?;
        let r4_tensor = TensorRef::from_array_view((self.r4.shape.as_slice(), self.r4.data.as_slice()))?;
        let ds_arr = [self.downsample_ratio];
        let ds_tensor = TensorRef::from_array_view(([1usize], ds_arr.as_slice()))?;

        let outputs = self.session.run(ort::inputs![
            "src" => src_tensor,
            "r1i" => r1_tensor,
            "r2i" => r2_tensor,
            "r3i" => r3_tensor,
            "r4i" => r4_tensor,
            "downsample_ratio" => ds_tensor,
        ])?;

        // Extract alpha matte (`pha`: [1, 1, H, W]) and color-decontaminated foreground (`fgr`: [1, 3, H, W])
        let (_pha_shape, pha_slice) = outputs["pha"].try_extract_tensor::<f32>()?;
        let (_fgr_shape, fgr_slice) = outputs["fgr"].try_extract_tensor::<f32>()?;
        let dw = MATTING_WIDTH as usize;
        let dh = MATTING_HEIGHT as usize;
        let plane = dw * dh;

        let mut sum_w = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;

        for (idx, (&a_f32, dst)) in pha_slice.iter().zip(self.mask.data.iter_mut()).enumerate() {
            let a = a_f32.clamp(0.0, 1.0);
            *dst = (a * 255.0 + 0.5) as u8;

            if idx + 2 * plane < fgr_slice.len() {
                let off = idx * 4;
                self.mask.fgr_rgba[off] = (fgr_slice[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 1] =
                    (fgr_slice[plane + idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 2] =
                    (fgr_slice[2 * plane + idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 3] = 255;
            }

            // Sample every 4th pixel for fast upper-body/face centroid tracking
            if idx & 3 == 0 && a > 0.35 {
                let x = (idx % dw) as f64 / dw as f64;
                let y = (idx / dw) as f64 / dh as f64;
                // Bias slightly toward upper part of silhouette (head/face)
                let head_bias = (1.15 - y * 0.6).max(0.25);
                let w = (a as f64) * head_bias;
                sum_w += w;
                sum_x += x * w;
                sum_y += y * w;
            }
        }

        if sum_w > 1.0 {
            let target_x = (sum_x / sum_w) as f32;
            let target_y = ((sum_y / sum_w) as f32 * 0.92).clamp(0.15, 0.85);
            if !self.centroid_initialized {
                self.mask.centroid = [target_x, target_y];
                self.centroid_initialized = true;
            } else {
                // Smooth EMA so Studio Light follows gently without jitter
                let smooth = 0.12f32;
                self.mask.centroid[0] += (target_x - self.mask.centroid[0]) * smooth;
                self.mask.centroid[1] += (target_y - self.mask.centroid[1]) * smooth;
            }
        }

        // Persist recurrent hidden states for the next frame
        update_recurrent(&outputs["r1o"], &mut self.r1)?;
        update_recurrent(&outputs["r2o"], &mut self.r2)?;
        update_recurrent(&outputs["r3o"], &mut self.r3)?;
        update_recurrent(&outputs["r4o"], &mut self.r4)?;

        Ok(&self.mask)
    }
}

fn update_recurrent(value: &ort::value::Value, dst: &mut RecurrentState) -> Result<()> {
    let (shape, slice) = value.try_extract_tensor::<f32>()?;
    dst.shape.clear();
    dst.shape.extend(shape.iter().map(|&d| d as usize));
    dst.data.clear();
    dst.data.extend_from_slice(slice);
    Ok(())
}
