use std::time::{Duration, Instant};

use anyhow::Result;
use log::info;

use crate::center_stage::{CenterStageController, CropRect};
use crate::gpu::GpuEffects;
use crate::segmenter::{AlphaMask, Segmenter};
use crate::settings::SharedSettings;

/// A single video frame in NV12 format (Y plane followed by interleaved UV plane).
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub nv12: Vec<u8>,
}

impl Frame {
    pub fn expected_nv12_size(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize) * 3 / 2
    }

    pub fn black(width: u32, height: u32) -> Self {
        let y_len = (width as usize) * (height as usize);
        let uv_len = y_len / 2;
        let mut nv12 = vec![16u8; y_len + uv_len];
        // Neutral chroma for NV12 is 128; limited-range black Y is 16.
        nv12[y_len..].fill(128);
        Self {
            width,
            height,
            nv12,
        }
    }
}

/// Processing pipeline applied to each captured frame before publishing to PipeWire.
pub trait FrameProcessor: Send {
    fn on_stream_start(&mut self) {}
    fn process(&mut self, frame: &mut Frame) -> Result<()>;
}

pub struct PassthroughProcessor;

impl FrameProcessor for PassthroughProcessor {
    fn process(&mut self, _frame: &mut Frame) -> Result<()> {
        Ok(())
    }
}

/// Full AI + GPU studio camera processor:
/// - ONNX Robust Video Matting (`Segmenter`)
/// - ONNX YuNet Center Stage (`CenterStageController`)
/// - Vulkan compute effects (`GpuEffects`: Bicubic zoom, Edge-guided matting, RVM fgr decontam,
///   FidelityFX CAS sharpness, Optical Bokeh, Studio Light)
pub struct StudioProcessor {
    segmenter: Box<dyn Segmenter>,
    center_stage: Option<CenterStageController>,
    gpu: GpuEffects,
    settings: SharedSettings,
    full_white_mask: AlphaMask,
    perf_frames: u64,
    perf_matting_ns: u128,
    perf_cs_ns: u128,
    perf_gpu_ns: u128,
    perf_window_start: Instant,
}

impl StudioProcessor {
    pub fn new(
        segmenter: Box<dyn Segmenter>,
        center_stage: Option<CenterStageController>,
        gpu: GpuEffects,
        settings: SharedSettings,
    ) -> Self {
        let (mw, mh) = segmenter.mask_dims();
        let pixels = (mw * mh) as usize;
        Self {
            segmenter,
            center_stage,
            gpu,
            settings,
            full_white_mask: AlphaMask {
                width: mw,
                height: mh,
                data: vec![255u8; pixels],
                fgr_rgba: vec![0u8; pixels * 4],
                centroid: [0.5, 0.42],
            },
            perf_frames: 0,
            perf_matting_ns: 0,
            perf_cs_ns: 0,
            perf_gpu_ns: 0,
            perf_window_start: Instant::now(),
        }
    }
}

impl FrameProcessor for StudioProcessor {
    fn on_stream_start(&mut self) {
        self.segmenter.reset();
        self.gpu.reset_temporal();
        if let Some(cs) = self.center_stage.as_mut() {
            cs.reset();
        }
        self.perf_frames = 0;
        self.perf_matting_ns = 0;
        self.perf_cs_ns = 0;
        self.perf_gpu_ns = 0;
        self.perf_window_start = Instant::now();
    }

    fn process(&mut self, frame: &mut Frame) -> Result<()> {
        let current = self.settings.read().unwrap().clone();
        if !current.needs_gpu() {
            return Ok(());
        }

        let t0 = Instant::now();
        let has_matting = current.needs_matting();
        let mask: &AlphaMask = if has_matting {
            self.segmenter
                .infer_alpha(&frame.nv12, frame.width, frame.height)?
        } else {
            &self.full_white_mask
        };
        let t1 = Instant::now();

        let (crop, face_center) = if let Some(cs) = self.center_stage.as_mut() {
            let mask_ref = if has_matting {
                Some((mask.data.as_slice(), mask.width, mask.height))
            } else {
                None
            };
            cs.update(
                &frame.nv12,
                frame.width,
                frame.height,
                mask_ref,
                current.center_stage,
                current.center_stage_zoom,
            )
        } else {
            (CropRect::default(), None)
        };
        let t2 = Instant::now();

        self.gpu.process_nv12(
            &mut frame.nv12,
            mask,
            has_matting,
            face_center,
            crop,
            &current,
        )?;
        let t3 = Instant::now();

        self.perf_frames += 1;
        self.perf_matting_ns += (t1 - t0).as_nanos();
        self.perf_cs_ns += (t2 - t1).as_nanos();
        self.perf_gpu_ns += (t3 - t2).as_nanos();

        if self.perf_window_start.elapsed() >= Duration::from_secs(5) {
            let n = self.perf_frames.max(1) as f64;
            let fps = self.perf_frames as f64 / self.perf_window_start.elapsed().as_secs_f64();
            let cs_ms = (self.perf_cs_ns as f64 / n) / 1_000_000.0;
            let mat_ms = (self.perf_matting_ns as f64 / n) / 1_000_000.0;
            let gpu_ms = (self.perf_gpu_ns as f64 / n) / 1_000_000.0;
            info!(
                "Pipeline perf: {:.1} FPS | center_stage={:.1} ms | matting={:.1} ms | gpu={:.1} ms | total={:.1} ms",
                fps,
                cs_ms,
                mat_ms,
                gpu_ms,
                cs_ms + mat_ms + gpu_ms
            );
            self.perf_frames = 0;
            self.perf_cs_ns = 0;
            self.perf_matting_ns = 0;
            self.perf_gpu_ns = 0;
            self.perf_window_start = Instant::now();
        }

        Ok(())
    }
}
