use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{Context, Result};
use log::info;

use crate::segmenter::{AlphaMask, Segmenter};

pub const RESNET_WIDTH: u32 = 768;
pub const RESNET_HEIGHT: u32 = 432;

#[repr(C)]
struct NcnnNetOpaque {
    _unused: [u8; 0],
}
#[repr(C)]
struct NcnnOptionOpaque {
    _unused: [u8; 0],
}
#[repr(C)]
struct NcnnMatOpaque {
    _unused: [u8; 0],
}
#[repr(C)]
struct NcnnExtractorOpaque {
    _unused: [u8; 0],
}

type NcnnNetT = *mut NcnnNetOpaque;
type NcnnOptionT = *mut NcnnOptionOpaque;
type NcnnMatT = *mut NcnnMatOpaque;
type NcnnExtractorT = *mut NcnnExtractorOpaque;
type NcnnAllocatorT = *mut c_void;

unsafe extern "C" {
    fn ncnn_option_create() -> NcnnOptionT;
    fn ncnn_option_destroy(opt: NcnnOptionT);
    fn ncnn_option_set_num_threads(opt: NcnnOptionT, num_threads: c_int);
    fn ncnn_option_set_use_vulkan_compute(opt: NcnnOptionT, enable: c_int);
    fn ncnn_option_set_use_fp16_packed(opt: NcnnOptionT, enable: c_int);
    fn ncnn_option_set_use_fp16_storage(opt: NcnnOptionT, enable: c_int);
    fn ncnn_option_set_use_fp16_arithmetic(opt: NcnnOptionT, enable: c_int);
    fn ncnn_option_set_use_cooperative_matrix(opt: NcnnOptionT, enable: c_int);

    fn ncnn_net_create() -> NcnnNetT;
    fn ncnn_net_destroy(net: NcnnNetT);
    fn ncnn_net_set_option(net: NcnnNetT, opt: NcnnOptionT);
    fn ncnn_net_set_vulkan_device(net: NcnnNetT, device_index: c_int);
    fn ncnn_net_load_param(net: NcnnNetT, path: *const c_char) -> c_int;
    fn ncnn_net_load_model(net: NcnnNetT, path: *const c_char) -> c_int;

    fn ncnn_mat_create_3d(w: c_int, h: c_int, c: c_int, allocator: NcnnAllocatorT) -> NcnnMatT;
    fn ncnn_mat_destroy(mat: NcnnMatT);
    fn ncnn_mat_fill_float(mat: NcnnMatT, v: f32);
    fn ncnn_mat_get_w(mat: NcnnMatT) -> c_int;
    fn ncnn_mat_get_h(mat: NcnnMatT) -> c_int;
    fn ncnn_mat_get_c(mat: NcnnMatT) -> c_int;
    fn ncnn_mat_get_channel_data(mat: NcnnMatT, c: c_int) -> *mut c_void;

    fn ncnn_extractor_create(net: NcnnNetT) -> NcnnExtractorT;
    fn ncnn_extractor_destroy(ex: NcnnExtractorT);
    fn ncnn_extractor_input(ex: NcnnExtractorT, name: *const c_char, mat: NcnnMatT) -> c_int;
    fn ncnn_extractor_extract(
        ex: NcnnExtractorT,
        name: *const c_char,
        mat: *mut NcnnMatT,
    ) -> c_int;
}

struct OwnedMat(NcnnMatT);

// SAFETY: NcnnRvmSegmenter is exclusively owned by the single capture worker thread.
unsafe impl Send for OwnedMat {}

impl OwnedMat {
    fn zeros_3d(w: usize, h: usize, c: usize) -> Self {
        let mat = unsafe { ncnn_mat_create_3d(w as c_int, h as c_int, c as c_int, ptr::null_mut()) };
        assert!(!mat.is_null(), "ncnn_mat_create_3d returned null");
        unsafe { ncnn_mat_fill_float(mat, 0.0) };
        Self(mat)
    }

    fn channel_slice(&self, ch: usize) -> &[f32] {
        let w = unsafe { ncnn_mat_get_w(self.0) } as usize;
        let h = unsafe { ncnn_mat_get_h(self.0) } as usize;
        let ptr = unsafe { ncnn_mat_get_channel_data(self.0, ch as c_int) as *const f32 };
        unsafe { std::slice::from_raw_parts(ptr, w * h) }
    }
}

impl Drop for OwnedMat {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ncnn_mat_destroy(self.0) };
            self.0 = ptr::null_mut();
        }
    }
}

/// Robust Video Matting (ResNet50, 768x432) accelerated on Vulkan GPU via `ncnn` FP16 compute!
pub struct NcnnRvmSegmenter {
    net: NcnnNetT,
    opt: NcnnOptionT,
    in0: OwnedMat,
    r1: OwnedMat,
    r2: OwnedMat,
    r3: OwnedMat,
    r4: OwnedMat,
    mask: AlphaMask,
    centroid_initialized: bool,
}

// SAFETY: Exclusively accessed by one capture worker thread (locked behind Mutex<Box<dyn FrameProcessor>>).
unsafe impl Send for NcnnRvmSegmenter {}

impl Drop for NcnnRvmSegmenter {
    fn drop(&mut self) {
        if !self.net.is_null() {
            unsafe { ncnn_net_destroy(self.net) };
        }
        if !self.opt.is_null() {
            unsafe { ncnn_option_destroy(self.opt) };
        }
    }
}

impl NcnnRvmSegmenter {
    pub fn new(param_path: &Path, bin_path: &Path, use_vulkan: bool) -> Result<Self> {
        let param_c = CString::new(
            param_path
                .to_str()
                .context("non-UTF8 ncnn param path")?,
        )?;
        let bin_c = CString::new(
            bin_path
                .to_str()
                .context("non-UTF8 ncnn bin path")?,
        )?;

        let opt = unsafe { ncnn_option_create() };
        anyhow::ensure!(!opt.is_null(), "ncnn_option_create failed");
        unsafe {
            ncnn_option_set_num_threads(opt, 4);
            ncnn_option_set_use_vulkan_compute(opt, if use_vulkan { 1 } else { 0 });
            ncnn_option_set_use_fp16_packed(opt, 1);
            ncnn_option_set_use_fp16_storage(opt, 1);
            ncnn_option_set_use_fp16_arithmetic(opt, 1);
            ncnn_option_set_use_cooperative_matrix(opt, if use_vulkan { 1 } else { 0 });
        }

        let net = unsafe { ncnn_net_create() };
        anyhow::ensure!(!net.is_null(), "ncnn_net_create failed");
        unsafe {
            ncnn_net_set_option(net, opt);
            if use_vulkan {
                ncnn_net_set_vulkan_device(net, 0);
            }
        }

        let rc_param = unsafe { ncnn_net_load_param(net, param_c.as_ptr()) };
        if rc_param != 0 {
            unsafe {
                ncnn_net_destroy(net);
                ncnn_option_destroy(opt);
            }
            anyhow::bail!(
                "ncnn_net_load_param failed ({rc_param}) for {}",
                param_path.display()
            );
        }

        let rc_model = unsafe { ncnn_net_load_model(net, bin_c.as_ptr()) };
        if rc_model != 0 {
            unsafe {
                ncnn_net_destroy(net);
                ncnn_option_destroy(opt);
            }
            anyhow::bail!(
                "ncnn_net_load_model failed ({rc_model}) for {}",
                bin_path.display()
            );
        }

        info!(
            "Loaded NCNN RVM ResNet50 ({}x{}, vulkan={}, fp16=true) from {}",
            RESNET_WIDTH,
            RESNET_HEIGHT,
            use_vulkan,
            param_path.display()
        );

        let pixels = (RESNET_WIDTH * RESNET_HEIGHT) as usize;
        let mut seg = Self {
            net,
            opt,
            in0: OwnedMat::zeros_3d(RESNET_WIDTH as usize, RESNET_HEIGHT as usize, 3),
            r1: OwnedMat::zeros_3d(192, 108, 16),
            r2: OwnedMat::zeros_3d(96, 54, 32),
            r3: OwnedMat::zeros_3d(48, 27, 64),
            r4: OwnedMat::zeros_3d(24, 14, 128),
            mask: AlphaMask {
                width: RESNET_WIDTH,
                height: RESNET_HEIGHT,
                data: vec![255u8; pixels],
                fgr_rgba: vec![0u8; pixels * 4],
                centroid: [0.5, 0.42],
            },
            centroid_initialized: false,
        };

        // Pre-compile Vulkan SPIR-V pipelines once on startup so first call frame is instant.
        if use_vulkan {
            let dummy_nv12 = vec![128u8; 64 * 36 * 3 / 2];
            if let Err(err) = seg.infer_alpha(&dummy_nv12, 64, 36) {
                log::warn!("NCNN Vulkan warmup skipped: {err:#}");
            }
            seg.reset();
        }

        Ok(seg)
    }

    /// Downsample NV12 (Rec. 709 limited) directly into NCNN 3-channel RGB float32 Mat [0..1].
    fn fill_input_from_nv12(&mut self, nv12: &[u8], src_w: u32, src_h: u32) {
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = RESNET_WIDTH as usize;
        let dh = RESNET_HEIGHT as usize;
        let uv_offset = sw * sh;

        // SAFETY: channels 0, 1, 2 are disjoint planes in NCNN Mat memory.
        let r_ptr = unsafe { ncnn_mat_get_channel_data(self.in0.0, 0) as *mut f32 };
        let g_ptr = unsafe { ncnn_mat_get_channel_data(self.in0.0, 1) as *mut f32 };
        let b_ptr = unsafe { ncnn_mat_get_channel_data(self.in0.0, 2) as *mut f32 };
        let r_plane = unsafe { std::slice::from_raw_parts_mut(r_ptr, dw * dh) };
        let g_plane = unsafe { std::slice::from_raw_parts_mut(g_ptr, dw * dh) };
        let b_plane = unsafe { std::slice::from_raw_parts_mut(b_ptr, dw * dh) };

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

impl Segmenter for NcnnRvmSegmenter {
    fn mask_dims(&self) -> (u32, u32) {
        (RESNET_WIDTH, RESNET_HEIGHT)
    }

    fn reset(&mut self) {
        self.r1 = OwnedMat::zeros_3d(192, 108, 16);
        self.r2 = OwnedMat::zeros_3d(96, 54, 32);
        self.r3 = OwnedMat::zeros_3d(48, 27, 64);
        self.r4 = OwnedMat::zeros_3d(24, 14, 128);
        self.centroid_initialized = false;
    }

    fn infer_alpha(&mut self, nv12: &[u8], src_w: u32, src_h: u32) -> Result<&AlphaMask> {
        self.fill_input_from_nv12(nv12, src_w, src_h);

        let ex = unsafe { ncnn_extractor_create(self.net) };
        anyhow::ensure!(!ex.is_null(), "ncnn_extractor_create failed");

        let run_result = (|| -> Result<(OwnedMat, OwnedMat, OwnedMat, OwnedMat, OwnedMat, OwnedMat)> {
            unsafe {
                anyhow::ensure!(ncnn_extractor_input(ex, c"in0".as_ptr(), self.in0.0) == 0, "in0 failed");
                anyhow::ensure!(ncnn_extractor_input(ex, c"in1".as_ptr(), self.r1.0) == 0, "in1 failed");
                anyhow::ensure!(ncnn_extractor_input(ex, c"in2".as_ptr(), self.r2.0) == 0, "in2 failed");
                anyhow::ensure!(ncnn_extractor_input(ex, c"in3".as_ptr(), self.r3.0) == 0, "in3 failed");
                anyhow::ensure!(ncnn_extractor_input(ex, c"in4".as_ptr(), self.r4.0) == 0, "in4 failed");

                let mut out0: NcnnMatT = ptr::null_mut();
                let mut out1: NcnnMatT = ptr::null_mut();
                let mut out2: NcnnMatT = ptr::null_mut();
                let mut out3: NcnnMatT = ptr::null_mut();
                let mut out4: NcnnMatT = ptr::null_mut();
                let mut out5: NcnnMatT = ptr::null_mut();

                anyhow::ensure!(ncnn_extractor_extract(ex, c"out0".as_ptr(), &mut out0) == 0, "out0 failed");
                anyhow::ensure!(ncnn_extractor_extract(ex, c"out1".as_ptr(), &mut out1) == 0, "out1 failed");
                anyhow::ensure!(ncnn_extractor_extract(ex, c"out2".as_ptr(), &mut out2) == 0, "out2 failed");
                anyhow::ensure!(ncnn_extractor_extract(ex, c"out3".as_ptr(), &mut out3) == 0, "out3 failed");
                anyhow::ensure!(ncnn_extractor_extract(ex, c"out4".as_ptr(), &mut out4) == 0, "out4 failed");
                anyhow::ensure!(ncnn_extractor_extract(ex, c"out5".as_ptr(), &mut out5) == 0, "out5 failed");

                Ok((
                    OwnedMat(out0),
                    OwnedMat(out1),
                    OwnedMat(out2),
                    OwnedMat(out3),
                    OwnedMat(out4),
                    OwnedMat(out5),
                ))
            }
        })();

        unsafe { ncnn_extractor_destroy(ex) };
        let (fgr_mat, pha_mat, r1o, r2o, r3o, r4o) = run_result?;

        let dw = RESNET_WIDTH as usize;
        let dh = RESNET_HEIGHT as usize;
        let pixels = dw * dh;

        let pha_slice = pha_mat.channel_slice(0);
        let fgr_c = unsafe { ncnn_mat_get_c(fgr_mat.0) };
        let has_rgb = fgr_c >= 3;
        let (r_s, g_s, b_s) = if has_rgb {
            (
                fgr_mat.channel_slice(0),
                fgr_mat.channel_slice(1),
                fgr_mat.channel_slice(2),
            )
        } else {
            (&[][..], &[][..], &[][..])
        };

        let mut sum_w = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;

        for idx in 0..pixels.min(pha_slice.len()) {
            let a = pha_slice[idx].clamp(0.0, 1.0);
            self.mask.data[idx] = (a * 255.0 + 0.5) as u8;

            if has_rgb && idx < r_s.len() {
                let off = idx * 4;
                self.mask.fgr_rgba[off] = (r_s[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 1] = (g_s[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 2] = (b_s[idx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                self.mask.fgr_rgba[off + 3] = 255;
            }

            if idx & 7 == 0 && a > 0.35 {
                let x = (idx % dw) as f64 / dw as f64;
                let y = (idx / dw) as f64 / dh as f64;
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
                let smooth = 0.12f32;
                self.mask.centroid[0] += (target_x - self.mask.centroid[0]) * smooth;
                self.mask.centroid[1] += (target_y - self.mask.centroid[1]) * smooth;
            }
        }

        // Persist recurrent states directly as NCNN Mats for the next frame
        self.r1 = r1o;
        self.r2 = r2o;
        self.r3 = r3o;
        self.r4 = r4o;

        Ok(&self.mask)
    }
}
