use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use image::imageops::FilterType;
use log::{info, warn};

use crate::center_stage::CropRect;
use crate::segmenter::AlphaMask;
use crate::settings::{BackgroundMode, EffectSettings};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    bg_mode: u32,
    frame_index: u32,
    blur_strength: f32,
    studio_light: f32,
    bg_image_blur: f32,
    sharpness: f32,
    centroid: [f32; 2],
    mask_size: [f32; 2],
    bg_color: [f32; 4],
    crop: [f32; 4], // (cx, cy, zoom, has_fgr_decontam)
}

/// Headless Vulkan compute pipeline (`wgpu`) performing:
/// 1A. NV12 -> RGBA8 (Rec. 709 limited)
/// 1B. Motion-Adaptive Temporal GPU Denoiser (history accumulation, zero ghosting)
/// 1C/1D. 2-stage 8-channel Residual Restoration CNN (Rgba16Float features -> micro-edge restoration)
/// 2. Half-res foreground-masked background prep
/// 3. 48-tap Golden-Angle disk bokeh blur (halo-free)
/// 4. Center Stage Catmull-Rom zoom + Edge-Guided alpha refinement + RVM fgr decontam +
///    FidelityFX CAS + De-haze + Studio Light -> packed NV12
pub struct GpuEffects {
    width: u32,
    height: u32,
    mask_width: u32,
    mask_height: u32,
    frame_index: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,

    uniform_buf: wgpu::Buffer,
    y_tex: wgpu::Texture,
    uv_tex: wgpu::Texture,
    alpha_tex: wgpu::Texture,
    fgr_tex: wgpu::Texture,
    bg_img_tex: wgpu::Texture,
    clean_rgba_tex: wgpu::Texture,
    prev_clean_tex: wgpu::Texture,
    nv12_out_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,

    p_nv12_to_rgba: wgpu::ComputePipeline,
    p_temporal_denoise: wgpu::ComputePipeline,
    p_rl_ratio1: wgpu::ComputePipeline,
    p_rl_update1: wgpu::ComputePipeline,
    p_rl_ratio2: wgpu::ComputePipeline,
    p_rl_update2: wgpu::ComputePipeline,
    p_prepare_half: wgpu::ComputePipeline,
    p_bokeh: wgpu::ComputePipeline,
    p_compose: wgpu::ComputePipeline,

    bg_nv12_to_rgba: wgpu::BindGroup,
    bg_temporal_denoise: wgpu::BindGroup,
    bg_rl_ratio1: wgpu::BindGroup,
    bg_rl_update1: wgpu::BindGroup,
    bg_rl_ratio2: wgpu::BindGroup,
    bg_rl_update2: wgpu::BindGroup,
    bg_prepare_half: wgpu::BindGroup,
    bg_bokeh: wgpu::BindGroup,
    bg_compose: wgpu::BindGroup,

    loaded_bg_path: Option<PathBuf>,
    has_bg_image: bool,
}

impl GpuEffects {
    pub fn new(width: u32, height: u32, mask_width: u32, mask_height: u32) -> Result<Self> {
        anyhow::ensure!(
            width % 4 == 0 && height % 2 == 0,
            "GpuEffects requires width % 4 == 0 and height % 2 == 0, got {width}x{height}"
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .context("failed to find a Vulkan/GPU adapter for wgpu")?;

        let adapter_info = adapter.get_info();
        info!(
            "GPU effects initialized on {} ({:?}, mask={}x{})",
            adapter_info.name, adapter_info.backend, mask_width, mask_height
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("omacam-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .context("failed to create wgpu logical device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("omacam-effects-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/effects.wgsl").into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("omacam-linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omacam-uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let y_tex = create_sampled_tex(&device, "y_tex", width, height, wgpu::TextureFormat::R8Unorm);
        let uv_tex = create_sampled_tex(
            &device,
            "uv_tex",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rg8Unorm,
        );
        let alpha_tex = create_sampled_tex(
            &device,
            "alpha_tex",
            mask_width,
            mask_height,
            wgpu::TextureFormat::R8Unorm,
        );
        let fgr_tex = create_sampled_tex(
            &device,
            "fgr_tex",
            mask_width,
            mask_height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let bg_img_tex = create_sampled_tex(
            &device,
            "bg_img_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let src_rgba_tex = create_storage_sampled_tex(
            &device,
            "src_rgba_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );
        let clean_rgba_tex = create_storage_sampled_tex(
            &device,
            "clean_rgba_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_SRC,
        );
        let prev_clean_tex = create_sampled_tex(
            &device,
            "prev_clean_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let feat0_tex = create_storage_sampled_tex(
            &device,
            "feat0_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::empty(),
        );
        let feat1_tex = create_storage_sampled_tex(
            &device,
            "feat1_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::empty(),
        );
        let restored_rgba_tex = create_storage_sampled_tex(
            &device,
            "restored_rgba_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );
        let half_bg_tex = create_storage_sampled_tex(
            &device,
            "half_bg_tex",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );
        let bokeh_tex = create_storage_sampled_tex(
            &device,
            "bokeh_tex",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );

        let nv12_bytes = (width as u64) * (height as u64) * 3 / 2;
        let nv12_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_out"),
            size: nv12_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_readback"),
            size: nv12_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let p_nv12_to_rgba = create_pipeline(&device, &shader, "nv12_to_rgba");
        let p_temporal_denoise = create_pipeline(&device, &shader, "temporal_denoise");
        let p_rl_ratio1 = create_pipeline(&device, &shader, "rl_ratio1");
        let p_rl_update1 = create_pipeline(&device, &shader, "rl_update1");
        let p_rl_ratio2 = create_pipeline(&device, &shader, "rl_ratio2");
        let p_rl_update2 = create_pipeline(&device, &shader, "rl_update2");
        let p_prepare_half = create_pipeline(&device, &shader, "prepare_half_bg");
        let p_bokeh = create_pipeline(&device, &shader, "bokeh_blur");
        let p_compose = create_pipeline(&device, &shader, "compose_nv12");

        let y_view = y_tex.create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let alpha_view = alpha_tex.create_view(&Default::default());
        let fgr_view = fgr_tex.create_view(&Default::default());
        let bg_img_view = bg_img_tex.create_view(&Default::default());
        let src_rgba_view = src_rgba_tex.create_view(&Default::default());
        let clean_rgba_view = clean_rgba_tex.create_view(&Default::default());
        let prev_clean_view = prev_clean_tex.create_view(&Default::default());
        let feat0_view = feat0_tex.create_view(&Default::default());
        let feat1_view = feat1_tex.create_view(&Default::default());
        let restored_rgba_view = restored_rgba_tex.create_view(&Default::default());
        let half_bg_view = half_bg_tex.create_view(&Default::default());
        let bokeh_view = bokeh_tex.create_view(&Default::default());

        let bg_nv12_to_rgba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-nv12-to-rgba"),
            layout: &p_nv12_to_rgba.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&src_rgba_view),
                },
            ],
        });

        let bg_temporal_denoise = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-temporal-denoise"),
            layout: &p_temporal_denoise.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&src_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&prev_clean_view),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
            ],
        });

        let bg_rl_ratio1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-rl-ratio1"),
            layout: &p_rl_ratio1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&feat0_view),
                },
            ],
        });

        let bg_rl_update1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-rl-update1"),
            layout: &p_rl_update1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&feat1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::TextureView(&feat0_view),
                },
            ],
        });

        let bg_rl_ratio2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-rl-ratio2"),
            layout: &p_rl_ratio2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&feat0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(&feat1_view),
                },
            ],
        });

        let bg_rl_update2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-rl-update2"),
            layout: &p_rl_update2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::TextureView(&feat0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(&feat1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(&restored_rgba_view),
                },
            ],
        });

        let bg_prepare_half = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-prepare-half"),
            layout: &p_prepare_half.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&bg_img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&alpha_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&half_bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
            ],
        });

        let bg_bokeh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-bokeh"),
            layout: &p_bokeh.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&half_bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&bokeh_view),
                },
            ],
        });

        let bg_compose = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-compose"),
            layout: &p_compose.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&bg_img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&alpha_view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&bokeh_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: nv12_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&fgr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clean_rgba_view),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: wgpu::BindingResource::TextureView(&restored_rgba_view),
                },
            ],
        });

        Ok(Self {
            width,
            height,
            mask_width,
            mask_height,
            frame_index: 0,
            device,
            queue,
            uniform_buf,
            y_tex,
            uv_tex,
            alpha_tex,
            fgr_tex,
            bg_img_tex,
            clean_rgba_tex,
            prev_clean_tex,
            nv12_out_buf,
            readback_buf,
            p_nv12_to_rgba,
            p_temporal_denoise,
            p_rl_ratio1,
            p_rl_update1,
            p_rl_ratio2,
            p_rl_update2,
            p_prepare_half,
            p_bokeh,
            p_compose,
            bg_nv12_to_rgba,
            bg_temporal_denoise,
            bg_rl_ratio1,
            bg_rl_update1,
            bg_rl_ratio2,
            bg_rl_update2,
            bg_prepare_half,
            bg_bokeh,
            bg_compose,
            loaded_bg_path: None,
            has_bg_image: false,
        })
    }

    pub fn reset_temporal(&mut self) {
        self.frame_index = 0;
    }

    pub fn ensure_background_image(&mut self, path: Option<&Path>) {
        if self.loaded_bg_path.as_deref() == path {
            return;
        }
        self.loaded_bg_path = path.map(Path::to_path_buf);
        self.has_bg_image = false;

        let Some(p) = path else {
            return;
        };

        match image::open(p) {
            Ok(img) => {
                let resized = img
                    .resize_to_fill(self.width, self.height, FilterType::Lanczos3)
                    .to_rgba8();
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.bg_img_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    resized.as_raw(),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.width * 4),
                        rows_per_image: Some(self.height),
                    },
                    wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                );
                self.has_bg_image = true;
                info!("Loaded background image {}", p.display());
            }
            Err(err) => {
                warn!("Failed to load background image {}: {err}", p.display());
            }
        }
    }

    pub fn process_nv12(
        &mut self,
        nv12: &mut [u8],
        mask: &AlphaMask,
        has_fgr: bool,
        centroid_override: Option<[f32; 2]>,
        crop: CropRect,
        settings: &EffectSettings,
    ) -> Result<()> {
        let y_len = (self.width as usize) * (self.height as usize);
        let uv_len = y_len / 2;
        anyhow::ensure!(
            nv12.len() >= y_len + uv_len,
            "NV12 buffer too small: {} < {}",
            nv12.len(),
            y_len + uv_len
        );

        self.ensure_background_image(settings.bg_image_path.as_deref());

        let effective_bg_mode = match settings.background_mode {
            BackgroundMode::None => 0u32,
            BackgroundMode::Blur => 1u32,
            BackgroundMode::Image if self.has_bg_image => 2u32,
            BackgroundMode::Image => 3u32,
            BackgroundMode::Color => 3u32,
        };

        let uniforms = Uniforms {
            width: self.width,
            height: self.height,
            bg_mode: effective_bg_mode,
            frame_index: self.frame_index,
            blur_strength: settings.blur_strength.clamp(0.0, 1.0),
            studio_light: settings.studio_light.clamp(0.0, 1.0),
            bg_image_blur: settings.bg_image_blur.clamp(0.0, 1.0),
            sharpness: settings.sharpness.clamp(0.0, 1.0),
            centroid: centroid_override.unwrap_or(mask.centroid),
            mask_size: [self.mask_width as f32, self.mask_height as f32],
            bg_color: [
                settings.bg_color[0],
                settings.bg_color[1],
                settings.bg_color[2],
                1.0,
            ],
            crop: [crop.cx, crop.cy, crop.zoom, if has_fgr { 1.0 } else { 0.0 }],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Upload Y plane
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.y_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &nv12[..y_len],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Upload interleaved UV plane
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &nv12[y_len..y_len + uv_len],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(self.height / 2),
            },
            wgpu::Extent3d {
                width: self.width / 2,
                height: self.height / 2,
                depth_or_array_layers: 1,
            },
        );

        // Upload alpha matte (768x432 ResNet50 Vulkan or 512x288 MobileNetV3)
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.alpha_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mask.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mask.width),
                rows_per_image: Some(mask.height),
            },
            wgpu::Extent3d {
                width: mask.width.min(self.mask_width),
                height: mask.height.min(self.mask_height),
                depth_or_array_layers: 1,
            },
        );

        if has_fgr && mask.fgr_rgba.len() >= (mask.width * mask.height * 4) as usize {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.fgr_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mask.fgr_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mask.width * 4),
                    rows_per_image: Some(mask.height),
                },
                wgpu::Extent3d {
                    width: mask.width.min(self.mask_width),
                    height: mask.height.min(self.mask_height),
                    depth_or_array_layers: 1,
                },
            );
        }

        let needs_bokeh = (effective_bg_mode == 1 && settings.blur_strength > 0.01)
            || (effective_bg_mode == 2 && settings.bg_image_blur > 0.01);

        let wg_x = self.width.div_ceil(16);
        let wg_y = self.height.div_ceil(16);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("omacam-frame-encoder"),
            });

        // Pass 1A: NV12 -> RGBA8
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12-to-rgba"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_nv12_to_rgba);
            pass.set_bind_group(0, &self.bg_nv12_to_rgba, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 1B: Motion-Adaptive Temporal GPU Denoiser
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temporal-denoise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_temporal_denoise);
            pass.set_bind_group(0, &self.bg_temporal_denoise, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Save clean_rgba into prev_clean_tex for next frame's temporal denoiser
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.clean_rgba_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.prev_clean_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Pass 1C..1F: 2-Iteration Richardson-Lucy Optical Deconvolution
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rl-ratio1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_rl_ratio1);
            pass.set_bind_group(0, &self.bg_rl_ratio1, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rl-update1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_rl_update1);
            pass.set_bind_group(0, &self.bg_rl_update1, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rl-ratio2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_rl_ratio2);
            pass.set_bind_group(0, &self.bg_rl_ratio2, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rl-update2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_rl_update2);
            pass.set_bind_group(0, &self.bg_rl_update2, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2 & 3: Optical Bokeh blur
        if needs_bokeh {
            let hw = self.width / 2;
            let hh = self.height / 2;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prepare-half-bg"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.p_prepare_half);
                pass.set_bind_group(0, &self.bg_prepare_half, &[]);
                pass.dispatch_workgroups(hw.div_ceil(16), hh.div_ceil(16), 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bokeh-blur"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.p_bokeh);
                pass.set_bind_group(0, &self.bg_bokeh, &[]);
                pass.dispatch_workgroups(hw.div_ceil(16), hh.div_ceil(16), 1);
            }
        }

        // Pass 4: Composite + Center Stage + CAS + Studio Light + NV12 pack
        {
            let words_per_row = self.width / 4;
            let uv_rows = self.height / 2;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compose-nv12"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_compose);
            pass.set_bind_group(0, &self.bg_compose, &[]);
            pass.dispatch_workgroups(words_per_row.div_ceil(16), uv_rows.div_ceil(16), 1);
        }

        let copy_size = (y_len + uv_len) as u64;
        encoder.copy_buffer_to_buffer(&self.nv12_out_buf, 0, &self.readback_buf, 0, copy_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(..copy_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .context("GPU readback channel closed")?
            .context("failed to map GPU readback buffer")?;

        {
            let mapped = slice
                .get_mapped_range()
                .context("failed to map GPU readback slice")?;
            nv12[..y_len + uv_len].copy_from_slice(&mapped);
        }
        self.readback_buf.unmap();
        self.frame_index = self.frame_index.saturating_add(1);

        Ok(())
    }
}

fn create_sampled_tex(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_storage_sampled_tex(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    extra_usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra_usage,
        view_formats: &[],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}
