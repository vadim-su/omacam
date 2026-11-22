mod capture;
mod center_stage;
mod gpu;
mod ipc;
mod ncnn_segmenter;
mod pipeline;
mod pw_camera;
mod segmenter;
mod settings;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use log::{info, warn};

use crate::capture::{AURORA_DENOISE_NODE, CaptureConfig};
use crate::center_stage::{CenterStageController, YuNetDetector};
use crate::gpu::GpuEffects;
use crate::ipc::{IpcHub, IpcRequest, run_watch_bridge, send_one_shot};
use crate::ncnn_segmenter::NcnnRvmSegmenter;
use crate::pipeline::{FrameProcessor, PassthroughProcessor, StudioProcessor};
use crate::pw_camera::{PwEvent, VirtualCameraConfig, run_virtual_camera};
use crate::segmenter::{RvmSegmenter, Segmenter};
use crate::settings::{BackgroundMode, EffectSettings, SettingsPatch};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CliBgMode {
    None,
    Blur,
    Image,
    Color,
}

impl From<CliBgMode> for BackgroundMode {
    fn from(m: CliBgMode) -> Self {
        match m {
            CliBgMode::None => BackgroundMode::None,
            CliBgMode::Blur => BackgroundMode::Blur,
            CliBgMode::Image => BackgroundMode::Image,
            CliBgMode::Color => BackgroundMode::Color,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MattingBackend {
    /// Prefer NCNN Vulkan RVM ResNet50 (768x432), fall back to ONNX Runtime CPU MobileNetV3
    Auto,
    /// Force NCNN Vulkan RVM ResNet50 (768x432 FP16)
    Ncnn,
    /// Force ONNX Runtime CPU RVM MobileNetV3 (512x288)
    Ort,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CaptureSource {
    /// PipeWire `aurora_camera_denoise` (with automatic V4L2 fallback)
    Aurora,
    /// Direct V4L2 device capture (`/dev/video0`)
    V4l2,
}

#[derive(Parser, Debug)]
#[command(
    name = "omacam",
    version,
    about = "macOS-style virtual camera daemon with Vulkan RVM ResNet50, Center Stage, Temporal Denoise & Residual CNN"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Capture source mode (PipeWire aurora-camera-denoise or direct V4L2)
    #[arg(long, value_enum, default_value_t = CaptureSource::Aurora)]
    source: CaptureSource,

    /// Physical V4L2 camera device path (used directly with --source v4l2 or as fallback)
    #[arg(short, long, default_value = "/dev/video0")]
    device: PathBuf,

    /// Matting inference backend
    #[arg(long, value_enum, default_value_t = MattingBackend::Auto)]
    backend: MattingBackend,

    /// Virtual camera name shown in Brave/Chrome/Discord and PipeWire
    #[arg(long, default_value = "Omacam Studio")]
    name: String,

    /// Internal PipeWire node.name
    #[arg(long, default_value = "omacam")]
    node_name: String,

    /// Capture and output width
    #[arg(long, default_value_t = 1920)]
    width: u32,

    /// Capture and output height
    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// Capture and output framerate
    #[arg(long, default_value_t = 30)]
    fps: u32,

    /// Path to RVM ONNX model (for --backend ort fallback)
    #[arg(long)]
    model: Option<PathBuf>,

    /// Path to YuNet face detector ONNX model (auto-detected for Center Stage)
    #[arg(long)]
    face_model: Option<PathBuf>,

    /// Override background effect mode on startup
    #[arg(long, value_enum)]
    bg: Option<CliBgMode>,

    /// Override portrait bokeh blur intensity (0.0..=1.0)
    #[arg(long)]
    blur: Option<f32>,

    /// Override Studio Light intensity (0.0..=1.0)
    #[arg(long)]
    studio_light: Option<f32>,

    /// Override foreground neural restoration + CAS sharpness (0.0..=1.0)
    #[arg(long)]
    sharpness: Option<f32>,

    /// Enable/disable Center Stage face tracking on startup
    #[arg(long)]
    center_stage: Option<bool>,

    /// Override Center Stage max zoom preference (0.0..=1.0)
    #[arg(long)]
    center_stage_zoom: Option<f32>,

    /// Override replacement background image path
    #[arg(long)]
    bg_image: Option<PathBuf>,

    /// Override bokeh blur on replacement background image (0.0..=1.0)
    #[arg(long)]
    bg_image_blur: Option<f32>,

    /// Override background studio color in hex (e.g. #1a2233)
    #[arg(long)]
    bg_color: Option<String>,

    /// RVM ONNX CPU internal downsample ratio (only for --backend ort)
    #[arg(long, default_value_t = 0.55)]
    downsample_ratio: f32,

    /// Number of CPU threads for ONNX Runtime
    #[arg(long, default_value_t = 6)]
    threads: usize,

    /// Disable AI/GPU effects and pass raw camera through
    #[arg(long)]
    passthrough: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Control a running omacam daemon over Unix socket IPC
    Ctl {
        #[command(subcommand)]
        action: CtlAction,
    },
}

#[derive(Subcommand, Debug)]
enum CtlAction {
    /// Print current daemon & effect state as JSON
    Status,
    /// Toggle effects on/off
    Toggle,
    /// Persistent bidirectional JSON bridge for Quickshell / bar widgets
    Watch,
    /// Update one or more effect settings live
    Set {
        #[arg(long)]
        enabled: Option<bool>,
        #[arg(long, value_enum)]
        bg: Option<CliBgMode>,
        #[arg(long)]
        blur: Option<f32>,
        #[arg(long)]
        studio_light: Option<f32>,
        #[arg(long)]
        sharpness: Option<f32>,
        #[arg(long)]
        center_stage: Option<bool>,
        #[arg(long)]
        center_stage_zoom: Option<f32>,
        #[arg(long)]
        bg_image_blur: Option<f32>,
        #[arg(long)]
        bg_image: Option<PathBuf>,
        #[arg(long)]
        clear_bg_image: bool,
        #[arg(long)]
        bg_color: Option<String>,
    },
}

fn resolve_model_file(explicit: Option<PathBuf>, filename: &str) -> Result<PathBuf> {
    if let Some(p) = explicit {
        anyhow::ensure!(p.exists(), "model not found at {}", p.display());
        return Ok(p);
    }

    let mut candidates = vec![PathBuf::from("models").join(filename)];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
            candidates.push(root.join("models").join(filename));
        }
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("models").join(filename));
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        candidates.push(
            PathBuf::from(home)
                .join(".local/share/omacam/models")
                .join(filename),
        );
    }

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    anyhow::bail!("Could not find {filename}. Checked: {candidates:?}");
}

fn run_ctl(action: CtlAction) -> Result<()> {
    match action {
        CtlAction::Status => {
            let snap = send_one_shot(&IpcRequest::Get)?;
            println!("{}", serde_json::to_string(&snap)?);
        }
        CtlAction::Toggle => {
            let snap = send_one_shot(&IpcRequest::Toggle)?;
            println!("{}", serde_json::to_string(&snap)?);
        }
        CtlAction::Watch => {
            run_watch_bridge()?;
        }
        CtlAction::Set {
            enabled,
            bg,
            blur,
            studio_light,
            sharpness,
            center_stage,
            center_stage_zoom,
            bg_image_blur,
            bg_image,
            clear_bg_image,
            bg_color,
        } => {
            let bg_image_path = if clear_bg_image {
                Some(None)
            } else {
                bg_image.map(Some)
            };
            let patch = SettingsPatch {
                enabled,
                background_mode: bg.map(Into::into),
                blur_strength: blur,
                studio_light,
                sharpness,
                center_stage,
                center_stage_zoom,
                bg_image_blur,
                bg_image_path,
                bg_color_hex: bg_color,
            };
            let snap = send_one_shot(&IpcRequest::Set {
                patch,
                persist: true,
            })?;
            println!("{}", serde_json::to_string(&snap)?);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(Command::Ctl { action }) = args.command {
        return run_ctl(action);
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut initial_settings = EffectSettings::load_or_default();
    if args.passthrough {
        initial_settings.enabled = false;
    }
    if let Some(bg) = args.bg {
        initial_settings.background_mode = bg.into();
    }
    if let Some(blur) = args.blur {
        initial_settings.blur_strength = blur.clamp(0.0, 1.0);
    }
    if let Some(sl) = args.studio_light {
        initial_settings.studio_light = sl.clamp(0.0, 1.0);
    }
    if let Some(sh) = args.sharpness {
        initial_settings.sharpness = sh.clamp(0.0, 1.0);
    }
    if let Some(cs) = args.center_stage {
        initial_settings.center_stage = cs;
    }
    if let Some(csz) = args.center_stage_zoom {
        initial_settings.center_stage_zoom = csz.clamp(0.0, 1.0);
    }
    if let Some(img_blur) = args.bg_image_blur {
        initial_settings.bg_image_blur = img_blur.clamp(0.0, 1.0);
    }
    if let Some(img) = args.bg_image {
        initial_settings.bg_image_path = Some(img);
        if args.bg.is_none() {
            initial_settings.background_mode = BackgroundMode::Image;
        }
    }
    if let Some(hex) = args.bg_color {
        initial_settings.bg_color =
            EffectSettings::parse_hex_color(&hex).context("invalid --bg-color value")?;
    }

    info!(
        "Starting omacam (source={:?}, backend={:?}, output='{}', mode={:?}, blur={:.2}, studio_light={:.2}, sharpness={:.2}, center_stage={})",
        args.source,
        args.backend,
        args.name,
        initial_settings.background_mode,
        initial_settings.blur_strength,
        initial_settings.studio_light,
        initial_settings.sharpness,
        initial_settings.center_stage
    );

    let shared_settings = Arc::new(RwLock::new(initial_settings));
    let ipc_hub = IpcHub::new(Arc::clone(&shared_settings));
    ipc_hub.start_server()?;

    let processor: Box<dyn FrameProcessor> = if args.passthrough {
        info!("Running in raw passthrough mode (AI/GPU disabled)");
        Box::new(PassthroughProcessor)
    } else {
        let ort_env = ort::init()
            .with_name("omacam")
            .build()
            .map_err(|e| anyhow::anyhow!("failed to initialize ONNX Runtime environment: {e}"))?;

        let segmenter: Box<dyn Segmenter> = match args.backend {
            MattingBackend::Ncnn => {
                let param =
                    resolve_model_file(None, "rvm_resnet50_768x432_static.ncnn.param")?;
                let bin = resolve_model_file(None, "rvm_resnet50_768x432_static.ncnn.bin")?;
                Box::new(NcnnRvmSegmenter::new(&param, &bin, true)?)
            }
            MattingBackend::Ort => {
                let model_path = resolve_model_file(args.model, "rvm_mobilenetv3_fp32.onnx")?;
                Box::new(RvmSegmenter::new(
                    &ort_env,
                    &model_path,
                    args.downsample_ratio.clamp(0.25, 1.0),
                    args.threads.max(1),
                )?)
            }
            MattingBackend::Auto => {
                let ncnn_try = (|| -> Result<Box<dyn Segmenter>> {
                    let param =
                        resolve_model_file(None, "rvm_resnet50_768x432_static.ncnn.param")?;
                    let bin = resolve_model_file(None, "rvm_resnet50_768x432_static.ncnn.bin")?;
                    Ok(Box::new(NcnnRvmSegmenter::new(&param, &bin, true)?))
                })();
                match ncnn_try {
                    Ok(seg) => seg,
                    Err(err) => {
                        warn!(
                            "NCNN Vulkan RVM ResNet50 unavailable ({err:#}) — falling back to ONNX CPU MobileNetV3"
                        );
                        let model_path =
                            resolve_model_file(args.model, "rvm_mobilenetv3_fp32.onnx")?;
                        Box::new(RvmSegmenter::new(
                            &ort_env,
                            &model_path,
                            args.downsample_ratio.clamp(0.25, 1.0),
                            args.threads.max(1),
                        )?)
                    }
                }
            }
        };

        let (mask_w, mask_h) = segmenter.mask_dims();

        let center_stage = match resolve_model_file(
            args.face_model,
            "face_detection_yunet_2023mar.onnx",
        ) {
            Ok(face_path) => match YuNetDetector::new(&ort_env, &face_path, 2) {
                Ok(det) => Some(CenterStageController::new(det)),
                Err(err) => {
                    warn!("Center Stage disabled (YuNet init failed): {err:#}");
                    None
                }
            },
            Err(err) => {
                warn!("Center Stage disabled (face model not found): {err:#}");
                None
            }
        };

        let gpu = GpuEffects::new(args.width, args.height, mask_w, mask_h)?;
        Box::new(StudioProcessor::new(
            segmenter,
            center_stage,
            gpu,
            Arc::clone(&shared_settings),
        ))
    };

    let pw_source_target = match args.source {
        CaptureSource::Aurora => Some(AURORA_DENOISE_NODE.to_string()),
        CaptureSource::V4l2 => None,
    };

    let config = VirtualCameraConfig {
        node_name: args.node_name,
        node_description: args.name,
        capture: CaptureConfig {
            device_path: args.device,
            pw_source_target,
            width: args.width,
            height: args.height,
            fps: args.fps,
        },
    };

    let hub_for_pw = Arc::clone(&ipc_hub);
    let on_streaming_changed: Arc<dyn Fn(bool) + Send + Sync> = Arc::new(move |streaming| {
        hub_for_pw.set_streaming(streaming);
    });

    run_virtual_camera(
        config,
        Arc::new(Mutex::new(processor)),
        on_streaming_changed,
        |tx| {
            if let Err(err) = ctrlc::set_handler(move || {
                let _ = tx.send(PwEvent::Quit);
            }) {
                log::warn!("Could not register Ctrl-C handler: {err}");
            }
        },
    )
}
