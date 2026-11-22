use std::io::{self, Cursor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use pw::spa::pod::serialize::PodSerializer;
use pw::spa::pod::{Pod, Value};
use pw::spa::utils::{Direction, Fraction, Rectangle};
use pw::stream::{StreamBox, StreamFlags, StreamState};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use crate::pipeline::{Frame, FrameProcessor};

/// Default PipeWire node exposed by `/usr/bin/aurora-camera-denoise` on Apple Silicon Omarchy.
pub const AURORA_DENOISE_NODE: &str = "aurora_camera_denoise.v4l2_input.platform-580000000.isp";

/// Shared slot holding the most recent processed NV12 frame.
/// Using a single overwrite slot guarantees zero queue buildup and minimal latency.
pub struct LatestFrame {
    pub frame: Frame,
    pub seq: u64,
}

impl LatestFrame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            frame: Frame::black(width, height),
            seq: 0,
        }
    }
}

#[derive(Clone)]
pub struct CaptureConfig {
    pub device_path: PathBuf,
    /// Optional PipeWire Video/Source node name to capture from (e.g. `aurora_camera_denoise...`).
    /// Falls back automatically to `device_path` (V4L2) if unavailable.
    pub pw_source_target: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// Manages on-demand camera capture (PipeWire `aurora_camera_denoise` with V4L2 fallback).
/// The camera / upstream denoise node is ONLY opened while `start()` is active and released on `stop()`.
pub struct CaptureManager {
    config: CaptureConfig,
    latest: Arc<Mutex<LatestFrame>>,
    on_frame: Arc<dyn Fn() + Send + Sync>,
    processor: Arc<Mutex<Box<dyn FrameProcessor>>>,
    running: Arc<AtomicBool>,
    pw_quit_tx: Arc<Mutex<Option<pw::channel::Sender<()>>>>,
    worker: Option<JoinHandle<()>>,
}

impl CaptureManager {
    pub fn new(
        config: CaptureConfig,
        latest: Arc<Mutex<LatestFrame>>,
        on_frame: Arc<dyn Fn() + Send + Sync>,
        processor: Arc<Mutex<Box<dyn FrameProcessor>>>,
    ) -> Self {
        Self {
            config,
            latest,
            on_frame,
            processor,
            running: Arc::new(AtomicBool::new(false)),
            pw_quit_tx: Arc::new(Mutex::new(None)),
            worker: None,
        }
    }

    pub fn start(&mut self) {
        if self.worker.is_some() {
            return;
        }
        info!("Client connected — starting on-demand camera capture");
        self.running.store(true, Ordering::SeqCst);

        let config = self.config.clone();
        let latest = Arc::clone(&self.latest);
        let on_frame = Arc::clone(&self.on_frame);
        let running = Arc::clone(&self.running);
        let processor = Arc::clone(&self.processor);
        let pw_quit_tx = Arc::clone(&self.pw_quit_tx);

        let handle = thread::Builder::new()
            .name("omacam-capture".into())
            .spawn(move || {
                {
                    let mut proc_guard = processor.lock().unwrap();
                    proc_guard.on_stream_start();
                }

                let mut need_v4l2_fallback = true;
                if let Some(target) = config.pw_source_target.as_deref() {
                    match pipewire_capture_loop(
                        target,
                        &config,
                        &latest,
                        &on_frame,
                        &running,
                        &pw_quit_tx,
                        &processor,
                    ) {
                        Ok(true) => {
                            need_v4l2_fallback = false;
                        }
                        Ok(false) => {
                            if running.load(Ordering::Relaxed) {
                                warn!(
                                    "PipeWire source '{target}' did not deliver frames — falling back to V4L2 {}",
                                    config.device_path.display()
                                );
                            } else {
                                need_v4l2_fallback = false;
                            }
                        }
                        Err(err) => {
                            if running.load(Ordering::Relaxed) {
                                warn!(
                                    "PipeWire source '{target}' failed ({err:#}) — falling back to V4L2 {}",
                                    config.device_path.display()
                                );
                            } else {
                                need_v4l2_fallback = false;
                            }
                        }
                    }
                }

                if need_v4l2_fallback && running.load(Ordering::Relaxed) {
                    let mut proc_guard = processor.lock().unwrap();
                    if let Err(err) =
                        v4l2_capture_loop(&config, &latest, &*on_frame, &running, proc_guard.as_mut())
                    {
                        error!("V4L2 capture loop exited with error: {err:#}");
                    }
                }

                *pw_quit_tx.lock().unwrap() = None;
                running.store(false, Ordering::SeqCst);
                info!("Camera capture stopped & upstream source released");
            })
            .expect("failed to spawn capture thread");

        self.worker = Some(handle);
    }

    pub fn stop(&mut self) {
        if let Some(handle) = self.worker.take() {
            info!("No active clients — stopping camera capture");
            self.running.store(false, Ordering::SeqCst);
            if let Some(tx) = self.pw_quit_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
            let _ = handle.join();
        }
    }
}

impl Drop for CaptureManager {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Captures denoised NV12 frames directly from PipeWire `aurora_camera_denoise` on-demand.
/// Returns `Ok(true)` if frames flowed cleanly until `running` became false, or `Ok(false)` if
/// the upstream node was unavailable and the caller should fall back to V4L2.
fn pipewire_capture_loop(
    target_node: &str,
    config: &CaptureConfig,
    latest: &Arc<Mutex<LatestFrame>>,
    on_frame: &Arc<dyn Fn() + Send + Sync>,
    running: &Arc<AtomicBool>,
    pw_quit_tx: &Arc<Mutex<Option<pw::channel::Sender<()>>>>,
    processor: &Arc<Mutex<Box<dyn FrameProcessor>>>,
) -> Result<bool> {
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .context("failed to create PipeWire capture MainLoop")?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .context("failed to create PipeWire capture Context")?;
    let core = context
        .connect_rc(None)
        .context("failed to connect PipeWire capture Core")?;

    let (quit_tx, quit_rx) = pw::channel::channel::<()>();
    *pw_quit_tx.lock().unwrap() = Some(quit_tx.clone());

    // Watchdog: if no frame arrives within 2.5s (e.g. aurora-camera-denoise stopped), quit & fallback.
    let got_first_frame = Arc::new(AtomicBool::new(false));
    let watchdog_flag = Arc::clone(&got_first_frame);
    let watchdog_running = Arc::clone(running);
    let watchdog_tx = quit_tx.clone();
    thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_millis(2500);
        while Instant::now() < deadline {
            if !watchdog_running.load(Ordering::Relaxed) || watchdog_flag.load(Ordering::Relaxed) {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        if watchdog_running.load(Ordering::Relaxed) && !watchdog_flag.load(Ordering::Relaxed) {
            let _ = watchdog_tx.send(());
        }
    });

    let stream = StreamBox::new(
        &core,
        "omacam-input",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Camera",
            *pw::keys::NODE_NAME => "omacam-input",
            *pw::keys::TARGET_OBJECT => target_node,
        },
    )
    .context("failed to create PipeWire input stream")?;

    let expected_size = Frame::expected_nv12_size(config.width, config.height);

    struct InputUserData {
        width: u32,
        height: u32,
        expected_size: usize,
        work_frame: Frame,
        latest: Arc<Mutex<LatestFrame>>,
        running: Arc<AtomicBool>,
        got_first_frame: Arc<AtomicBool>,
        processor: Arc<Mutex<Box<dyn FrameProcessor>>>,
        on_frame: Arc<dyn Fn() + Send + Sync>,
        frame_count: u64,
        fps_timer: Instant,
    }

    let user_data = InputUserData {
        width: config.width,
        height: config.height,
        expected_size,
        work_frame: Frame {
            width: config.width,
            height: config.height,
            nv12: vec![0u8; expected_size],
        },
        latest: Arc::clone(latest),
        running: Arc::clone(running),
        got_first_frame: Arc::clone(&got_first_frame),
        processor: Arc::clone(processor),
        on_frame: Arc::clone(on_frame),
        frame_count: 0,
        fps_timer: Instant::now(),
    };

    let target_owned = target_node.to_string();
    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(move |_, _, old, new| {
            debug!("PipeWire input '{target_owned}' state: {old:?} -> {new:?}");
            if let StreamState::Streaming = new {
                info!("Connected to PipeWire source '{target_owned}'");
            }
        })
        .param_changed(|_, data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let mut info = spa::param::video::VideoInfoRaw::new();
            if info.parse(param).is_ok() {
                let sz = info.size();
                if sz.width > 0 && sz.height > 0 {
                    data.width = sz.width;
                    data.height = sz.height;
                    data.expected_size = Frame::expected_nv12_size(sz.width, sz.height);
                    data.work_frame.width = sz.width;
                    data.work_frame.height = sz.height;
                    data.work_frame.nv12.resize(data.expected_size, 0);
                    info!(
                        "PipeWire input format: {:?} {}x{} @ {}/{} fps",
                        info.format(),
                        sz.width,
                        sz.height,
                        info.framerate().num,
                        info.framerate().denom
                    );
                }
            }
        })
        .process(|stream, data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            if !data.running.load(Ordering::Relaxed) {
                return;
            }
            let datas = buffer.datas_mut();
            let Some(pw_data) = datas.first_mut() else {
                return;
            };
            let chunk_offset = pw_data.chunk().offset() as usize;
            let chunk_size = pw_data.chunk().size() as usize;
            let stride = pw_data.chunk().stride().max(0) as usize;
            let Some(slice) = pw_data.data() else {
                return;
            };
            if chunk_size == 0 || chunk_offset + chunk_size > slice.len() {
                return;
            }
            let buf = &slice[chunk_offset..chunk_offset + chunk_size];
            if buf.len() < data.expected_size {
                return;
            }

            data.got_first_frame.store(true, Ordering::Relaxed);

            let width = data.width as usize;
            let height = data.height as usize;
            if stride == width || stride == 0 {
                data.work_frame.nv12[..data.expected_size]
                    .copy_from_slice(&buf[..data.expected_size]);
            } else {
                let y_rows = height;
                let uv_rows = height / 2;
                for row in 0..y_rows {
                    let src_off = row * stride;
                    let dst_off = row * width;
                    if src_off + width <= buf.len() {
                        data.work_frame.nv12[dst_off..dst_off + width]
                            .copy_from_slice(&buf[src_off..src_off + width]);
                    }
                }
                let src_uv_base = y_rows * stride;
                let dst_uv_base = y_rows * width;
                for row in 0..uv_rows {
                    let src_off = src_uv_base + row * stride;
                    let dst_off = dst_uv_base + row * width;
                    if src_off + width <= buf.len() {
                        data.work_frame.nv12[dst_off..dst_off + width]
                            .copy_from_slice(&buf[src_off..src_off + width]);
                    }
                }
            }

            {
                let mut proc_guard = data.processor.lock().unwrap();
                if let Err(err) = proc_guard.process(&mut data.work_frame) {
                    warn!("Frame processing error: {err:#}");
                }
            }

            {
                let mut guard = data.latest.lock().unwrap();
                guard.frame.width = data.work_frame.width;
                guard.frame.height = data.work_frame.height;
                std::mem::swap(&mut guard.frame.nv12, &mut data.work_frame.nv12);
                if data.work_frame.nv12.len() != data.expected_size {
                    data.work_frame.nv12.resize(data.expected_size, 0);
                }
                guard.seq += 1;
            }

            (data.on_frame)();

            data.frame_count += 1;
            let elapsed = data.fps_timer.elapsed();
            if elapsed >= Duration::from_secs(5) {
                debug!(
                    "PipeWire input throughput: {:.1} FPS",
                    data.frame_count as f64 / elapsed.as_secs_f64()
                );
                data.frame_count = 0;
                data.fps_timer = Instant::now();
            }
        })
        .register()
        .context("failed to register PipeWire input listener")?;

    let format_obj = spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaType,
            Id,
            spa::param::format::MediaType::Video
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaSubtype,
            Id,
            spa::param::format::MediaSubtype::Raw
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFormat,
            Id,
            spa::param::video::VideoFormat::NV12
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoSize,
            Rectangle,
            Rectangle {
                width: config.width,
                height: config.height
            }
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            Fraction {
                num: config.fps,
                denom: 1
            },
            Fraction { num: 1, denom: 1 },
            Fraction {
                num: config.fps,
                denom: 1
            }
        ),
    );
    let format_bytes = PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(format_obj))
        .expect("failed to serialize input format pod")
        .0
        .into_inner();
    let mut params = [Pod::from_bytes(&format_bytes).unwrap()];

    stream
        .connect(
            Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .context("failed to connect PipeWire input stream")?;

    let ml_quit = mainloop.clone();
    let _attached = quit_rx.attach(mainloop.loop_(), move |_| {
        ml_quit.quit();
    });

    if !running.load(Ordering::SeqCst) {
        let _ = stream.disconnect();
        return Ok(true);
    }

    mainloop.run();
    let _ = stream.disconnect();
    Ok(got_first_frame.load(Ordering::Relaxed))
}

fn v4l2_capture_loop(
    config: &CaptureConfig,
    latest: &Arc<Mutex<LatestFrame>>,
    on_frame: &(dyn Fn() + Send + Sync),
    running: &AtomicBool,
    processor: &mut dyn FrameProcessor,
) -> Result<()> {
    let dev = Device::with_path(&config.device_path)
        .with_context(|| format!("failed to open V4L2 device {}", config.device_path.display()))?;

    let mut fmt = dev.format().context("failed to query V4L2 format")?;
    fmt.width = config.width;
    fmt.height = config.height;
    fmt.fourcc = FourCC::new(b"NV12");
    let fmt = dev
        .set_format(&fmt)
        .context("failed to set NV12 V4L2 format")?;

    if fmt.fourcc != FourCC::new(b"NV12") {
        anyhow::bail!(
            "camera returned unsupported pixel format {:?}, expected NV12",
            fmt.fourcc
        );
    }

    if let Ok(mut params) = dev.params() {
        params.interval = v4l::Fraction::new(1, config.fps);
        if let Err(err) = dev.set_params(&params) {
            warn!("Could not set framerate to {} fps: {err}", config.fps);
        }
    }

    info!(
        "Opened V4L2 {} ({}x{}, {:?}, stride={})",
        config.device_path.display(),
        fmt.width,
        fmt.height,
        fmt.fourcc,
        fmt.stride
    );

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)
        .context("failed to start V4L2 mmap stream")?;
    stream.set_timeout(Duration::from_millis(200));

    let width = fmt.width as usize;
    let height = fmt.height as usize;
    let stride = fmt.stride as usize;
    let expected_size = Frame::expected_nv12_size(fmt.width, fmt.height);

    let mut work_frame = Frame {
        width: fmt.width,
        height: fmt.height,
        nv12: vec![0u8; expected_size],
    };

    let mut frame_count: u64 = 0;
    let mut fps_timer = Instant::now();

    while running.load(Ordering::Relaxed) {
        match stream.next() {
            Ok((buf, _meta)) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }

                if buf.len() < expected_size {
                    warn!(
                        "Short V4L2 frame: got {} bytes, expected {}",
                        buf.len(),
                        expected_size
                    );
                    continue;
                }

                if stride == width || stride == 0 {
                    work_frame.nv12[..expected_size].copy_from_slice(&buf[..expected_size]);
                } else {
                    let y_rows = height;
                    let uv_rows = height / 2;
                    for row in 0..y_rows {
                        let src_off = row * stride;
                        let dst_off = row * width;
                        work_frame.nv12[dst_off..dst_off + width]
                            .copy_from_slice(&buf[src_off..src_off + width]);
                    }
                    let src_uv_base = y_rows * stride;
                    let dst_uv_base = y_rows * width;
                    for row in 0..uv_rows {
                        let src_off = src_uv_base + row * stride;
                        let dst_off = dst_uv_base + row * width;
                        work_frame.nv12[dst_off..dst_off + width]
                            .copy_from_slice(&buf[src_off..src_off + width]);
                    }
                }

                if let Err(err) = processor.process(&mut work_frame) {
                    warn!("Frame processing error: {err:#}");
                }

                {
                    let mut guard = latest.lock().unwrap();
                    guard.frame.width = work_frame.width;
                    guard.frame.height = work_frame.height;
                    std::mem::swap(&mut guard.frame.nv12, &mut work_frame.nv12);
                    if work_frame.nv12.len() != expected_size {
                        work_frame.nv12.resize(expected_size, 0);
                    }
                    guard.seq += 1;
                }

                on_frame();

                frame_count += 1;
                let elapsed = fps_timer.elapsed();
                if elapsed >= Duration::from_secs(5) {
                    debug!(
                        "V4L2 capture throughput: {:.1} FPS",
                        frame_count as f64 / elapsed.as_secs_f64()
                    );
                    frame_count = 0;
                    fps_timer = Instant::now();
                }
            }
            Err(err)
                if err.kind() == io::ErrorKind::TimedOut
                    || err.kind() == io::ErrorKind::WouldBlock =>
            {
                continue;
            }
            Err(err) => {
                return Err(err).context("V4L2 frame dequeue failed");
            }
        }
    }

    Ok(())
}
