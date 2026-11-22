use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use pw::spa::pod::serialize::PodSerializer;
use pw::spa::pod::{ChoiceValue, Object, Pod, Property, PropertyFlags, Value};
use pw::spa::sys as spa_sys;
use pw::spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Direction, Fraction, Rectangle};
use pw::stream::{StreamFlags, StreamRc, StreamState};

use crate::capture::{CaptureConfig, CaptureManager, LatestFrame};
use crate::pipeline::{Frame, FrameProcessor};

pub enum PwEvent {
    FrameReady,
    Quit,
}

pub struct VirtualCameraConfig {
    pub node_name: String,
    pub node_description: String,
    pub capture: CaptureConfig,
}

struct StreamUserData {
    width: u32,
    height: u32,
    format_negotiated: bool,
    streaming: bool,
    capture: CaptureManager,
    latest: Arc<Mutex<LatestFrame>>,
    on_streaming_changed: Arc<dyn Fn(bool) + Send + Sync>,
}

/// Runs the PipeWire main loop, exposing a virtual `Video/Source` camera node.
/// The underlying physical V4L2 camera is opened strictly on-demand while a client streams.
pub fn run_virtual_camera(
    config: VirtualCameraConfig,
    processor: Arc<Mutex<Box<dyn FrameProcessor>>>,
    on_streaming_changed: Arc<dyn Fn(bool) + Send + Sync>,
    setup_signal: impl FnOnce(pw::channel::Sender<PwEvent>),
) -> Result<()> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .context("failed to create PipeWire MainLoop")?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .context("failed to create PipeWire Context")?;
    let core = context
        .connect_rc(None)
        .context("failed to connect to PipeWire daemon (is pipewire running?)")?;

    let (tx, rx) = pw::channel::channel::<PwEvent>();
    setup_signal(tx.clone());

    let width = config.capture.width;
    let height = config.capture.height;
    let fps = config.capture.fps;

    let latest = Arc::new(Mutex::new(LatestFrame::new(width, height)));

    let frame_tx = tx.clone();
    let on_frame: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        let _ = frame_tx.send(PwEvent::FrameReady);
    });

    let capture = CaptureManager::new(
        config.capture.clone(),
        Arc::clone(&latest),
        on_frame,
        processor,
    );

    let stream = StreamRc::new(
        core.clone(),
        &config.node_name,
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Source",
            *pw::keys::MEDIA_ROLE => "Camera",
            *pw::keys::MEDIA_CLASS => "Video/Source",
            *pw::keys::NODE_NAME => config.node_name.as_str(),
            *pw::keys::NODE_DESCRIPTION => config.node_description.as_str(),
            *pw::keys::NODE_NICK => config.node_description.as_str(),
            "node.pause-on-idle" => "true",
        },
    )
    .context("failed to create PipeWire video stream")?;

    let user_data = StreamUserData {
        width,
        height,
        format_negotiated: false,
        streaming: false,
        capture,
        latest: Arc::clone(&latest),
        on_streaming_changed,
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|stream, data, old, new| {
            info!(
                "PipeWire stream '{}' state: {:?} -> {:?} (node_id={})",
                stream.name(),
                old,
                new,
                stream.node_id()
            );
            match new {
                StreamState::Streaming => {
                    data.streaming = true;
                    data.capture.start();
                    (data.on_streaming_changed)(true);
                }
                StreamState::Paused | StreamState::Unconnected | StreamState::Error(_) => {
                    data.streaming = false;
                    data.capture.stop();
                    (data.on_streaming_changed)(false);
                }
                StreamState::Connecting => {}
            }
        })
        .param_changed(|stream, data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }

            let Ok((media_type, media_subtype)) =
                spa::param::format_utils::parse_format(param)
            else {
                return;
            };

            if media_type != spa::param::format::MediaType::Video
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }

            let mut video_info = spa::param::video::VideoInfoRaw::new();
            if let Err(err) = video_info.parse(param) {
                warn!("Failed to parse negotiated VideoInfoRaw: {err:?}");
                return;
            }

            let size = video_info.size();
            let fr = video_info.framerate();
            info!(
                "Negotiated PipeWire format: {:?} {}x{} @ {}/{} fps",
                video_info.format(),
                size.width,
                size.height,
                fr.num,
                fr.denom
            );

            data.width = size.width;
            data.height = size.height;
            data.format_negotiated = true;

            let buf_bytes = build_buffers_param(size.width, size.height);
            if let Some(buf_pod) = Pod::from_bytes(&buf_bytes) {
                let mut params = [buf_pod];
                if let Err(err) = stream.update_params(&mut params) {
                    error!("Failed to update SPA_PARAM_Buffers: {err:?}");
                }
            }
        })
        .process(|stream, data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                debug!("No PipeWire buffer available to dequeue");
                return;
            };

            let datas = buffer.datas_mut();
            let Some(pw_data) = datas.first_mut() else {
                return;
            };

            let expected_size = Frame::expected_nv12_size(data.width, data.height);
            let stride = data.width as i32;

            let written = if let Some(dst) = pw_data.data() {
                let guard = data.latest.lock().unwrap();
                let src = &guard.frame.nv12;
                let copy_len = expected_size.min(dst.len()).min(src.len());
                dst[..copy_len].copy_from_slice(&src[..copy_len]);
                copy_len
            } else {
                0
            };

            let chunk = pw_data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.size_mut() = written as u32;
            *chunk.stride_mut() = stride;
        })
        .register()
        .context("failed to register PipeWire stream listener")?;

    let format_bytes = build_enum_format_param(width, height, fps);
    let format_pod =
        Pod::from_bytes(&format_bytes).context("failed to parse serialized format pod")?;
    let mut params = [format_pod];

    stream
        .connect(
            Direction::Output,
            None,
            StreamFlags::DRIVER | StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .context("failed to connect PipeWire virtual camera stream")?;

    info!(
        "Virtual camera '{}' registered in PipeWire ({}x{} NV12 @ {} fps, on-demand idle)",
        config.node_description, width, height, fps
    );

    let stream_for_rx = stream.clone();
    let mainloop_for_rx = mainloop.clone();
    let _attached = rx.attach(mainloop.loop_(), move |event| match event {
        PwEvent::FrameReady => {
            if stream_for_rx.state() == StreamState::Streaming && stream_for_rx.is_driving() {
                let _ = stream_for_rx.trigger_process();
            }
        }
        PwEvent::Quit => {
            info!("Shutting down PipeWire virtual camera...");
            mainloop_for_rx.quit();
        }
    });

    mainloop.run();
    let _ = stream.disconnect();
    Ok(())
}

fn build_enum_format_param(width: u32, height: u32, fps: u32) -> Vec<u8> {
    let obj = spa::pod::object!(
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
            Rectangle { width, height }
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            Fraction {
                num: fps,
                denom: 1
            },
            Fraction { num: 15, denom: 1 },
            Fraction {
                num: fps,
                denom: 1
            }
        ),
    );

    PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("failed to serialize SPA_PARAM_EnumFormat")
        .0
        .into_inner()
}

fn build_buffers_param(width: u32, height: u32) -> Vec<u8> {
    let size = Frame::expected_nv12_size(width, height) as i32;
    let stride = width as i32;
    let data_types =
        (1i32 << spa_sys::SPA_DATA_MemPtr) | (1i32 << spa_sys::SPA_DATA_MemFd);

    let obj = Object {
        type_: spa_sys::SPA_TYPE_OBJECT_ParamBuffers,
        id: spa_sys::SPA_PARAM_Buffers,
        properties: vec![
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_buffers,
                flags: PropertyFlags::empty(),
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: 4,
                        min: 2,
                        max: 8,
                    },
                ))),
            },
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_blocks,
                flags: PropertyFlags::empty(),
                value: Value::Int(1),
            },
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(size),
            },
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_stride,
                flags: PropertyFlags::empty(),
                value: Value::Int(stride),
            },
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_align,
                flags: PropertyFlags::empty(),
                value: Value::Int(16),
            },
            Property {
                key: spa_sys::SPA_PARAM_BUFFERS_dataType,
                flags: PropertyFlags::empty(),
                value: Value::Int(data_types),
            },
        ],
    };

    PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("failed to serialize SPA_PARAM_Buffers")
        .0
        .into_inner()
}
