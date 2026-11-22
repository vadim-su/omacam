use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::settings::{SettingsPatch, SharedSettings, StatusSnapshot};

pub fn socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir).join("omacam.sock");
    }
    PathBuf::from("/tmp/omacam.sock")
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum IpcRequest {
    Get,
    Toggle,
    Subscribe,
    Set {
        patch: SettingsPatch,
        #[serde(default = "default_true")]
        persist: bool,
    },
}

fn default_true() -> bool {
    true
}

/// Shared IPC hub owned by the daemon.
/// Broadcasts `StatusSnapshot` JSON lines to all connected subscribers on every state change.
pub struct IpcHub {
    settings: SharedSettings,
    streaming: AtomicBool,
    subscribers: Mutex<Vec<UnixStream>>,
}

impl IpcHub {
    pub fn new(settings: SharedSettings) -> Arc<Self> {
        Arc::new(Self {
            settings,
            streaming: AtomicBool::new(false),
            subscribers: Mutex::new(Vec::new()),
        })
    }

    pub fn snapshot(&self) -> StatusSnapshot {
        let s = self.settings.read().unwrap();
        let streaming = self.streaming.load(Ordering::SeqCst);
        StatusSnapshot::from_settings(&s, streaming)
    }

    pub fn set_streaming(&self, streaming: bool) {
        let prev = self.streaming.swap(streaming, Ordering::SeqCst);
        if prev != streaming {
            self.broadcast();
        }
    }

    pub fn broadcast(&self) {
        let snap = self.snapshot();
        let Ok(mut line) = serde_json::to_string(&snap) else {
            return;
        };
        line.push('\n');
        let bytes = line.as_bytes();

        let mut subs = self.subscribers.lock().unwrap();
        subs.retain_mut(|stream| stream.write_all(bytes).is_ok());
    }

    fn add_subscriber(&self, stream: UnixStream) {
        let mut subs = self.subscribers.lock().unwrap();
        subs.push(stream);
    }

    /// Starts the Unix domain socket listener in a background thread.
    pub fn start_server(self: &Arc<Self>) -> Result<()> {
        let path = socket_path();
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        let listener = UnixListener::bind(&path)
            .with_context(|| format!("failed to bind IPC socket at {}", path.display()))?;
        info!("IPC socket listening at {}", path.display());

        let hub = Arc::clone(self);
        thread::Builder::new()
            .name("omacam-ipc".into())
            .spawn(move || {
                for incoming in listener.incoming() {
                    match incoming {
                        Ok(stream) => {
                            let hub = Arc::clone(&hub);
                            thread::spawn(move || handle_client(hub, stream));
                        }
                        Err(err) => {
                            warn!("IPC accept error: {err}");
                        }
                    }
                }
            })
            .context("failed to spawn IPC listener thread")?;

        Ok(())
    }
}

fn send_snapshot(stream: &mut UnixStream, snap: &StatusSnapshot) -> io::Result<()> {
    let mut line = serde_json::to_string(snap).unwrap_or_default();
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()
}

fn handle_client(hub: Arc<IpcHub>, stream: UnixStream) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let reader = BufReader::new(stream);
    let mut subscribed = false;

    for line_res in reader.lines() {
        let Ok(line) = line_res else {
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: IpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(err) => {
                debug!("Malformed IPC request '{trimmed}': {err}");
                continue;
            }
        };

        match req {
            IpcRequest::Get => {
                let snap = hub.snapshot();
                if send_snapshot(&mut writer, &snap).is_err() {
                    break;
                }
            }
            IpcRequest::Subscribe => {
                if !subscribed {
                    let snap = hub.snapshot();
                    if send_snapshot(&mut writer, &snap).is_err() {
                        break;
                    }
                    if let Ok(sub_stream) = writer.try_clone() {
                        hub.add_subscriber(sub_stream);
                        subscribed = true;
                    }
                }
            }
            IpcRequest::Toggle => {
                {
                    let mut s = hub.settings.write().unwrap();
                    s.enabled = !s.enabled;
                    s.revision = s.revision.wrapping_add(1);
                    s.save_to_disk();
                }
                hub.broadcast();
                if !subscribed {
                    let snap = hub.snapshot();
                    if send_snapshot(&mut writer, &snap).is_err() {
                        break;
                    }
                }
            }
            IpcRequest::Set { patch, persist } => {
                let apply_res = {
                    let mut s = hub.settings.write().unwrap();
                    let r = s.apply_patch(patch);
                    if r.is_ok() && persist {
                        s.save_to_disk();
                    }
                    r
                };
                if let Err(err) = apply_res {
                    warn!("Failed to apply IPC patch: {err}");
                    continue;
                }
                hub.broadcast();
                if !subscribed {
                    let snap = hub.snapshot();
                    if send_snapshot(&mut writer, &snap).is_err() {
                        break;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Client helpers (`omacam ctl ...`)
// -----------------------------------------------------------------------------

pub fn send_one_shot(req: &IpcRequest) -> Result<StatusSnapshot> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).with_context(|| {
        format!(
            "cannot connect to omacam daemon at {} (is omacam running?)",
            path.display()
        )
    })?;
    let mut line = serde_json::to_string(req)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader
        .read_line(&mut resp)
        .context("failed to read response from omacam daemon")?;
    let snap: StatusSnapshot =
        serde_json::from_str(resp.trim()).context("invalid JSON response from daemon")?;
    Ok(snap)
}

/// Persistent bidirectional bridge for Quickshell (`omacam ctl watch`):
/// - Emits a `StatusSnapshot` JSON line to stdout on connect, disconnect, and every state change.
/// - Forwards any JSON `IpcRequest` lines written to stdin straight to the daemon socket.
pub fn run_watch_bridge() -> Result<()> {
    let path = socket_path();
    let active_writer: Arc<Mutex<Option<UnixStream>>> = Arc::new(Mutex::new(None));

    // Stdin reader thread: forwards JSON commands from Quickshell to the active daemon socket.
    let writer_for_stdin = Arc::clone(&active_writer);
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut guard = writer_for_stdin.lock().unwrap();
            if let Some(stream) = guard.as_mut() {
                let mut payload = trimmed.to_string();
                payload.push('\n');
                if stream.write_all(payload.as_bytes()).is_err() {
                    *guard = None;
                }
            }
        }
        // Stdin closed (parent shell exited) -> terminate cleanly.
        std::process::exit(0);
    });

    let mut last_connected = true;
    loop {
        match UnixStream::connect(&path) {
            Ok(mut stream) => {
                last_connected = true;
                let sub = format!("{}\n", serde_json::to_string(&IpcRequest::Subscribe)?);
                if stream.write_all(sub.as_bytes()).is_err() {
                    thread::sleep(Duration::from_millis(1000));
                    continue;
                }
                if let Ok(writer_clone) = stream.try_clone() {
                    *active_writer.lock().unwrap() = Some(writer_clone);
                }
                let reader = BufReader::new(stream);
                let mut stdout = io::stdout().lock();
                for line in reader.lines() {
                    match line {
                        Ok(l) if !l.trim().is_empty() => {
                            if writeln!(stdout, "{}", l.trim()).is_err() || stdout.flush().is_err() {
                                return Ok(());
                            }
                        }
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
                *active_writer.lock().unwrap() = None;
            }
            Err(_) => {
                if last_connected {
                    last_connected = false;
                    let offline = serde_json::json!({
                        "connected": false,
                        "streaming": false,
                        "enabled": false,
                        "background_mode": "blur",
                        "blur_strength": 0.65,
                        "studio_light": 0.45,
                        "sharpness": 0.65,
                        "center_stage": false,
                        "center_stage_zoom": 0.55,
                        "bg_image_blur": 0.15,
                        "bg_image_path": null,
                        "bg_color_hex": "#1c2433",
                        "revision": 0
                    });
                    let mut stdout = io::stdout().lock();
                    if writeln!(stdout, "{offline}").is_err() || stdout.flush().is_err() {
                        return Ok(());
                    }
                }
                thread::sleep(Duration::from_millis(1500));
            }
        }
    }
}
