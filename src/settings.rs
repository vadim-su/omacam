use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundMode {
    /// No background effect (original background)
    None,
    /// macOS Portrait Mode — optical disk bokeh background blur
    Blur,
    /// Replace background with a custom image (PNG / JPEG / WebP)
    Image,
    /// Replace background with a subtle studio gradient around a base color
    Color,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EffectSettings {
    /// Master bypass switch (when false, passes raw camera through at ~0% cost)
    pub enabled: bool,
    /// Active background mode
    pub background_mode: BackgroundMode,
    /// Portrait bokeh blur intensity: 0.0..=1.0
    pub blur_strength: f32,
    /// Studio Light intensity (illuminates face/foreground & softly dims background): 0.0..=1.0
    pub studio_light: f32,
    /// Foreground FidelityFX CAS sharpness + local clarity + de-haze: 0.0..=1.0
    pub sharpness: f32,
    /// Apple-style Center Stage automatic face tracking & smooth zoom
    pub center_stage: bool,
    /// Center Stage framing/max-zoom preference: 0.0..=1.0
    pub center_stage_zoom: f32,
    /// Optional subtle blur applied to replacement background image (0.0..=1.0)
    pub bg_image_blur: f32,
    /// Path to background replacement image (used when background_mode == Image)
    pub bg_image_path: Option<PathBuf>,
    /// Base RGB color [0.0..=1.0; 3] for Color mode
    pub bg_color: [f32; 3],
    /// Monotonic revision counter bumped whenever settings change
    pub revision: u64,
}

impl Default for EffectSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            background_mode: BackgroundMode::Blur,
            blur_strength: 0.65,
            studio_light: 0.45,
            sharpness: 0.65,
            center_stage: false,
            center_stage_zoom: 0.55,
            bg_image_blur: 0.15,
            bg_image_path: None,
            bg_color: [0.11, 0.14, 0.20],
            revision: 1,
        }
    }
}

/// Partial update sent over IPC (Unix socket / `omacam ctl`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SettingsPatch {
    pub enabled: Option<bool>,
    pub background_mode: Option<BackgroundMode>,
    pub blur_strength: Option<f32>,
    pub studio_light: Option<f32>,
    pub sharpness: Option<f32>,
    pub center_stage: Option<bool>,
    pub center_stage_zoom: Option<f32>,
    pub bg_image_blur: Option<f32>,
    /// `Some(Some(path))` sets an image, `Some(None)` clears it, `None` leaves unchanged.
    #[serde(default)]
    pub bg_image_path: Option<Option<PathBuf>>,
    /// Hex color string like `#1c2433`
    pub bg_color_hex: Option<String>,
}

/// Snapshot serialized over IPC to UI widgets (e.g. Omarchy Shell Quickshell plugin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub connected: bool,
    pub streaming: bool,
    pub enabled: bool,
    pub background_mode: BackgroundMode,
    pub blur_strength: f32,
    pub studio_light: f32,
    pub sharpness: f32,
    pub center_stage: bool,
    pub center_stage_zoom: f32,
    pub bg_image_blur: f32,
    pub bg_image_path: Option<String>,
    pub bg_color_hex: String,
    pub revision: u64,
}

impl StatusSnapshot {
    pub fn from_settings(s: &EffectSettings, streaming: bool) -> Self {
        Self {
            connected: true,
            streaming,
            enabled: s.enabled,
            background_mode: s.background_mode,
            blur_strength: s.blur_strength,
            studio_light: s.studio_light,
            sharpness: s.sharpness,
            center_stage: s.center_stage,
            center_stage_zoom: s.center_stage_zoom,
            bg_image_blur: s.bg_image_blur,
            bg_image_path: s.bg_image_path.as_ref().map(|p| p.display().to_string()),
            bg_color_hex: s.bg_color_hex(),
            revision: s.revision,
        }
    }
}

impl EffectSettings {
    /// Returns true if any active effect needs the AI person alpha matte.
    pub fn needs_matting(&self) -> bool {
        if !self.enabled {
            return false;
        }
        let bg_active = match self.background_mode {
            BackgroundMode::None => false,
            BackgroundMode::Blur => self.blur_strength > 0.01,
            BackgroundMode::Image | BackgroundMode::Color => true,
        };
        bg_active || self.studio_light > 0.01
    }

    /// Returns true if GPU compute pass should run (matting effects, sharpness/clarity, or Center Stage).
    pub fn needs_gpu(&self) -> bool {
        if !self.enabled {
            return false;
        }
        self.needs_matting() || self.sharpness > 0.01 || self.center_stage
    }

    pub fn bg_color_hex(&self) -> String {
        let r = (self.bg_color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (self.bg_color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (self.bg_color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        format!("#{r:02x}{g:02x}{b:02x}")
    }

    pub fn parse_hex_color(hex: &str) -> Result<[f32; 3]> {
        let s = hex.trim().trim_start_matches('#');
        if s.len() != 6 {
            anyhow::bail!("expected 6-digit hex color like #1a2233, got '{hex}'");
        }
        let r = u8::from_str_radix(&s[0..2], 16)? as f32 / 255.0;
        let g = u8::from_str_radix(&s[2..4], 16)? as f32 / 255.0;
        let b = u8::from_str_radix(&s[4..6], 16)? as f32 / 255.0;
        Ok([r, g, b])
    }

    pub fn apply_patch(&mut self, patch: SettingsPatch) -> Result<()> {
        if let Some(v) = patch.enabled {
            self.enabled = v;
        }
        if let Some(v) = patch.background_mode {
            self.background_mode = v;
        }
        if let Some(v) = patch.blur_strength {
            self.blur_strength = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.studio_light {
            self.studio_light = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.sharpness {
            self.sharpness = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.center_stage {
            self.center_stage = v;
        }
        if let Some(v) = patch.center_stage_zoom {
            self.center_stage_zoom = v.clamp(0.0, 1.0);
        }
        if let Some(v) = patch.bg_image_blur {
            self.bg_image_blur = v.clamp(0.0, 1.0);
        }
        if let Some(opt_path) = patch.bg_image_path {
            self.bg_image_path = opt_path;
        }
        if let Some(hex) = patch.bg_color_hex {
            self.bg_color = Self::parse_hex_color(&hex)?;
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub fn config_path() -> Option<PathBuf> {
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
            && !xdg.is_empty()
        {
            return Some(PathBuf::from(xdg).join("omacam/settings.json"));
        }
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".config/omacam/settings.json"))
    }

    pub fn load_or_default() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<Self>(&raw) {
                Ok(cfg) => {
                    info!("Loaded saved settings from {}", path.display());
                    cfg
                }
                Err(err) => {
                    warn!("Invalid settings file {}: {err} (using defaults)", path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save_to_disk(&self) {
        let Some(path) = Self::config_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let tmp = path.with_extension("json.tmp");
            if fs::write(&tmp, json).is_ok() {
                let _ = fs::rename(&tmp, &path);
            }
        }
    }
}

pub type SharedSettings = Arc<RwLock<EffectSettings>>;
