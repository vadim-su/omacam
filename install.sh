#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
MODEL_DIR="$HOME/.local/share/omacam/models"
SYSTEMD_DIR="$HOME/.config/systemd/user"
PLUGIN_DIR="$HOME/.config/omarchy/plugins/vadim-su.omacam"

mkdir -p "$BIN_DIR" "$MODEL_DIR" "$SYSTEMD_DIR"

echo "==> Checking ncnn library (system or ~/.local/lib)..."
if ! ldconfig -p 2>/dev/null | grep -q "libncnn.so" && [ ! -f "$HOME/.local/lib/libncnn.so" ]; then
  echo "libncnn.so not found. Install it with: sudo pacman -S ncnn"
  echo "Or build ncnn into ~/.local with -DNCNN_VULKAN=ON -DNCNN_C_API=ON -DNCNN_SHARED_LIB=ON"
  exit 1
fi

echo "==> Downloading ONNX models..."
if [ ! -f "$MODEL_DIR/rvm_mobilenetv3_fp32.onnx" ]; then
  curl -fSL -o "$MODEL_DIR/rvm_mobilenetv3_fp32.onnx" \
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
fi
if [ ! -f "$MODEL_DIR/face_detection_yunet_2023mar.onnx" ]; then
  curl -fSL -o "$MODEL_DIR/face_detection_yunet_2023mar.onnx" \
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi

if [ -f "$REPO_DIR/models/rvm_resnet50_768x432_static.ncnn.param" ] && [ -f "$REPO_DIR/models/rvm_resnet50_768x432_static.ncnn.bin" ]; then
  cp -u "$REPO_DIR/models/rvm_resnet50_768x432_static.ncnn.param" "$MODEL_DIR/"
  cp -u "$REPO_DIR/models/rvm_resnet50_768x432_static.ncnn.bin" "$MODEL_DIR/"
fi

echo "==> Building omacam (release)..."
if command -v bgrun >/dev/null 2>&1; then
  bgrun cargo build --release --manifest-path "$REPO_DIR/Cargo.toml"
else
  cargo build --release --manifest-path "$REPO_DIR/Cargo.toml"
fi

install -m 755 "$REPO_DIR/target/release/omacam" "$BIN_DIR/omacam"

echo "==> Installing systemd user service..."
install -m 644 "$REPO_DIR/systemd/omacam.service" "$SYSTEMD_DIR/omacam.service"
systemctl --user daemon-reload
systemctl --user enable --now omacam.service

if command -v omarchy >/dev/null 2>&1; then
  echo "==> Installing Omarchy Shell bar plugin (vadim-su.omacam)..."
  mkdir -p "$PLUGIN_DIR"
  cp -r "$REPO_DIR/manifest.json" "$REPO_DIR/BarWidget.qml" "$REPO_DIR/pick-bg.sh" "$REPO_DIR/i18n" "$PLUGIN_DIR/"
  chmod +x "$PLUGIN_DIR/pick-bg.sh"
  omarchy plugin validate "$PLUGIN_DIR"
  omarchy bar put vadim-su.omacam --section right || true
  if command -v omarchy-restart-shell >/dev/null 2>&1; then
    omarchy-restart-shell || true
  fi
fi

echo "==> Done! Select 'Omacam Studio' in Brave, Chrome, Discord, or any PipeWire-capable app."
