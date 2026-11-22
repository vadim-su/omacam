#!/bin/bash
set -euo pipefail

dirs=()
for d in \
  "$HOME/Pictures" \
  "$HOME/Pictures/Generated" \
  "$HOME/Pictures/Screenshots" \
  "$HOME/.config/omarchy/backgrounds" \
  "$HOME/.config/omarchy/current/theme/backgrounds" \
  /usr/share/omarchy/themes/*/backgrounds
do
  [ -d "$d" ] && dirs+=("$d")
done

if (( ${#dirs[@]} == 0 )); then
  exit 0
fi

chosen=$(omarchy-menu-images --show-labels --filterable --lazy-thumbnails "${dirs[@]}" || true)
if [ -n "$chosen" ] && [ -f "$chosen" ]; then
  "$HOME/.local/bin/omacam" ctl set --bg image --bg-image "$chosen" >/dev/null
fi
