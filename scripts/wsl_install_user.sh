#!/usr/bin/env bash
# Installs the Swift toolchain (via swiftly) and xtool into the user's home. No sudo needed.
# Usage (inside WSL Ubuntu, from the repo root):  bash scripts/wsl_install_user.sh
set -euo pipefail
ARCH="$(uname -m)"
mkdir -p "$HOME/.local/bin"

# --- Swift toolchain via swiftly ---------------------------------------------
if ! command -v swiftly >/dev/null 2>&1 && [ ! -x "$HOME/.local/share/swiftly/bin/swiftly" ]; then
  tmp="$(mktemp -d)"
  curl -fsSL "https://download.swift.org/swiftly/linux/swiftly-${ARCH}.tar.gz" -o "$tmp/swiftly.tar.gz"
  tar -xzf "$tmp/swiftly.tar.gz" -C "$tmp"
  # Installs the latest stable Swift toolchain; skip apt deps (handled by wsl_install_system.sh)
  "$tmp/swiftly" init --assume-yes --quiet-shell-followup --skip-install || true
  rm -rf "$tmp"
fi
# shellcheck disable=SC1091
. "${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/env.sh"
# swiftly exits non-zero when apt deps are still missing (wsl_install_system.sh fixes that) — not fatal here.
swiftly install --use latest --assume-yes || echo "(swiftly reported missing system deps — run wsl_install_system.sh with sudo)"
hash -r
swift --version || true

# --- xtool (AppImage) ---------------------------------------------------------
curl -fL "https://github.com/xtool-org/xtool/releases/latest/download/xtool-${ARCH}.AppImage" \
  -o "$HOME/.local/bin/xtool"
chmod +x "$HOME/.local/bin/xtool"
grep -q 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
# swiftly only edits ~/.profile (login shells); interactive WSL shells read ~/.bashrc
grep -q 'swiftly/env.sh' "$HOME/.bashrc" || echo '. "$HOME/.local/share/swiftly/env.sh"' >> "$HOME/.bashrc"
# AppImages need FUSE (libfuse2). Fall back to extract-and-run if it is not installed yet.
"$HOME/.local/bin/xtool" --version 2>/dev/null || APPIMAGE_EXTRACT_AND_RUN=1 "$HOME/.local/bin/xtool" --version
echo "Done. Open a new shell (or 'source ~/.bashrc') to get swift + xtool on PATH."
