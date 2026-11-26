#!/usr/bin/env bash
# System packages for Swift + xtool + iPhone over USB. Needs sudo (asks for your WSL password once).
# Usage (inside WSL Ubuntu, from the repo root):  sudo bash scripts/wsl_install_system.sh
set -euo pipefail
apt-get update
apt-get install -y \
  usbmuxd libimobiledevice-utils \
  libfuse2t64 unzip zip gnupg2 \
  libcurl4-openssl-dev libxml2-dev libncurses-dev libz3-dev \
  binutils libc6-dev libedit2 libgcc-13-dev libpython3-dev libsqlite3-0 \
  libstdc++-13-dev pkg-config tzdata zlib1g-dev usbutils
echo "System deps installed."
