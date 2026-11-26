# Source this in WSL from the repo root:   source scripts/wsl_use_windows_usbmux.sh
# Points xtool / libimobiledevice at Windows' Apple Mobile Device service (see windows_usbmux_bridge.ps1)
# and defines `xdevices` / `xdev` shortcuts for this repo.
POCKET_SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIN_HOST="$(ip route show default | awk '{print $3}')"
export USBMUXD_SOCKET_ADDRESS="${WIN_HOST}:27015"
# A local usbmuxd would compete for the device; stop it if running.
pgrep -x usbmuxd >/dev/null && { echo "Stopping local usbmuxd..."; sudo pkill -x usbmuxd; }
echo "USBMUXD_SOCKET_ADDRESS=$USBMUXD_SOCKET_ADDRESS"
# Apple Devices also lists the phone over Wi-Fi with the same UDID; xtool 1.24 then fails with
# "Error: unknown". Restrict xtool to USB. Release build: optimised Swift (post-scan processing
# is ~10x slower in a debug build).
alias xdevices='xtool devices --usb'
alias xdev="(cd \"$POCKET_SLAM_ROOT/App\" && xtool dev run --usb -c release)"
echo "Use: xdevices  (= xtool devices --usb)   |   xdev  (= build + install the app, release)"
if timeout 3 bash -c "</dev/tcp/${WIN_HOST}/27015" 2>/dev/null; then
  echo "Windows usbmux service reachable."
else
  echo "Cannot reach ${WIN_HOST}:27015 - is Apple Devices open and windows_usbmux_bridge.ps1 run as admin?"
fi
