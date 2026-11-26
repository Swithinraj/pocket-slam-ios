# Troubleshooting: problems we hit and how they were solved

Each entry: symptom → cause → fix, with the links that helped.

## 1. Install hangs: "message was too large (65536 bytes, max = 65535)"

**Symptom.** `xtool dev` builds and signs, then hangs at `[Connecting] 100%`. `usbmuxd -f -v` in WSL logs
`device_control_input: ERROR (on device): asyncReadComplete, message was too large (65536 bytes, max = 65535)`.

**Cause.** usbmuxd running inside WSL with the iPhone passed through by usbipd: one packet exceeds the
device's limit over USB/IP. Open upstream, no fix in usbmuxd or usbipd-win.

**Fix.** Do not pass the phone through. Let Windows' Apple Mobile Device service handle USB and point
xtool at it over TCP: `scripts/windows_usbmux_bridge.ps1` (netsh portproxy + firewall) and
`scripts/wsl_use_windows_usbmux.sh` (`USBMUXD_SOCKET_ADDRESS=<WSL gateway>:27015`).

* xtool issue #19 (workaround via Windows' service): <https://github.com/xtool-org/xtool/issues/19>
* usbmuxd issue #247: <https://github.com/libimobiledevice/usbmuxd/issues/247>
* usbipd-win issue #959: <https://github.com/dorssel/usbipd-win/issues/959>
* Microsoft, connecting USB devices to WSL: <https://learn.microsoft.com/en-us/windows/wsl/connect-usb>

## 2. `xtool devices` waits forever although `lsusb` shows the phone (usbipd route)

**Cause.** WSL without systemd: udev never starts `usbmuxd` when the phone is attached.
**Fix.** `sudo usbmuxd -f -v` by hand. Obsolete with the bridge of entry 1.

## 3. Windows does not see the phone / port 27015 has 0 devices

**Causes and fixes.**
* The phone is still *shared/attached* by usbipd (its driver hides the device from Windows):
  `usbipd detach --busid <id>` and `usbipd unbind --busid <id>` (admin), replug.
* An `usbipd attach --auto-attach` loop keeps re-grabbing it: stop that window/process.
* The Microsoft Store **iTunes** service (MobileDevice 8.x) listened on 27015 but reported **0 devices**
  with the current Apple USB driver (`appleusb.inf` 538). Installing **Apple Devices** fixed it:
  <https://apps.microsoft.com/detail/9np83lwlpz9k>.

## 4. `xtool devices` → `Error: unknown`

**Cause.** With Apple Devices the phone is listed twice with the same UDID (USB and Wi-Fi); xtool 1.24
fails on the network entry. **Fix.** Restrict to USB: `xtool devices --usb`, `xtool dev run --usb`
(the `xdevices` / `xdev` aliases do this).

## 5. `swift: command not found` after installing with swiftly

**Cause.** swiftly writes its PATH line to `~/.profile`, which interactive WSL shells do not read.
**Fix.** `scripts/wsl_install_user.sh` also adds `. ~/.local/share/swiftly/env.sh` to `~/.bashrc`.
Swift on Linux also needs `libncurses-dev` etc. (`scripts/wsl_install_system.sh`).
Swift install docs: <https://www.swift.org/install/linux/>

## 6. `Unable to locate package libimobiledevice-1.0-6`

Ubuntu 24.04 names it `libimobiledevice6`; it comes in with `libimobiledevice-utils` (fixed in the script).

## 7. ARKit delivers 30 fps / long gaps; frames "missed"

**Symptoms.** Recordings with 600-800 ms gaps, or a steady 33 ms frame interval with a 60 fps format.
**Causes.**
* The recorder held ARKit's `capturedImage` buffers while encoding, starving ARKit's small buffer pool,
  so ARKit skipped frames. **Fix:** copy the image into an own `CVPixelBufferPool` immediately and encode
  in parallel (`Recorder.copyRGB`).
* Heavy work during the scan (live VLIO, depth upsampling, PNG encoding) heated the phone; iOS lowers
  the camera rate when warm or in Low Power Mode. **Fix:** scan = raw capture only, everything derived runs
  after the scan; thermal state and Low Power Mode are logged in `meta.json`.
* Result: 60.0 fps, 0 missed frames, nominal thermal state over a 198 s scan.

Apple: [Responding to power notifications](https://developer.apple.com/documentation/xcode/responding-to-power-notifications) ·
[Reducing disk writes](https://developer.apple.com/documentation/xcode/reducing-disk-writes) ·
[Scheduling CPU work efficiently](https://developer.apple.com/documentation/xcode/scheduling-cpu-work-efficiently) ·
[ARConfiguration.VideoFormat](https://developer.apple.com/documentation/arkit/arconfiguration/videoformat)

## 8. Post-scan processing very slow (15.6 min for a 3.3 min scan)

**Causes.** `.utility` QoS (efficiency cores), a **debug** Swift build (`xtool dev` default), and
per-pixel Swift loops. **Fix.** `.userInitiated` while the user watches the progress, `vDSP`
(Accelerate) for the per-pixel conversions, and install with `-c release` (`xdev` alias).
Background completion uses [`beginBackgroundTask`](https://developer.apple.com/documentation/uikit/uiapplication/beginbackgroundtask(expirationhandler:)).

## 9. "Low-resolution depth" (e.g. in Polycam)

ARKit's LiDAR depth is **256x192** (hardware output). AVFoundation's LiDAR Depth Camera gives
320x240 (stream) / 768x576 (photo), but Apple produces it by fusing LiDAR and RGB with an ML model, and it
cannot run together with ARKit tracking. This app stores the raw 256x192 LiDAR **and** a depth map
pixel-aligned with the RGB image at the RGB size (Core Image edge-preserving upsample). For
higher-resolution *metric* depth, Prompt Depth Anything (LiDAR-prompted, up to 4K) is the strongest
option offline.

* Apple, [Capturing depth using the LiDAR camera](https://developer.apple.com/documentation/avfoundation/capturing-depth-using-the-lidar-camera) ·
  [WWDC22: advancements in iOS camera capture](https://developer.apple.com/videos/play/wwdc2022-110429)
* Prompt Depth Anything (CVPR 2025): <https://promptda.github.io>
* Depth Anything V2 Core ML (relative depth, on device): <https://huggingface.co/apple/coreml-depth-anything-v2-small>

## 10. Offline tools

* `cv2.findFundamentalMat(..., USAC_MAGSAC)` can **assert** on degenerate pairs instead of returning None
  (OpenCV 5.1): wrapped in `try/except` in `tools/slamtogo/frontend.py`.
* Open3D 0.20 `ScalableTSDFVolume` integrated nothing in this build; `tools/run_ba.py` uses the tensor
  `VoxelBlockGrid` (uint16 depth + **uint8** colour).
* PowerShell → `wsl bash -c "..."` mangles quotes and `$`: put commands in a script file and run
  `wsl bash script.sh`. Scripts must have LF line endings (`.gitattributes` enforces it).

## 11. Our VLIO looked "5-7 % too small"

Not a scale bug: a whole-trajectory Sim(3) fit of a trajectory with heading drift reads as shrinking.
Over 1-5 s motions the scale is right (see [FINDINGS](FINDINGS.md#4-scale-local-vs-whole-trajectory)).
Use the **local scale** column of `tools/eval_trajectories.py`.
