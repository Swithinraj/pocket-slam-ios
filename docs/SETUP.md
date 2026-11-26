# Setup: build and run on an iPhone from Windows (no Mac)

Everything below was run on Windows 11 with WSL2. All commands are run from the repo root unless noted.
Expected outputs are shown so each step can be verified.

## 0. Requirements

| Item | Version used | Notes |
|---|---|---|
| iPhone with LiDAR | iPhone 15 Pro, iOS 27.2.1 | any LiDAR iPhone/iPad Pro should work |
| Windows | 11 | admin rights needed once |
| WSL2 distro | Ubuntu 24.04 | `wsl --install -d Ubuntu-24.04` |
| Swift | 6.4.3 | installed by `scripts/wsl_install_user.sh` (swiftly) |
| xtool | 1.24.0 | <https://github.com/xtool-org/xtool> |
| Xcode `.xip` | 27.1 | only used to extract the iOS SDK; download needs an Apple ID |
| Apple Devices app | Microsoft Store | provides the Windows iPhone USB service |
| Apple ID | free account works | paid account: 1-year signing, no 7-day expiry |

## 1. WSL toolchain (once)

```bash
git clone https://github.com/Swithinraj/pocket-slam-ios.git && cd pocket-slam-ios
sudo bash scripts/wsl_install_system.sh      # apt: usbmuxd, libimobiledevice-utils, libfuse2t64, Swift deps
bash scripts/wsl_install_user.sh             # swiftly -> Swift 6.4.3, xtool AppImage -> ~/.local/bin
source ~/.bashrc
swift --version                              # Swift version 6.4.3 (swift-6.4.3-RELEASE)
xtool --version                              # xtool 1.24.0
```

## 2. iOS SDK and Apple ID (once)

1. Download **Xcode 27.1** (`Xcode_27.1.xip`) in a browser from
   <https://developer.apple.com/download/all/?q=Xcode> (Apple ID login). Put it somewhere without
   spaces in the path, e.g. `C:\dev\Xcode_27.1.xip`.
2. In WSL:
   ```bash
   xtool setup
   # Choice (0-1): 1        (password mode, works with any Apple ID; 0 = API key, paid account)
   # Apple ID / password / 2FA code
   # Path to Xcode.xip: /mnt/c/dev/Xcode_27.1.xip
   # ... [Installing SDKs] - iPhoneOS27.1.sdk ...
   swift sdk list                            # darwin
   ```
   xtool's own guide: [Installation (Linux)](https://github.com/xtool-org/xtool/blob/main/Documentation/xtool.docc/Installation-Linux.md).
   Password mode uses private Apple APIs; xtool recommends a secondary Apple ID.

## 3. Connect the iPhone to WSL (once per boot)

USB passthrough with usbipd **does not work** for installing apps (see
[TROUBLESHOOTING](TROUBLESHOOTING.md#1-install-hangs-message-was-too-large-65536-bytes-max--65535)). Instead,
Windows' own Apple Mobile Device service talks to the phone and WSL reaches it over TCP 27015.

1. Install **Apple Devices** from the Microsoft Store (`winget install --id 9NP83LWLPZ9K --source msstore`).
   The Store version of iTunes ships an older device service that may not see the phone.
2. Plug in the iPhone, unlock it, tap **Trust**. Apple Devices must show the phone.
3. On the iPhone: **Settings → Privacy & Security → Developer Mode → On** (it restarts).
4. Admin PowerShell, from the repo folder (re-run after every Windows reboot, the WSL IP changes):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\windows_usbmux_bridge.ps1
   # Apple Mobile Device service is listening on port 27015.
   # WSL reaches Windows at 172.x.x.1
   ```
   It adds a `netsh portproxy` from the WSL gateway to `127.0.0.1:27015` and a firewall rule limited to the
   WSL subnet.
5. WSL:
   ```bash
   source scripts/wsl_use_windows_usbmux.sh  # sets USBMUXD_SOCKET_ADDRESS, defines xdevices / xdev
   # Windows usbmux service reachable.
   xdevices                                  # <your iPhone> [usb]: 0000XXXX-XXXXXXXXXXXXXXXX
   ```

## 4. Build, install, run

```bash
xdev            # = cd App && xtool dev run --usb -c release
```

The first build takes a few minutes (C++ core with Eigen at `-O3` + the Swift app). Always use the
**release** configuration: the post-scan processing is roughly 10x slower in a debug build.

First launch on the phone: **Settings → General → VPN & Device Management → <Apple ID> → Trust**, then open
**SLAM to Go** and allow camera and motion access. With a free Apple ID the app expires after 7 days;
run `xdev` again to re-sign.

Build only (no phone needed): `cd App && xtool dev build -c release`.

## 5. Scan

* Red button: start / stop. While recording the screen has a red border and a badge with time, RGB-D
  frames, size, dropped and missed frames.
* After **stop**, processing starts by itself (banner with progress). It writes the RGB-aligned depth and
  runs the VLIO core over the scan. It pauses while you record the next scan and resumes afterwards.
* Folder button: recordings with their processing state, share (zip), delete, **Process** for older scans.
* Settings: camera rate (60 / 30 fps), RGB-D rate and size, mesh export, live VLIO preview.

Tips for good data: move slowly, keep textured surfaces 0.5-3 m away, avoid pointing at blank walls, and
walk a loop back to where you started (helps the offline BA and, later, loop closure).

## 6. Get recordings onto the PC

Any of: Files app → On My iPhone → SLAM to Go; the share button (zip → AirDrop, cloud drive);
Apple Devices on Windows → your iPhone → Files → SLAM to Go. To fetch zips shared through a Google Drive
folder: `gdown "<folder url>" --continue` (installed with the Python tools).

## 7. Analyse on the PC

```bash
bash tools/setup_env.sh                       # ~/venvs/slamtogo with the pinned packages
source ~/venvs/slamtogo/bin/activate
REC=/path/to/recordings/2026-11-26_14-08-02

python tools/inspect_recording.py $REC        # rates, gaps, IMU axes vs ARKit, depth encoding/alignment
python tools/run_ba.py $REC                   # offline BA reference + TSDF meshes  -> $REC/_slam/
python tools/eval_trajectories.py $REC        # ARKit vs BA vs VLIO (ATE, scale, local scale, LiDAR consistency)
python tools/view_reconstruction.py $REC      # 3D viewer: P points/mesh, B BA/ARKit, T trajectories, L ARKit mesh

# replay the C++ core exactly as the app runs it (Linux build)
cmake -S core -B ~/build/slamcore -DCMAKE_BUILD_TYPE=Release && cmake --build ~/build/slamcore -j
~/build/slamcore/test_core                    # synthetic ground-truth tests -> ALL PASSED
mkdir -p $REC/_vlio && ~/build/slamcore/run_recording $REC $REC/_vlio
bash tools/run_ablation.sh $REC               # ablation variants -> $REC/_var/

# regenerate every figure and number in docs/
python tools/make_figures.py --recordings /path/to/recordings --out docs --with-triangulation
```

The 3D viewer opens on the Windows desktop through [WSLg](https://github.com/microsoft/wslg).

## Building on a Mac instead

xtool also runs on macOS, and the SwiftPM packages in `App/` and `core/` open in Xcode
(`xtool dev generate-xcode-project`), so the Windows/WSL connection steps (3) are not needed there.
This path was not used for the results in this repo.
