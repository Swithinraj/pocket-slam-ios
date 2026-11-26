# Run in an *Administrator* PowerShell.
# Workaround for usbmuxd-over-usbipd failing ("message was too large (65536 bytes, max = 65535)").
# Instead of passing the iPhone's USB into WSL, Windows' own Apple Mobile Device service (from iTunes /
# Apple Devices) talks to the phone, and WSL reaches that service over TCP port 27015.
# See xtool-org/xtool#19.
#
# Usage (from the repo folder):  powershell -ExecutionPolicy Bypass -File .\scripts\windows_usbmux_bridge.ps1
# Re-run after a reboot (the WSL host IP can change).
$ErrorActionPreference = 'Stop'
$Port = 27015

# 1. Give the iPhone back to Windows if usbipd still owns it
if (Get-Command usbipd -ErrorAction SilentlyContinue) {
    $line = usbipd list | Select-String -Pattern 'Apple|iPhone' | Select-Object -First 1
    if ($line) {
        $busid = ($line.ToString().Trim() -split '\s+')[0]
        Write-Host "Releasing iPhone (BUSID $busid) from usbipd..."
        try { usbipd detach --busid $busid 2>$null } catch {}
        try { usbipd unbind --busid $busid 2>$null } catch {}
    }
}

# 2. Apple Mobile Device service must be listening on 127.0.0.1:27015
$listen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listen) {
    Write-Warning "Nothing is listening on port $Port. Open iTunes (or the Apple Devices app), plug in + unlock the iPhone, tap 'Trust', then re-run this script."
    exit 1
}
Write-Host "Apple Mobile Device service is listening on port $Port."

# 3. Windows host IP as seen from WSL (NAT mode)
$route = (wsl -d Ubuntu -- ip route show default | Select-Object -First 1)   # "default via 172.x.x.1 dev eth0 ..."
$gw = (($route -split '\s+') | Where-Object { $_ })[2]
if (-not ($gw -as [ipaddress])) { Write-Error 'Could not determine the WSL gateway IP.'; exit 1 }
Write-Host "WSL reaches Windows at $gw"

# 4. Forward <gw>:27015 -> 127.0.0.1:27015 (replace any old 27015 forwards)
netsh interface portproxy show v4tov4 | Select-String ":?\s*$Port\s" | ForEach-Object {
    $addr = ($_.ToString().Trim() -split '\s+')[0]
    netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=$addr | Out-Null
}
netsh interface portproxy add v4tov4 listenaddress=$gw listenport=$Port connectaddress=127.0.0.1 connectport=$Port | Out-Null

# 5. Allow inbound 27015 only from the WSL virtual network
$ruleName = 'pocket-slam usbmuxd from WSL'
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port `
    -LocalAddress $gw -RemoteAddress LocalSubnet -Action Allow | Out-Null

netsh interface portproxy show v4tov4
Write-Host "`nDone. In WSL, from the repo root, run:  source scripts/wsl_use_windows_usbmux.sh"
