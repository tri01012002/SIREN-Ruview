# WiFi-DensePose Live Setup — Real Data from 3 ESP32 Nodes

## ✅ Fixes Applied

### 1. WebSocket Connection Logic (ui/observatory/js/hud-controller.js)
**Problem:** Event listener for URL input used `'change'` instead of `'input'`
- `'change'` only triggers when input loses focus
- User could enter URL + switch to WS mode before blur
- Result: URL not saved to state → connection not initiated

**Fix Applied:**
- Changed from `addEventListener('change', ...)` to `addEventListener('input', ...)`
- Added immediate reconnect when URL updated
- Also kept blur handler for safety
- Now real-time connection when user types or pastes URL

### 2. Server Network Binding
**Problem:** Default `--bind-addr 127.0.0.1` only allows localhost
- Remote Observatory from 192.168.137.x cannot connect

**Fix Applied:**
- Server already supports `--bind-addr` flag correctly
- **Always run with:** `--bind-addr 0.0.0.0`
- Now listening on all interfaces ✅

**Status:** ✅ Server running on 0.0.0.0:8765 (WebSocket) and 0.0.0.0:8080 (HTTP)

---

## 📋 Complete Setup Steps

### Step 1: Provision Node 3 (COM8) — If Not Done
```powershell
cd D:\SIREN-Ruview

# Generate NVS binary
python firmware\esp32-csi-node\provision.py `
  --port COM8 `
  --ssid "TRINGUYEN" `
  --password "1234567890" `
  --target-ip 192.168.137.1 `
  --node-id 3 `
  --edge-tier 1 `
  --dry-run

# Flash to ESP32
python -m esptool --chip esp32s3 --port COM8 write-flash 0x9000 nvs_provision.bin
```

**Expected output:** `Hash of data verified` + `Hard resetting via RTS pin`

### Step 2: Restart All 3 ESP32 Nodes
- **COM5** (node 1): Unplug → wait 2s → plug back
- **COM8** (node 2): Unplug → wait 2s → plug back  
- **COM9** (node 3): Unplug → wait 2s → plug back

*Monitor any one with:* `python -m serial.tools.miniterm COM5 115200`
- Look for: `WiFi connected to TRINGUYEN` + `Sending CSI to 192.168.137.1:5005`

### Step 3: Rust Server Already Running ✅
```bash
# Server is already running with:
cargo run -p wifi-densepose-sensing-server --release -- --bind-addr 0.0.0.0
```

Status: 
```
UDP listening on 0.0.0.0:5005 for ESP32 CSI frames ✅
WebSocket server listening on 0.0.0.0:8765 ✅
HTTP server listening on 0.0.0.0:8080 ✅
```

### Step 4: Open Observatory & Connect

1. **Open browser** → `http://192.168.137.1:8080`
   - *Not* `localhost` — must use actual IP for network Observatory!
   
2. **Wait 2-3 seconds** for auto-detect
   - Observatory probes `/health` endpoint at server IP
   
3. If not auto-detected, **manual connect:**
   - Click ⚙️ **Settings** button (top-right)
   - Go to **Data** tab
   - **Data Source** → Select "Live WebSocket"
   - **WS URL** → Paste: `ws://192.168.137.1:8765/ws/sensing`
   - Press Enter or Tab to confirm

4. **Badge should change:**
   - FROM: 🔴 **DEMO** (red dot, left side of screen)
   - TO: 🟢 **LIVE** (green dot, animated)
   - Within 2-3 seconds of URL entry

5. **Verify data flow:**
   - 3D pose skeleton animates (real WiFi motion)
   - Vital Signs panel shows HR/BR values updating
   - Signal panel shows RSSI + motion values changing

---

## 🐛 Troubleshooting

### Symptom: Badge still 🔴 DEMO after entering WS URL

**Debug steps:**
1. Open DevTools (F12) → **Console** tab
2. Look for:
   - `WebSocket connection failed` → Network/firewall issue
   - `Connection refused` → Server not running or wrong port
   - `invalid URL` → Check URL format (ws://, not http://)

3. Try manual **refresh + reconnect:**
   - Press F5 to reload Observatory
   - Immediately go to Settings → Data
   - Re-enter URL and confirm

4. Check server logs in Rust terminal:
   - Should see `WebSocket connection from 192.168.137.x`
   - If nothing: server may not be running, or firewall blocks it

### Symptom: Server not receiving UDP from ESP32

**Debug steps:**
1. Check each ESP32 console:
   ```bash
   python -m serial.tools.miniterm COM5 115200
   ```
   Must see: `WiFi connected to TRINGUYEN` + `Sending CSI to 192.168.137.1:5005`
   
2. If WiFi not connected:
   - Credentials wrong (check SSID/password match provision)
   - WiFi not broadcasting on 2.4 GHz (ESP32-S3 doesn't support 5 GHz)
   - Node too far from router
   
3. If WiFi connected but no CSI:
   - Try restarting ESP32 (unplug + replug)
   - Or: reflash firmware with proper CSI mode

4. Windows Firewall blocking UDP:
   - Settings → Firewall & network protection
   - Advanced settings → Inbound Rules
   - Check if port 5005 allowed for Python/Rust

### Symptom: WebSocket connects but no pose data shown

**Debug steps:**
1. Check settings:
   - **Scenario:** Don't keep on "auto" (cycles every 30s)
   - Try specific scenario: "VITAL SIGNS" or "MULTI-PERSON"
   
2. Check signal quality:
   - Signal panel shows RSSI: should be > -80 dBm for good signal
   - If RSSI near -100: ESP32 too far or weak signal
   
3. Check server logs for errors:
   - "CSI frame invalid" → Data format issue
   - "No keypoints detected" → Pose tracking initializing
   - Look for 3+ UDP packets per second from nodes

---

## 📊 Expected Data Flow

```
ESP32 Node 1 (192.168.137.x)
    ↓ UDP CSI frame (20ms intervals)
ESP32 Node 2 (192.168.137.x)
    ↓ UDP CSI frame
ESP32 Node 3 (192.168.137.x)
    ↓ UDP CSI frame
    
⬇️⬇️⬇️

Rust Server (0.0.0.0:5005)
    ↓ Receive UDP
    ↓ Parse ESP32 frame (magic 0xC511_0001)
    ↓ Process signal (RuVector + RuvSense)
    ↓ Extract pose + vitals
    ↓ Broadcast to WebSocket

⬇️⬇️⬇️

Observatory (WebSocket 0.0.0.0:8765)
    ↓ Receive SensingUpdate JSON
    ↓ Parse pose keypoints (17-point skeleton)
    ↓ Render 3D in Three.js
    ↓ Update vital signs HUD
    
✅ Real-time pose + vitals displayed
```

---

## ✅ Final Validation

Run through all items:
- [ ] COM8 provisioned & flashed
- [ ] All 3 ESP32 nodes restarted
- [ ] Rust server running with `--bind-addr 0.0.0.0`
- [ ] Observatory opened at `http://192.168.137.1:8080` (not localhost)
- [ ] Badge changed from 🔴 DEMO → 🟢 LIVE
- [ ] 3D pose skeleton visible & animating
- [ ] HR/BR values updating in Vital Signs panel
- [ ] Real WiFi CSI data flowing (not simulator demo data)

**Success:** All items ✅ = Live WiFi-DensePose system ready!
