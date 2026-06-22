#!/usr/bin/env python3
"""
Live demo launcher: Rust sensing backend + PyTorch pose inference + web UI.

Architecture
------------
- Rust sensing-server (internal HTTP 3010, public WS 3001 for /ws/sensing)
- This server (public HTTP 3000): serves UI, proxies REST, runs PyTorch pose WS

Usage
-----
  # Full stack with ESP32 (build Rust first: cd v2 && cargo build -p wifi-densepose-sensing-server)
  python scripts/run_live_demo.py \\
      --model scripts/best_hybrid_2person.pth \\
      --npz data/ground-truth/2person/train_data_20260615_150935.npz \\
      --source esp32

  # Simulated CSI (no hardware)
  python scripts/run_live_demo.py \\
      --model scripts/best_csi_pose_model.pth \\
      --npz data/ground-truth/1person/train_data_20260615_150327.npz \\
      --source simulated

  # Pose bridge only (Rust already running on 3010/3001)
  python scripts/run_live_demo.py --model scripts/best_hybrid_2person.pth \\
      --npz data/ground-truth/2person/train_data_20260615_150935.npz \\
      --no-rust
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from pytorch_infer import PyTorchPoseInferencer  # noqa: E402

RUST_HTTP = 3010
RUST_WS = 3001
PUBLIC_HTTP = 3000

KP_POSE_CLIENTS: set[WebSocket] = set()
INFERENCER: PyTorchPoseInferencer | None = None
RUST_PROC: subprocess.Popen | None = None
SENSING_TASK: asyncio.Task | None = None


def _resolve_rust_binary() -> Path | None:
    candidates = [
        ROOT / "v2" / "target" / "release" / "sensing-server.exe",
        ROOT / "v2" / "target" / "release" / "sensing-server",
        ROOT / "v2" / "target" / "debug" / "sensing-server.exe",
        ROOT / "v2" / "target" / "debug" / "sensing-server",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def start_rust_backend(source: str, udp_port: int) -> subprocess.Popen:
    binary = _resolve_rust_binary()
    if binary is None:
        raise RuntimeError(
            "sensing-server binary not found. Build it first:\n"
            "  cd v2\n"
            "  cargo build -p wifi-densepose-sensing-server --no-default-features"
        )

    cmd = [
        str(binary),
        "--source", source,
        "--http-port", str(RUST_HTTP),
        "--ws-port", str(RUST_WS),
        "--tick-ms", "100",
        "--udp-port", str(udp_port),
        "--bind-addr", "127.0.0.1",
    ]
    print(f"[rust] starting: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=ROOT / "v2",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


async def wait_for_rust_health(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"http://127.0.0.1:{RUST_HTTP}/health")
                if resp.status_code == 200:
                    print("[rust] health OK")
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Rust sensing-server did not become healthy within {timeout_s}s")


async def broadcast_pose(pose_msg: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    payload = json.dumps(pose_msg)
    for ws in list(KP_POSE_CLIENTS):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        KP_POSE_CLIENTS.discard(ws)


async def sensing_consumer_loop(sensing_ws_url: str) -> None:
    assert INFERENCER is not None
    while True:
        try:
            async with websockets.connect(sensing_ws_url, ping_interval=20) as ws:
                print(f"[pose] connected to sensing stream: {sensing_ws_url}")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") != "sensing_update" and msg.get("msg_type") != "sensing_update":
                        continue

                    persons = INFERENCER.predict_from_sensing(msg)
                    if not persons:
                        continue

                    pose_msg = {
                        "type": "pose_data",
                        "zone_id": "zone_1",
                        "timestamp": msg.get("timestamp", time.time()),
                        "payload": {
                            "pose": {"persons": persons},
                            "confidence": msg.get("classification", {}).get("confidence", 0.0),
                            "activity": msg.get("classification", {}).get("motion_level", "unknown"),
                            "pose_source": "model_inference",
                            "metadata": {
                                "frame_id": f"pytorch_frame_{msg.get('tick', 0)}",
                                "processing_time_ms": 1,
                                "source": msg.get("source", "unknown"),
                                "tick": msg.get("tick", 0),
                                "model": str(INFERENCER.model_path.name),
                                "estimated_persons": len(persons),
                            },
                        },
                    }
                    await broadcast_pose(pose_msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[pose] sensing stream error: {exc}; reconnecting in 2s")
            await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SENSING_TASK
    sensing_url = app.state.sensing_ws_url
    SENSING_TASK = asyncio.create_task(sensing_consumer_loop(sensing_url))
    yield
    if SENSING_TASK:
        SENSING_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await SENSING_TASK


def create_app(sensing_ws_url: str) -> FastAPI:
    app = FastAPI(title="SIREN PyTorch Live Demo", lifespan=lifespan)
    app.state.sensing_ws_url = sensing_ws_url

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{RUST_HTTP}/health")
                rust_ok = resp.status_code == 200
        except httpx.HTTPError:
            rust_ok = False
        return JSONResponse({
            "status": "ok" if rust_ok and INFERENCER else "degraded",
            "rust_backend": rust_ok,
            "pytorch_model": INFERENCER.model_path.name if INFERENCER else None,
            "pose_clients": len(KP_POSE_CLIENTS),
        })

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_api(path: str, request: Request) -> Response:
        if path.startswith("v1/stream/pose"):
            return JSONResponse({"error": "use WebSocket /api/v1/stream/pose"}, status_code=400)
        url = f"http://127.0.0.1:{RUST_HTTP}/api/{path}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(request.method, url, content=body, headers=headers, params=request.query_params)
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

    @app.api_route("/{path:path}", methods=["GET"], include_in_schema=False)
    async def proxy_root(path: str, request: Request) -> Response:
        skip = ("ui", "health", "api", "ws")
        if path.split("/")[0] in skip or not path:
            return JSONResponse({"error": "not found"}, status_code=404)
        url = f"http://127.0.0.1:{RUST_HTTP}/{path}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=request.query_params)
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.websocket("/api/v1/stream/pose")
    async def ws_pose(websocket: WebSocket) -> None:
        await websocket.accept()
        KP_POSE_CLIENTS.add(websocket)
        await websocket.send_json({
            "type": "connection_established",
            "payload": {
                "status": "connected",
                "backend": "pytorch",
                "model": INFERENCER.model_path.name if INFERENCER else None,
            },
        })
        try:
            while True:
                msg = await websocket.receive_text()
                if msg.strip().lower() in ("ping", '{"type":"ping"}'):
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            KP_POSE_CLIENTS.discard(websocket)

    ui_dir = ROOT / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

        @app.get("/")
        async def root_redirect() -> FileResponse:
            return FileResponse(ui_dir / "index.html")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live demo with PyTorch pose model")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--npz", required=True, help="Training .npz for CSI normalization stats")
    parser.add_argument("--model-type", choices=["auto", "csi", "hybrid"], default="auto")
    parser.add_argument("--source", default="simulated", choices=["simulated", "simulate", "esp32", "auto", "wifi"])
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--port", type=int, default=PUBLIC_HTTP)
    parser.add_argument("--rust-http", type=int, default=RUST_HTTP)
    parser.add_argument("--rust-ws", type=int, default=RUST_WS)
    parser.add_argument("--no-rust", action="store_true", help="Skip spawning Rust (already running)")
    parser.add_argument("--sensing-url", default=None, help="Override sensing WebSocket URL")
    return parser.parse_args()


def main() -> None:
    global INFERENCER, RUST_PROC, RUST_HTTP, RUST_WS

    args = parse_args()
    RUST_HTTP = args.rust_http
    RUST_WS = args.rust_ws

    source = "simulate" if args.source == "simulated" else args.source

    print(f"[pytorch] loading model: {args.model}")
    INFERENCER = PyTorchPoseInferencer(
        model_path=args.model,
        npz_path=args.npz,
        model_type=args.model_type,
    )
    print(f"[pytorch] ready on {INFERENCER.device} ({INFERENCER.model_type})")

    if not args.no_rust:
        RUST_PROC = start_rust_backend(source, args.udp_port)
        asyncio.run(wait_for_rust_health())
    else:
        print("[rust] skipped (--no-rust); expecting backend already running")

    sensing_ws_url = args.sensing_url or f"ws://127.0.0.1:{RUST_WS}/ws/sensing"
    app = create_app(sensing_ws_url)

    print()
    print("=" * 60)
    print("  Live demo ready")
    print(f"  Dashboard : http://127.0.0.1:{args.port}/ui/index.html")
    print(f"  Live Demo : http://127.0.0.1:{args.port}/ui/index.html  ->  Live Demo tab")
    print(f"  Model     : {Path(args.model).name}")
    print(f"  CSI source: {source}")
    print("  Steps:")
    print("    1. Open the Dashboard URL above")
    print("    2. Go to the 'Live Demo' tab")
    print("    3. Click 'Start Detection'")
    print("=" * 60)
    print()

    def _shutdown(*_args: Any) -> None:
        if RUST_PROC and RUST_PROC.poll() is None:
            RUST_PROC.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
