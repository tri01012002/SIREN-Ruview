#!/usr/bin/env python3
"""PyTorch CSI pose inference for live demo deployment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm_stats_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)["samples"]
    vectors = []
    for item in data:
        csi = np.asarray(item.get("csi", []), dtype=np.float32)
        if csi.shape == (168,):
            vectors.append(csi)
    if not vectors:
        raise ValueError(f"No 168-dim CSI samples found in {npz_path}")
    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0).clip(min=1e-5)
    return mean, std


def extract_csi_vector(sensing_msg: dict[str, Any], expected_dim: int = 168) -> np.ndarray | None:
    """Build the 168-dim amplitude vector used during training."""
    nodes = sensing_msg.get("nodes") or []
    if not nodes:
        return None

    amps: list[float] = []
    for node in sorted(nodes, key=lambda n: n.get("node_id", 0)):
        node_amp = node.get("amplitude") or []
        if node_amp:
            amps.extend(float(v) for v in node_amp)

    if not amps:
        return None

    # Single-node simulated/live streams may ship 56 subcarriers; training used 3×56.
    if len(amps) == 56 and expected_dim == 168:
        amps = amps * 3
    elif len(amps) < expected_dim:
        while len(amps) < expected_dim:
            chunk = amps[: min(56, expected_dim - len(amps))]
            if not chunk:
                break
            amps.extend(chunk)

    if len(amps) < expected_dim:
        return None
    return np.asarray(amps[:expected_dim], dtype=np.float32)


class PyTorchPoseInferencer:
    """Loads a trained .pth model and runs live CSI → keypoint inference."""

    def __init__(
        self,
        model_path: str | Path,
        npz_path: str | Path,
        model_type: str = "auto",
        max_persons: int = 3,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.npz_path = Path(npz_path)
        self.max_persons = max_persons
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Training npz not found: {self.npz_path}")

        resolved_type = model_type
        if resolved_type == "auto":
            name = self.model_path.name.lower()
            resolved_type = "hybrid" if "hybrid" in name else "csi"
        self.model_type = resolved_type

        self.csi_mean, self.csi_std = _norm_stats_from_npz(self.npz_path)
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

    def _build_model(self) -> torch.nn.Module:
        if self.model_type == "hybrid":
            mod = _load_module("retrain_model", SCRIPTS / "retrain_model.py")
            return mod.HybridCSIModel(num_persons=self.max_persons, pretrained_path=None)
        mod = _load_module("train_csi", SCRIPTS / "train-csi.py")
        return mod.ImprovedCSIModel(num_persons=self.max_persons)

    def _load_weights(self) -> None:
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _normalize(self, csi: np.ndarray) -> np.ndarray:
        return (csi - self.csi_mean) / self.csi_std

    @torch.no_grad()
    def predict(self, csi_raw: np.ndarray) -> list[dict[str, Any]]:
        if csi_raw.shape != (168,):
            return []

        csi = torch.from_numpy(self._normalize(csi_raw)).unsqueeze(0).to(self.device)
        kp_pred, act_pred = self.model(csi)
        kp_flat = kp_pred[0].cpu().numpy()
        activity = int(act_pred[0].argmax().item())

        persons: list[dict[str, Any]] = []
        for person_idx in range(self.max_persons):
            start = person_idx * 34
            end = start + 34
            chunk = kp_flat[start:end]
            energy = float(np.abs(chunk).sum())
            if energy < 1e-3:
                continue

            keypoints = []
            for joint_idx in range(17):
                x = float(np.clip(chunk[joint_idx * 2], 0.0, 1.0))
                y = float(np.clip(chunk[joint_idx * 2 + 1], 0.0, 1.0))
                keypoints.append({
                    "name": KP_NAMES[joint_idx],
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "confidence": 0.85 if activity > 0 else 0.5,
                })

            xs = [kp["x"] for kp in keypoints]
            ys = [kp["y"] for kp in keypoints]
            persons.append({
                "id": person_idx + 1,
                "confidence": 0.85 if activity > 0 else 0.5,
                "keypoints": keypoints,
                "bbox": {
                    "x": float(np.mean(xs)),
                    "y": float(np.mean(ys)),
                    "width": float(max(xs) - min(xs) + 0.05),
                    "height": float(max(ys) - min(ys) + 0.05),
                },
                "zone": "zone_1",
                "_energy": energy,
            })

        max_out = 1 if self.model_type == "csi" else 2
        persons.sort(key=lambda p: p.pop("_energy", 0.0), reverse=True)
        return persons[:max_out]

    def predict_from_sensing(self, sensing_msg: dict[str, Any]) -> list[dict[str, Any]]:
        csi = extract_csi_vector(sensing_msg)
        if csi is None:
            return []
        return self.predict(csi)
