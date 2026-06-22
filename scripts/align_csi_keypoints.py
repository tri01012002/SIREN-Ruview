#!/usr/bin/env python3
"""
Script Align CSI + Keypoints theo timestamp
Dùng để chuẩn bị data train model WiFi Pose Estimation
"""

from collections import defaultdict
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def load_keypoints(file_path: Path) -> List[Dict]:
    """Load keypoints JSONL"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_csi(file_path: Path) -> List[Dict]:
    """Load CSI recording JSONL"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def filter_by_confidence(keypoints_list: List[Dict], min_conf: float = 0.5) -> List[Dict]:
    """Lọc frame có confidence thấp"""
    filtered = []
    dropped = 0
    for kp in keypoints_list:
        if kp.get("n_persons", 0) == 0:
            dropped += 1
            continue
        # Lấy confidence cao nhất trong các người
        max_conf = max((p.get("confidence", 0.0) for p in kp.get("persons", [])), default=0.0)
        if max_conf >= min_conf:
            filtered.append(kp)
        else:
            dropped += 1
    print(f"Đã lọc confidence < {min_conf}: Giữ {len(filtered)} / {len(keypoints_list)} frames")
    return filtered

def align_data(keypoints_list: List[Dict], csi_list: List[Dict], 
               time_tolerance_ms: int = 150) -> Tuple[List[Dict], Dict]:
    """Align CSI + Multi-Person Keypoints"""
    aligned = []
    stats = defaultdict(int)
    csi_idx = 0
    csi_len = len(csi_list)
    
    for kp in keypoints_list:
        kp_time_ms = kp["ts_ns"] / 1_000_000  # convert ns → ms
        
        # Tìm CSI frame gần nhất
        best_csi = None
        min_diff = float('inf')
        
        for i in range(csi_idx, len(csi_list)):
            csi = csi_list[i]
            csi_time_ms = csi.get("timestamp", 0) * 1000
            
            diff = abs(kp_time_ms - csi_time_ms)
            if diff < min_diff:
                min_diff = diff
                best_csi = csi
            if csi_time_ms > kp_time_ms + time_tolerance_ms:
                break
            if diff < 50:  # Tìm được frame rất gần thì dừng sớm
                break

        if best_csi and min_diff <= time_tolerance_ms:
            aligned.append({
                "timestamp_ms": kp_time_ms,
                "keypoints": kp.get("persons", []),   # Multi-person
                "csi": best_csi,
                "time_diff_ms": round(min_diff, 2)
            })
            stats["aligned"] += 1
        else:
            stats["dropped"] += 1

    stats["total_keypoints"] = len(keypoints_list)
    stats["total_csi"] = len(csi_list)
    stats["align_rate"] = len(aligned) / len(keypoints_list) * 100 if keypoints_list else 0

    return aligned, stats

def save_as_npz(aligned_data: List[Dict], output_path: Path):
    """Chuyển sang định dạng .npz cho PyTorch training"""
    samples = []
    
    for item in aligned_data:
        # Keypoints từ tất cả người
        kp_list = []
        for person in item["keypoints"]:
            kp_flat = []
            for point in person.get("keypoints", []):
                kp_flat.extend(point)  # [x, y] → flatten
            kp_list.append(kp_flat)
        
        # CSI amplitude (lấy node đầu tiên hoặc merge)
        csi_amp = []
        for node in item["csi"].get("nodes", []):
            csi_amp.extend(node.get("amplitude", []))
        
        samples.append({
            "keypoints": np.array(kp_list, dtype=np.float32),   # shape: (N_person, 34)
            "csi": np.array(csi_amp, dtype=np.float32),
            "timestamp": item["timestamp_ms"]
        })

    np.savez_compressed(output_path, samples=samples)
    print(f"Đã lưu .npz: {len(samples)} samples")


def plot_time_diff_histogram(aligned_data: List[Dict], save_path: Path = None):
    diffs = [item["time_diff_ms"] for item in aligned_data]
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Time Difference between CSI and Keypoints')
    plt.xlabel('Time Difference (ms)')
    plt.ylabel('Number of Frames')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Align CSI + Multi-Person Keypoints")
    #parser.add_argument("--keypoints", required=True, help="Path to keypoints .jsonl")
    #parser.add_argument("--csi", required=True, help="Path to CSI .jsonl")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Minimum confidence")
    parser.add_argument("--tolerance", type=int, default=150, help="Time tolerance (ms)")
    parser.add_argument("--output", default=None, help="Output .npz file")
    args = parser.parse_args()

    """Hugos code"""
    _folder_path = Path("../data/recordings")
    for i, file in enumerate(_folder_path.glob("keypoint*.jsonl")):
        print(f"{i}: {file.name}")
    keypoint_index = int(input("Select your keypoint file"))
    keypoint_filename = list(_folder_path.glob("keypoint*.jsonl"))[keypoint_index]
    for i, file in enumerate(_folder_path.glob("rec*.jsonl")):
        print(f"{i}: {file.name}")
    csi_index = int(input("Select your csi file"))
    csi_filename = list(_folder_path.glob("rec*.jsonl"))[csi_index]
    keypoints = load_keypoints(Path(keypoint_filename))
    csi_data = load_csi(Path(csi_filename))
    
    """End of Hugos code"""

    # Load data
    print("Đang load dữ liệu...")
    #keypoints = load_keypoints(Path(args.keypoints))
    #csi_data = load_csi(Path(args.csi))

    # Lọc confidence
    keypoints = filter_by_confidence(keypoints, args.min_conf)

    # Align
    print("Đang align dữ liệu...")
    aligned, stats = align_data(keypoints, csi_data, args.tolerance)

    # Save aligned JSONL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #jsonl_out = Path(args.keypoints).parent / f"aligned_{timestamp}.jsonl"
    jsonl_out = Path(keypoint_filename).parent / f"aligned_{timestamp}.jsonl"
    
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for item in aligned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save .npz cho training
    #npz_path = Path(args.output) if args.output else Path(args.keypoints).parent / f"train_data_{timestamp}.npz"
    npz_path = Path(args.output) if args.output else Path(keypoint_filename).parent / f"train_data_{timestamp}.npz"
    save_as_npz(aligned, npz_path)

    # Statistics
    print("\n" + "="*60)
    print("📊 ALIGNMENT STATISTICS")
    print("="*60)
    print(f"Tổng keypoints sau lọc : {stats['total_keypoints']}")
    print(f"Tổng CSI frames        : {stats['total_csi']}")
    print(f"Frames được align      : {stats['aligned']}")
    print(f"Tỷ lệ align            : {stats['align_rate']:.2f}%")
    print(f"Frames bị drop         : {stats.get('dropped', 0)}")
    print(f"File JSONL aligned     : {jsonl_out}")
    print(f"File .npz (PyTorch)    : {npz_path}")

    # Plot histogram
    plot_time_diff_histogram(aligned, npz_path.with_suffix('.png'))


if __name__ == "__main__":
    main()