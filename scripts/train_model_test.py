import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch.cuda.amp as amp  # Mixed Precision

class CSIPoseDataset(Dataset):
    def __init__(self, npz_path: str, max_persons=3):
        data = np.load(npz_path, allow_pickle=True)['samples']
        self.samples = []
        self.max_persons = max_persons

        for item in data:
            csi = torch.tensor(item['csi'], dtype=torch.float32)
            keypoints = torch.tensor(item.get('keypoints', []), dtype=torch.float32)

            # Pad keypoints
            if len(keypoints) < max_persons:
                pad = torch.zeros((max_persons - len(keypoints), 34))
                keypoints = torch.cat([keypoints, pad], dim=0)
            else:
                keypoints = keypoints[:max_persons]

            # Placeholder activity label (bạn có thể cải tiến sau)
            activity_label = 0 if len(item.get('keypoints', [])) == 0 else 1

            self.samples.append({
                'csi': csi,
                'keypoints': keypoints.flatten(),
                'activity': torch.tensor(activity_label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CNNLSTMModel(nn.Module):
    def __init__(self, num_subcarriers=192, num_persons=3, num_joints=17, num_classes=5):
        super().__init__()
        self.num_output_kp = num_persons * num_joints * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.25)

        self.fc_keypoints = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_output_kp)
        )

        self.fc_activity = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)                    # (B, 1, C)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)                # (B, T, 256)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]                # Last timestep

        keypoints = self.fc_keypoints(x)
        activity = self.fc_activity(x)
        return keypoints, activity


def train_model(npz_path: str, epochs=60, batch_size=16, lr=0.0008):
    # === GPU DETECTION & SETUP ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU được sử dụng: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Không tìm thấy GPU, đang dùng CPU (chậm)")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # === DataLoader ===
    dataset = CSIPoseDataset(npz_path)
    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # === Model & Training ===
    model = CNNLSTMModel().to(device)
    criterion_kp = nn.MSELoss()
    criterion_act = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Mixed Precision (tiết kiệm VRAM)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_loss = float('inf')

    print(f"Bắt đầu training với batch_size = {batch_size} trên {device}\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Chuyển data sang GPU
            csi = batch['csi'].to(device, non_blocking=True)
            kp_target = batch['keypoints'].to(device, non_blocking=True)
            act_target = batch['activity'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                kp_pred, act_pred = model(csi)
                loss_kp = criterion_kp(kp_pred, kp_target)
                loss_act = criterion_act(act_pred, act_target)
                loss = loss_kp * 0.7 + loss_act * 0.3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.6f} | Device: {device}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, "best_csi_pose_model.pth")
            print(f"   → Saved best model!")

    # Export ONNX
    model.eval()
    dummy_input = torch.randn(1, 192).to(device)
    torch.onnx.export(model, dummy_input, "csi_pose_model.onnx",
                      export_params=True, opset_version=13,
                      input_names=['csi_input'],
                      output_names=['keypoints', 'activity'],
                      dynamic_axes={'csi_input': {0: 'batch_size'}})

    print("\n🎉 TRAINING HOÀN THÀNH!")
    print(f"Best model: best_csi_pose_model.pth")
    print(f"ONNX model: csi_pose_model.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to aligned .npz file")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16, help="Giảm xuống 8 hoặc 12 nếu OOM")
    parser.add_argument("--lr", type=float, default=0.0008)
    args = parser.parse_args()

    train_model(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
