import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast   # Import đúng

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# ====================== DATASET (giữ nguyên) ======================
class CSIPoseDataset(Dataset):
    def __init__(self, npz_path: str, max_persons=3, normalize=True):
        data = np.load(npz_path, allow_pickle=True)['samples']
        self.samples = []
        self.max_persons = max_persons
        self.normalize = normalize
        
        good_count = 0
        for item in data:
            csi_np = np.asarray(item.get('csi', []), dtype=np.float32)
            if csi_np.shape != (168,):
                continue
            
            csi = torch.tensor(csi_np, dtype=torch.float32)
            
            kp = item.get('keypoints', [])
            if len(kp) == 0:
                keypoints = torch.zeros((max_persons, 34), dtype=torch.float32)
            else:
                keypoints = torch.tensor(kp, dtype=torch.float32)
                if keypoints.ndim == 1:
                    keypoints = keypoints.unsqueeze(0)
                
                if keypoints.shape[1] != 34:
                    continue
                
                if keypoints.shape[0] < max_persons:
                    pad = torch.zeros((max_persons - keypoints.shape[0], 34), dtype=torch.float32)
                    keypoints = torch.cat([keypoints, pad], dim=0)
                keypoints = keypoints[:max_persons]
            
            activity = 1 if keypoints.abs().sum() > 1e-5 else 0
            
            self.samples.append({
                'csi': csi,
                'keypoints': keypoints.flatten(),
                'activity': torch.tensor(activity, dtype=torch.long)
            })
            good_count += 1
        
        print(f"✅ Loaded {good_count}/{len(data)} good samples | Subcarriers: 168")
        
        if normalize and good_count > 0:
            all_csi = torch.stack([s['csi'] for s in self.samples])
            self.csi_mean = all_csi.mean(dim=0)
            self.csi_std = all_csi.std(dim=0).clamp(min=1e-5)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        csi = sample['csi']
        if self.normalize:
            csi = (csi - self.csi_mean) / self.csi_std
        return {
            'csi': csi,
            'keypoints': sample['keypoints'],
            'activity': sample['activity']
        }


# ====================== MODEL (giữ nguyên) ======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(dropout)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x += self.shortcut(residual)
        x = nn.ReLU(inplace=True)(x)
        return x


class ImprovedCSIModel(nn.Module):
    def __init__(self, num_subcarriers=168, num_persons=3, num_joints=17, num_classes=5):
        super().__init__()
        self.num_output_kp = num_persons * num_joints * 2
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            ResidualBlock(64, 128, kernel_size=5),
            nn.MaxPool1d(2),
            
            ResidualBlock(128, 256, kernel_size=5),
            nn.MaxPool1d(2),
            
            ResidualBlock(256, 512, kernel_size=3),
            nn.AdaptiveAvgPool1d(8),
        )
        
        self.feature_size = 512 * 8
        
        self.fc_keypoints = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_output_kp)
        )
        
        self.fc_activity = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        
        keypoints = self.fc_keypoints(x)
        activity = self.fc_activity(x)
        return keypoints, activity


# ====================== TRAINING FUNCTION (ĐÃ SỬA AMP) ======================
def train_model(npz_path, num_epochs=50, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    dataset = CSIPoseDataset(npz_path, max_persons=3, normalize=True)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True, 
                             drop_last=True)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    model = ImprovedCSIModel().to(device)
    
    criterion_kp = nn.MSELoss()
    criterion_act = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # ====================== AMP ĐÚNG CHO PYTORCH 2.12 ======================
    scaler = GradScaler("cuda", enabled=True)          # <-- Sửa đúng cách
    
    best_val_loss = float('inf')
    save_path = "best_csi_pose_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            csi = batch['csi'].to(device, non_blocking=True)
            kp_target = batch['keypoints'].to(device, non_blocking=True)
            act_target = batch['activity'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=True):       # <-- Sửa đúng cách
                kp_pred, act_pred = model(csi)
                loss_kp = criterion_kp(kp_pred, kp_target)
                loss_act = criterion_act(act_pred, act_target)
                loss = loss_kp * 0.65 + loss_act * 0.35
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                csi = batch['csi'].to(device)
                kp_target = batch['keypoints'].to(device)
                act_target = batch['activity'].to(device)
                
                with autocast("cuda", enabled=True):
                    kp_pred, act_pred = model(csi)
                    loss = criterion_kp(kp_pred, kp_target) * 0.65 + criterion_act(act_pred, act_target) * 0.35
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss
            }, save_path)
            print(f"   → Best model saved!")
    
    print("\n✅ Training completed!")
    return model


if __name__ == "__main__":
    npz_path = r"D:\SIREN-Ruview\data\ground-truth\1person\train_data_20260615_150327.npz"
    train_model(npz_path, num_epochs=50, batch_size=4)