import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import os
from pathlib import Path
# ====================== DATASET ======================
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
        
        print(f"✅ Loaded {good_count} good samples")
        
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


# ====================== HYBRID CNN + TRANSFORMER MODEL ======================
class HybridCSIModel(nn.Module):
    def __init__(self, num_persons=3, num_joints=17, num_classes=5, pretrained_path=r"D:\SIREN-Ruview\scripts\best_csi_pose_model.pth"):
        super().__init__()
        self.num_output_kp = num_persons * num_joints * 2
        self.d_model = 256
        
        # ==================== CNN Backbone (from 1 person's model) ====================
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        
        # ==================== Transformer Encoder ====================
        self.pos_encoder = nn.Parameter(torch.zeros(1, 21, self.d_model))  # ~168/8
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Projection từ CNN feature -> Transformer
        self.cnn_to_trans = nn.Linear(256, self.d_model)
        
        # Heads
        self.fc_keypoints = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_output_kp)
        )
        
        self.fc_activity = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Load pretrained from 1-person model 
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Load CNN layers (transfer learning)
            cnn_dict = {k: v for k, v in state_dict.items() if k.startswith('cnn.')}
            self.cnn.load_state_dict(cnn_dict, strict=False)
            print(f"✅ Loaded pretrained CNN weights from 1-person model")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
    
    def forward(self, x):
        # x: (B, 168)
        x = x.unsqueeze(1)                    # (B, 1, 168)
        cnn_feat = self.cnn(x)                # (B, 256, ~21)
        
        # Project to transformer dimension
        x = cnn_feat.transpose(1, 2)          # (B, ~21, 256)
        x = self.cnn_to_trans(x)              # (B, ~21, 256)
        x = x + self.pos_encoder[:, :x.size(1)]
        
        x = self.transformer(x)               # (B, ~21, 256)
        x = x.mean(dim=1)                     # Global Average Pooling
        
        keypoints = self.fc_keypoints(x)
        activity = self.fc_activity(x)
        return keypoints, activity


# ====================== TRAINING ======================
def train_hybrid_2person(npz_path, 
                        pretrained_path=None, 
                        num_epochs=50, 
                        batch_size=12,
                        save_name="best_hybrid_2person.pth"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device} | Data: {npz_path}")
    
    dataset = CSIPoseDataset(npz_path, max_persons=3, normalize=True)
    train_size = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HybridCSIModel(pretrained_path=pretrained_path).to(device)
    
    # Continue training
    start_epoch = 0
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ Continue training from epoch {start_epoch}")
        except:
            print("⚠️ Load checkpoint thất bại, train từ đầu.")
    
    criterion_kp = nn.MSELoss()
    criterion_act = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler("cuda", enabled=True)
    
    best_val_loss = float('inf')
    save_path = f"models/{save_name}"
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            csi = batch['csi'].to(device, non_blocking=True)
            kp_target = batch['keypoints'].to(device, non_blocking=True)
            act_target = batch['activity'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast("cuda", enabled=True):
                kp_pred, act_pred = model(csi)
                loss = criterion_kp(kp_pred, kp_target)*0.55 + criterion_act(act_pred, act_target)*0.45
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    loss = criterion_kp(kp_pred, kp_target) * 0.55 + criterion_act(act_pred, act_target) * 0.45
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
            print(f"   → Best model saved: {save_path}")
    
    print(f"\n✅ Training finished for {npz_path}!")
    print(f"Model saved at: {save_path}")
    return model

if __name__ == "__main__":
    # Train file 1
    train_hybrid_2person(
        npz_path=r"D:\SIREN-Ruview\data\ground-truth\2person\train_data_20260615_150935.npz",
        pretrained_path=r"D:\SIREN-Ruview\scripts\best_csi_pose_model.pth",  # Load từ model 1 người           
        num_epochs=50,
        save_name="best_hybrid_2person_v1.pth"
    )
    
    train_hybrid_2person(
        npz_path=r"D:\SIREN-Ruview\data\ground-truth\2person\train_data_20260615_153918.npz",
        pretrained_path=r"models/best_hybrid_2person_v1.pth",
        num_epochs=40,
        save_name="best_hybrid_2person_v2.pth"
    )
    
    train_hybrid_2person(
        npz_path=r"D:\SIREN-Ruview\data\ground-truth\2person\train_data_20260615_154912.npz",
        pretrained_path=r"models/best_hybrid_2person_v2.pth",
        num_epochs=40,
        save_name="best_hybrid_2person_v3.pth"
    )

    train_hybrid_2person(
        npz_path=r"D:\SIREN-Ruview\data\ground-truth\2person\train_data_20260615_154953.npz",
        pretrained_path=r"models/best_hybrid_2person_v3.pth",
        num_epochs=40,
        save_name="best_hybrid_2person_v4.pth"
    )
