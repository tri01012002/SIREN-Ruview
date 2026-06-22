import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from train_csi import CSIPoseDataset, ImprovedCSIModel  # Import từ file train của bạn

def evaluate_model(npz_path, model_path="best_csi_pose_model.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    # Load dataset
    dataset = CSIPoseDataset(npz_path, max_persons=3, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Load model
    model = ImprovedCSIModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    criterion_kp = nn.MSELoss()
    all_kp_pred, all_kp_true = [], []
    all_act_pred, all_act_true = [], []
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            csi = batch['csi'].to(device)
            kp_target = batch['keypoints'].to(device)
            act_target = batch['activity'].to(device)
            
            with torch.amp.autocast("cuda"):
                kp_pred, act_pred = model(csi)
            
            loss = criterion_kp(kp_pred, kp_target) * 0.65 + nn.CrossEntropyLoss()(act_pred, act_target) * 0.35
            total_loss += loss.item()
            
            all_kp_pred.append(kp_pred.cpu())
            all_kp_true.append(kp_target.cpu())
            all_act_pred.append(act_pred.argmax(dim=1).cpu())
            all_act_true.append(act_target.cpu())
    
    # Tính metrics
    kp_mse = criterion_kp(torch.cat(all_kp_pred), torch.cat(all_kp_true)).item()
    act_acc = accuracy_score(torch.cat(all_act_true), torch.cat(all_act_pred))
    avg_loss = total_loss / len(loader)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Keypoints MSE: {kp_mse:.6f}")
    print(f"Activity Accuracy: {act_acc*100:.2f}%")
    
    # ==================== VẼ BIỂU ĐỒ ====================
    plt.figure(figsize=(15, 5))
    
    # Loss (chỉ có tổng loss)
    plt.subplot(1, 3, 1)
    plt.bar(['Total Loss', 'KP MSE'], [avg_loss, kp_mse])
    plt.title('Loss Metrics')
    plt.ylabel('Value')
    
    # Activity Accuracy
    plt.subplot(1, 3, 2)
    plt.bar(['Activity Accuracy'], [act_acc*100])
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Confusion Matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(torch.cat(all_act_true), torch.cat(all_act_pred))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Activity Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return kp_mse, act_acc


if __name__ == "__main__":
    # Thay đường dẫn data 1 người của bạn
    npz_path = r"D:\SIREN-Ruview\data\ground-truth\1person\train_data_20260615_150327.npz"
    model_path = "best_csi_pose_model.pth"
    
    evaluate_model(npz_path, model_path)