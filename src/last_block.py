import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from dataset_dual_input import DualInputRULADataset

# === RANDOM SEED ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === PATH ===
csv_path = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\keypoints_rula_FULL_clean.csv'
image_dir = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\image'
model_save_path = r'D:\Kuliah\Sem 7\TA1\CODE TA\model\model_last_block.pt'

# === TRANSFORMASI GAMBAR ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === MODEL LAST BLOCK ===
class DualInputRULALastBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # FREEZE SEMUA LAYER KECUALI LAYER4
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # UNFREEZE LAST BLOCK
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(512 + 66, 256),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),       
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 7)
        )

    def forward(self, image, keypoints):
        image_features = self.resnet(image)
        combined = torch.cat([image_features, keypoints], dim=1)
        return self.fc(combined)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# === Helper: normalisasi & cek label ===
def normalize_label(label_tensor: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    lab = label_tensor.long()
    min_l = int(lab.min().item())
    max_l = int(lab.max().item())

    if 0 <= min_l and max_l < num_classes:
        pass
    elif 1 <= min_l and max_l <= num_classes:
        lab = lab - 1
    else:
        uniques = torch.unique(lab).tolist()
        raise ValueError(f"Label di luar rentang (ditemukan {uniques}); "
                         f"harus 0..{num_classes-1} atau 1..{num_classes}.")
    return lab

# === TRAINING LOOP - LAST BLOCK ===
def train_last_block():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔧 Training on: {device} | FINETUNE_MODE=last_block")
    
    dataset = DualInputRULADataset(csv_file=csv_path, image_dir=image_dir, transform=image_transform)
    
    # === STRATIFIED SPLIT 80-10-10 ===
    # Dapatkan semua labels dari dataset
    all_labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        all_labels.append(label.item() if torch.is_tensor(label) else label)
    
    all_labels = np.array(all_labels)
    
    # First split: train vs temp (80% vs 20%)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=seed,
        stratify=all_labels
    )
    
    # Second split: val vs test (dari 20% temp, bagi menjadi 10% val dan 10% test)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=[all_labels[i] for i in temp_indices]
    )
    
    # Create subsets menggunakan indices yang sudah di-stratify
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

    model = DualInputRULALastBlock().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_trainable_parameters()
    print(f"📦 Params total: {total_params:,} | trainable: {trainable_params:,}")
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params_list, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    train_losses, val_losses, val_accs = [], [], []
    EPOCHS = 30

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        train_loss = 0

        for img, key, label in train_loader:
            img, key = img.to(device), key.to(device)
            label = normalize_label(label, num_classes=7).to(device)

            optimizer.zero_grad()
            logits = model(img, key)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total_val = 0, 0, 0
        with torch.no_grad():
            for img, key, label in val_loader:
                img, key = img.to(device), key.to(device)
                label = normalize_label(label, num_classes=7).to(device)

                logits = model(img, key)
                loss = criterion(logits, label)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total_val += label.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total_val
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            saved_indicator = "(saved)"
        else:
            patience_counter += 1
            saved_indicator = ""

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        durasi = time.time() - start
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | ValAcc {val_acc:.2f}% | ⏱️ {durasi:.2f}s {saved_indicator}")
        
        if patience_counter >= patience:
            print(f"⛔ Early stopping at epoch {epoch+1}. Restoring best weights...")
            break

    print("\n[TEST] Evaluasi model terbaik:")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    correct, total_test = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, key, label in test_loader:
            img, key = img.to(device), key.to(device)
            label = normalize_label(label, num_classes=7).to(device)

            logits = model(img, key)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == label).sum().item()
            total_test += label.size(0)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_acc = 100.0 * correct / total_test
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")

    y_true_cls = np.array(y_true) + 1
    y_pred_cls = np.array(y_pred) + 1
    
    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, 
                               target_names=['RULA-1', 'RULA-2', 'RULA-3', 'RULA-4', 'RULA-5', 'RULA-6', 'RULA-7'],
                               digits=2))

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[1,2,3,4,5,6,7])
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(cm, index=[1,2,3,4,5,6,7], columns=[1,2,3,4,5,6,7]),
                annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted RULA Score")
    plt.ylabel("Actual RULA Score")
    plt.title(f"Confusion Matrix - Last Block\nTest Accuracy: {test_acc:.2f}%")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LAST BLOCK - Loss per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LAST BLOCK - Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return test_acc

if __name__ == "__main__":
    test_accuracy = train_last_block()
    
    print("\n" + "="*70)
    print(f"LAST BLOCK TRAINING COMPLETED!")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("="*70)
