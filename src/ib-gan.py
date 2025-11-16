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
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from dataset_dual_input_FULL import DualInputRULADataset

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
model_save_path = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\model\model_last3blocks_ibgan.pt'

# === TRANSFORMASI GAMBAR ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === TOGGLES SARAN ===
USE_WEIGHTED_SAMPLER = False
USE_FOCAL = False
FOCAL_GAMMA = 1.5
MIXUP_ALPHA = 0.0

# >>> IBGAN: Toggles & Hyperparams
USE_IBGAN = True
IB_WARMUP_EPOCHS = 3
IB_Z_DIM = 64
IB_SYN_RATE = 0.25
IB_D_STEPS = 1
IB_G_STEPS = 1
IB_LAMBDA_AUX = 1.0
IB_LAMBDA_RESID = 1e-2
IB_LAMBDA_SYN = 0.25
IB_FM_COEF = 0.5

# === DEFINISI MODEL DENGAN LAST 3 BLOCKS ===
class DualInputRULAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # === LAST 3 BLOCKS FINE-TUNING ===
        # FREEZE: layer1 (block1) + conv1, bn1, dll
        # UNFREEZE: layer2, layer3, layer4 (last 3 blocks)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # UNFREEZE LAST 3 BLOCKS
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
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
        x = self.resnet(image)
        x = torch.cat([x, keypoints], dim=1)
        return self.fc(x)

    def head_hidden(self, fusion):
        h = self.fc[0](fusion)
        h = self.fc[1](h)
        return h

    def count_trainable_parameters(self):
        """Hitung parameter yang bisa di-training"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# === Helper: Focal Loss ===
def focal_loss(logits, target, alpha=None, gamma=1.5, reduction='mean'):
    ce = F.cross_entropy(logits, target, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    fl = ((1 - pt) ** gamma) * ce
    if reduction == 'mean':
        return fl.mean()
    elif reduction == 'sum':
        return fl.sum()
    return fl

# === IBGAN Components ===
class IBGenerator(nn.Module):
    def __init__(self, fusion_dim=512+66, z_dim=64, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, fusion_dim),
        )
    def forward(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        return self.fc(inp)

class IBDiscriminator(nn.Module):
    def __init__(self, fusion_dim=512+66, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, fusion, y_onehot):
        inp = torch.cat([fusion, y_onehot], dim=1)
        return self.fc(inp)

# === TRAINING LOOP ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Training on: {device} | FINETUNE_MODE=last_3blocks")

    dataset = DualInputRULADataset(csv_file=csv_path, image_dir=image_dir, transform=image_transform)

    # Stratified Split
    labels_all = pd.Series(dataset.labels)
    def stratified_indices(y, train_ratio=0.8, val_ratio=0.1, seed=42):
        rng = np.random.RandomState(seed)
        train_idx, val_idx, test_idx = [], [], []
        for c in sorted(np.unique(y.values)):
            idx = np.where(y.values == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_tr = int(train_ratio * n)
            n_va = int(val_ratio * n)
            train_idx += idx[:n_tr].tolist()
            val_idx   += idx[n_tr:n_tr+n_va].tolist()
            test_idx  += idx[n_tr+n_va:].tolist()
        rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
        return train_idx, val_idx, test_idx

    train_ids, val_ids, test_ids = stratified_indices(labels_all, 0.8, 0.1, seed)

    train_set = Subset(dataset, train_ids)
    val_set   = Subset(dataset, val_ids)
    test_set  = Subset(dataset, test_ids)

    # Class Weights
    train_labels = labels_all.iloc[train_ids].values
    counts = np.bincount(train_labels, minlength=7).astype(np.float32)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    focus_minority_set = set([0, 1, 2, 3])

    # DataLoader
    if USE_WEIGHTED_SAMPLER:
        inv_freq = 1.0 / np.maximum(counts, 1.0)
        sample_weights = inv_freq[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double, device='cpu'),
            num_samples=len(train_labels),
            replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=2)

    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=2)

    # Initialize Model
    model = DualInputRULAModel().to(device)
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_trainable_parameters()
    print(f"📦 Params total: {total_params:,} | trainable: {trainable_params:,}")

    # Loss, Optimizer, Scheduler
    if USE_FOCAL:
        criterion = lambda logits, target: focal_loss(logits, target, alpha=class_weights, gamma=FOCAL_GAMMA, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimize hanya parameter yang requires_grad
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Z-score Keypoints
    xy_cols = [f'{a}{i}' for i in range(33) for a in ['x','y']]
    def get_kp_matrix(ds, idxs):
        kp = []
        for i in idxs:
            row = ds.df.iloc[i]
            vals = pd.to_numeric(row.loc[xy_cols], errors='coerce').values.astype('float32')
            kp.append(vals)
        return np.stack(kp, axis=0)

    kp_train = get_kp_matrix(dataset, train_ids)
    kp_mean = kp_train.mean(axis=0).astype('float32')
    kp_std  = kp_train.std(axis=0).astype('float32')
    kp_std[kp_std < 1e-6] = 1e-6
    kp_mean_t = torch.tensor(kp_mean, device=device)
    kp_std_t  = torch.tensor(kp_std,  device=device)

    # IBGAN Initialization
    fusion_dim = 512 + 66
    if USE_IBGAN:
        G = IBGenerator(fusion_dim=fusion_dim, z_dim=IB_Z_DIM, num_classes=7).to(device)
        D = IBDiscriminator(fusion_dim=fusion_dim, num_classes=7).to(device)
        opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses, val_losses, val_accs = [], [], []
    EPOCHS = 30
    patience = 5
    stale = 0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        train_loss = 0

        for img, key, label in train_loader:
            img, key = img.to(device), key.to(device)
            key = (key - kp_mean_t) / kp_std_t
            label = label.long().to(device)

            # Forward pass
            feat = model.resnet(img)
            fusion = torch.cat([feat, key], dim=1)
            
            if MIXUP_ALPHA > 0.0:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx = torch.randperm(fusion.size(0), device=fusion.device)
                fusion_m = lam * fusion + (1 - lam) * fusion[idx]
                y_a, y_b = label, label[idx]
                logits_real = model.fc(fusion_m)
                loss_real = lam * criterion(logits_real, y_a) + (1 - lam) * criterion(logits_real, y_b)
            else:
                logits_real = model.fc(fusion)
                loss_real = criterion(logits_real, label)

            loss_ibgan = torch.tensor(0.0, device=device)

            # IBGAN Training
            use_gan_now = USE_IBGAN and (epoch >= IB_WARMUP_EPOCHS)

            if use_gan_now:
                with torch.no_grad():
                    minority_mask = torch.tensor(
                        [int(l.item()) in focus_minority_set for l in label],
                        device=device, dtype=torch.bool
                    )
                if minority_mask.any():
                    fusion_min = fusion[minority_mask]
                    y_min = label[minority_mask]
                    M = fusion_min.size(0)
                    S = max(1, int(IB_SYN_RATE * M))

                    # Train D
                    for _ in range(IB_D_STEPS):
                        z = torch.randn(S, IB_Z_DIM, device=device)
                        sel = torch.randint(0, M, (S,), device=device)
                        y_c = y_min[sel]
                        y_c_oh = F.one_hot(y_c, num_classes=7).float()

                        with torch.no_grad():
                            base = fusion_min[sel]

                        resid = G(z, y_c_oh)
                        resid = 0.05 * torch.tanh(resid)
                        fusion_syn = base + resid

                        y_oh_real = F.one_hot(y_min, num_classes=7).float()
                        real_logits = D(fusion_min.detach(), y_oh_real)
                        fake_logits = D(fusion_syn.detach(), y_c_oh)

                        real_targets = torch.full_like(real_logits, 0.9)
                        fake_targets = torch.full_like(fake_logits, 0.1)

                        loss_D = bce(real_logits, real_targets) + bce(fake_logits, fake_targets)
                        opt_D.zero_grad()
                        loss_D.backward()
                        opt_D.step()

                    # Train G
                    for _ in range(IB_G_STEPS):
                        z = torch.randn(S, IB_Z_DIM, device=device)
                        sel = torch.randint(0, M, (S,), device=device)
                        y_c = y_min[sel]
                        y_c_oh = F.one_hot(y_c, num_classes=7).float()

                        with torch.no_grad():
                            base = fusion_min[sel]
                        resid = G(z, y_c_oh)
                        resid = 0.05 * torch.tanh(resid)
                        fusion_syn = base + resid

                        logits_fake = D(fusion_syn, y_c_oh)
                        lbl_real = torch.full_like(logits_fake, 0.9)
                        loss_G_adv = bce(logits_fake, lbl_real)

                        logits_aux = model.fc(fusion_syn)
                        loss_G_aux = criterion(logits_aux, y_c)

                        loss_resid = (resid**2).mean()

                        with torch.no_grad():
                            h_real = model.head_hidden(fusion_min).mean(0)
                        h_syn = model.head_hidden(fusion_syn).mean(0)
                        loss_fm = (h_syn - h_real).pow(2).mean()

                        loss_G = loss_G_adv + IB_LAMBDA_AUX*loss_G_aux + IB_LAMBDA_RESID*loss_resid + IB_FM_COEF*loss_fm
                        opt_G.zero_grad()
                        loss_G.backward()
                        opt_G.step()

                    # Train classifier on synthetic
                    with torch.no_grad():
                        z = torch.randn(S, IB_Z_DIM, device=device)
                        sel = torch.randint(0, M, (S,), device=device)
                        y_c = y_min[sel]
                        y_c_oh = F.one_hot(y_c, num_classes=7).float()
                        base = fusion_min[sel]
                        resid = G(z, y_c_oh)
                        resid = 0.05 * torch.tanh(resid)
                        fusion_syn = base + resid

                    logits_syn = model.fc(fusion_syn)
                    loss_syn = criterion(logits_syn, y_c)
                    loss_ibgan = IB_LAMBDA_SYN * loss_syn

            # Update backbone
            loss_total = loss_real + loss_ibgan
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()

        # Validation
        model.eval()
        val_loss, correct, total_val = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for img, key, label in val_loader:
                img, key = img.to(device), key.to(device)
                key = (key - kp_mean_t) / kp_std_t
                label = label.long().to(device)

                feat = model.resnet(img)
                fusion = torch.cat([feat, key], dim=1)
                logits = model.fc(fusion)
                loss = criterion(logits, label)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total_val += label.size(0)

                y_true.extend(label.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total_val)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        durasi = time.time() - start
        improved = avg_val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = avg_val_loss
            stale = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Epoch {epoch+1:02d}/{EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | ValAcc {val_acc:.2f}% | ⏱️ {durasi:.2f}s (saved)")
        else:
            stale += 1
            print(f"📉 Epoch {epoch+1:02d}/{EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | ValAcc {val_acc:.2f}% | ⏱️ {durasi:.2f}s")
            if stale >= patience:
                print(f"⛔ Early stopping at epoch {epoch+1}. Restoring best weights...")
                break

    # Test Evaluation
    print("\n[TEST] Evaluasi model terbaik:")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    correct, total_test = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, key, label in test_loader:
            img, key = img.to(device), key.to(device)
            key = (key - kp_mean_t) / kp_std_t
            label = label.long().to(device)

            feat = model.resnet(img)
            fusion = torch.cat([feat, key], dim=1)
            logits = model.fc(fusion)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == label).sum().item()
            total_test += label.size(0)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_acc = 100.0 * correct / max(1, total_test)
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")

    # Confusion Matrix
    y_true_cls = np.array(y_true) + 1
    y_pred_cls = np.array(y_pred) + 1
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[1,2,3,4,5,6,7])
    sns.heatmap(pd.DataFrame(cm, index=[1,2,3,4,5,6,7], columns=[1,2,3,4,5,6,7]),
                annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (RULA 1–7)")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, digits=2))

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title("Loss per Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend(); plt.title("Validation Accuracy per Epoch")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
