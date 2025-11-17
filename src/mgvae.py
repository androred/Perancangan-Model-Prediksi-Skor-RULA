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
from sklearn.model_selection import train_test_split
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
model_save_path = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\model\model_last3blocks_mgvae.pt'

# === TRANSFORMASI GAMBAR ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === BASE TOGGLES ===
USE_WEIGHTED_SAMPLER = False
USE_FOCAL = False
MIXUP_ALPHA = 0.0

# >>> MGVAE: Hyperparams
USE_MGVAE = True
MG_Z_DIM = 64
MG_BETA_KL = 0.1
MG_SYN_RATE = 1.0
MG_PRETRAIN_EPOCHS = 1
MG_EWC_LAMBDA = 1e-3
MG_VAE_LR = 1e-4
MG_COND_ONEHOT = True

# === MODEL DENGAN LAST 3 BLOCKS ===
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

    def count_trainable_parameters(self):
        """Hitung parameter yang bisa di-training"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# === Focal Loss ===
def focal_loss(logits, target, alpha=None, gamma=1.5, reduction='mean'):
    ce = F.cross_entropy(logits, target, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    fl = ((1 - pt) ** gamma) * ce
    if reduction == 'mean': return fl.mean()
    if reduction == 'sum':  return fl.sum()
    return fl

# === MGVAE modules ===
class MGEncoder(nn.Module):
    def __init__(self, fusion_dim=512+66, z_dim=64, num_classes=7, cond=True):
        super().__init__()
        in_dim = fusion_dim + (num_classes if cond else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)
        self.cond = cond
        self.num_classes = num_classes

    def forward(self, fusion, y=None):
        if self.cond:
            y_oh = F.one_hot(y, num_classes=self.num_classes).float()
            x = torch.cat([fusion, y_oh], dim=1)
        else:
            x = fusion
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class MGDecoder(nn.Module):
    def __init__(self, fusion_dim=512+66, z_dim=64, num_classes=7, cond=True):
        super().__init__()
        in_dim = z_dim + (num_classes if cond else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fusion_dim)
        )
        self.cond = cond
        self.num_classes = num_classes

    def forward(self, z, y=None):
        if self.cond:
            y_oh = F.one_hot(y, num_classes=self.num_classes).float()
            x = torch.cat([z, y_oh], dim=1)
        else:
            x = z
        return self.net(x)

def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss(recon, target, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

# === TRAIN ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Training on: {device} | FINETUNE_MODE=last_3blocks | MGVAE={USE_MGVAE}")

    dataset = DualInputRULADataset(csv_file=csv_path, image_dir=image_dir, transform=image_transform)

    # === STRATIFIED SPLIT 80-10-10 (SAMA SEPERTI BASELINE) ===
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

    print(f"📊 Dataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Class weights
    train_labels = [all_labels[i] for i in train_indices]
    counts = np.bincount(train_labels, minlength=7).astype(np.float32)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Mayoritas/minoritas
    majority_k = 2
    majority_classes = np.argsort(counts)[-majority_k:]
    majority_set = set(majority_classes.tolist())

    # Dataloader
    if USE_WEIGHTED_SAMPLER:
        inv = 1.0 / np.maximum(counts, 1.0)
        sample_w = inv[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double, device='cpu'),
            num_samples=len(train_labels), replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=2)

    # Initialize Model dengan LAST 3 BLOCKS
    model = DualInputRULAModel().to(device)
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_trainable_parameters()
    print(f"📦 Params total: {total_params:,} | trainable: {trainable_params:,}")

    # Loss, Optimizer (hanya untuk parameter yang requires_grad)
    if USE_FOCAL:
        criterion = lambda logits, target: focal_loss(logits, target, alpha=class_weights, gamma=1.5, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimize hanya parameter yang requires_grad
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Z-score keypoints
    xy_cols = [f'{a}{i}' for i in range(33) for a in ['x','y']]
    def get_kp_matrix(ds, idxs):
        kp = []
        for i in idxs:
            row = ds.df.iloc[i]
            vals = pd.to_numeric(row.loc[xy_cols], errors='coerce').values.astype('float32')
            kp.append(vals)
        return np.stack(kp, axis=0)

    kp_train = get_kp_matrix(dataset, train_indices)
    kp_mean = kp_train.mean(axis=0).astype('float32')
    kp_std  = kp_train.std(axis=0).astype('float32')
    kp_std[kp_std < 1e-6] = 1e-6
    kp_mean_t = torch.tensor(kp_mean, device=device)
    kp_std_t  = torch.tensor(kp_std,  device=device)

    # === MGVAE init ===
    fusion_dim = 512 + 66
    if USE_MGVAE:
        Enc = MGEncoder(fusion_dim=fusion_dim, z_dim=MG_Z_DIM, num_classes=7, cond=MG_COND_ONEHOT).to(device)
        Dec = MGDecoder(fusion_dim=fusion_dim, z_dim=MG_Z_DIM, num_classes=7, cond=MG_COND_ONEHOT).to(device)
        opt_VAE = optim.Adam(list(Enc.parameters()) + list(Dec.parameters()), lr=MG_VAE_LR)

        # pretrain di mayoritas
        maj_idx = [i for i in train_indices if int(all_labels[i]) in majority_set]
        maj_set = Subset(dataset, maj_idx)
        maj_loader = DataLoader(maj_set, batch_size=64, shuffle=True, num_workers=2)

        model.eval()
        for ep in range(MG_PRETRAIN_EPOCHS):
            pt_loss = 0.0; steps = 0
            for img, key, lab in maj_loader:
                with torch.no_grad():
                    img = img.to(device); key = key.to(device)
                    key = (key - kp_mean_t) / kp_std_t
                    feat = model.resnet(img)
                    fusion = torch.cat([feat, key], dim=1)
                    lab = lab.long().to(device)
                mu, logvar = Enc(fusion, lab)
                z = reparameterize(mu, logvar)
                recon = Dec(z, lab)
                loss_vae, _, _ = vae_loss(recon, fusion, mu, logvar, beta=MG_BETA_KL)
                opt_VAE.zero_grad()
                loss_vae.backward()
                opt_VAE.step()
                pt_loss += loss_vae.item(); steps += 1
            print(f"[MGVAE pretrain] epoch {ep+1}/{MG_PRETRAIN_EPOCHS} loss={pt_loss/max(1,steps):.4f}")

        Enc_anchor = {k: v.detach().clone() for k, v in Enc.state_dict().items()}
        Dec_anchor = {k: v.detach().clone() for k, v in Dec.state_dict().items()}

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

            # Fusion dari backbone (dengan last 3 blocks aktif)
            feat = model.resnet(img)
            fusion = torch.cat([feat, key], dim=1)

            # CE pada real
            logits_real = model.fc(fusion)
            loss_real = criterion(logits_real, label)

            loss_mgvae = torch.tensor(0.0, device=device)

            if USE_MGVAE:
                with torch.no_grad():
                    minority_mask = torch.tensor([(int(l.item()) not in majority_set) for l in label],
                                                 device=device, dtype=torch.bool)
                if minority_mask.any():
                    fusion_min = fusion[minority_mask]
                    y_min = label[minority_mask]
                    M = fusion_min.size(0)
                    S = max(1, int(MG_SYN_RATE * M))

                    # finetune VAE pada minority
                    fusion_min_det = fusion_min.detach()
                    mu, logvar = Enc(fusion_min_det, y_min)
                    z = reparameterize(mu, logvar)
                    recon = Dec(z, y_min)
                    loss_vae, _, _ = vae_loss(recon, fusion_min_det, mu, logvar, beta=MG_BETA_KL)

                    # EWC-like ke anchor mayoritas
                    ewc = 0.0
                    for n, p in Enc.named_parameters():
                        ewc += (p - Enc_anchor[n]).pow(2).mean()
                    for n, p in Dec.named_parameters():
                        ewc += (p - Dec_anchor[n]).pow(2).mean()
                    loss_vae_total = loss_vae + MG_EWC_LAMBDA * ewc

                    opt_VAE.zero_grad()
                    loss_vae_total.backward()
                    opt_VAE.step()

                    # generate synthetic fusion untuk minority
                    with torch.no_grad():
                        z_syn = torch.randn(S, MG_Z_DIM, device=device)
                        sel = torch.randint(0, M, (S,), device=device)
                        y_c = y_min[sel]
                        fusion_syn = Dec(z_syn, y_c)

                    # latih classifier pada synthetic
                    logits_syn = model.fc(fusion_syn.detach())
                    loss_syn = criterion(logits_syn, y_c)
                    loss_mgvae = loss_syn

            # update classifier (last 3 blocks + fc)
            loss_total = loss_real + loss_mgvae
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()

        # === VALIDASI ===
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

    # === TEST ===
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
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (RULA 1–7)")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls, digits=2))

    # Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title("Loss per Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend(); plt.title("Validation Accuracy per Epoch")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    train()
