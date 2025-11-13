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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score  # + QWK
from dataset_dual_input import DualInputRULADataset

# =========================
# CONFIG (edit di sini)
# =========================
seed = 42

# CSV terpisah (train: hasil oversampling, val & test: distribusi asli)
train_csv_path = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\balanced_splits_generatif\train_over.csv'
val_csv_path   = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\balanced_splits_generatif\val_real.csv'
test_csv_path  = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\balanced_splits_generatif\test_real.csv'

# Direktori gambar tunggal
image_dir = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\image'

# Path penyimpanan model
model_save_path = r'D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\Graph and Data\oversampling trad\small\model_3block_resnet_small.pt'

# Hiperparameter
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 2e-5
DROP = 0.5

# Prior-correction hyperparam
TAU = 1.5     # skala koreksi prior (boleh coba 1.0–3.0)
EPS = 1e-12   # untuk jaga-jaga log(0)

# =========================
# SEED
# =========================
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# UTIL: AUDIT
# =========================
def audit_csv(csv_path, label_col='rula_score_final'):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        print(f"[AUDIT] {os.path.basename(csv_path)} | Kolom '{label_col}' tidak ada. Kolom tersedia: {list(df.columns)}")
        return df
    s = pd.to_numeric(df[label_col], errors='coerce')
    uniques = sorted(s.dropna().unique().tolist())
    bad = df[~s.isin(list(range(0,7)) + list(range(1,8)))]
    print(f"[AUDIT] {os.path.basename(csv_path)} | label_col='{label_col}' | total={len(df)} | unik={uniques[:20]}")
    if len(bad):
        print("  -> Label tidak valid di baris (contoh 20):", bad.index.tolist()[:20])
    return df

def check_label_range(name, loader):
    mins, maxs = [], []
    with torch.no_grad():
        for i, (_, _, lbl) in enumerate(loader):
            mn, mx = int(lbl.min().item()), int(lbl.max().item())
            mins.append(mn); maxs.append(mx)
            if i >= 2:
                break
    if mins and maxs:
        print(f"[CHECK] {name}: min={min(mins)}, max={max(maxs)} (harus 0..6)")
    else:
        print(f"[CHECK] {name}: (loader kosong?)")

def get_class_prior(dataset, num_classes=7):
    """Ambil prior kelas dari dataset.labels (0..6)."""
    labels = np.asarray(dataset.labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    prior = counts / max(1.0, counts.sum())
    return prior, counts

# =========================
# TRANSFORM
# =========================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# MODEL
# =========================
class DualInputRULAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # keluaran fitur 512
        self.fc = nn.Sequential(
            nn.Linear(512 + 66, 256),
            nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROP),
            nn.Linear(128, 7)  # 7 kelas (RULA 1–7) -> logits
        )
    def forward(self, image, keypoints):
        x_img = self.resnet(image)
        x = torch.cat([x_img, keypoints], dim=1)
        return self.fc(x)

# =========================
# TRAINING
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Training on: {device}")

    # Audit CSV
    _ = audit_csv(train_csv_path, 'rula_score_final')
    _ = audit_csv(val_csv_path,   'rula_score_final')
    _ = audit_csv(test_csv_path,  'rula_score_final')

    # Dataset
    train_set = DualInputRULADataset(csv_file=train_csv_path, image_dir=image_dir, transform=image_transform)
    val_set   = DualInputRULADataset(csv_file=val_csv_path,   image_dir=image_dir, transform=image_transform)
    test_set  = DualInputRULADataset(csv_file=test_csv_path,  image_dir=image_dir, transform=image_transform)

    # Priors (train vs val/test)
    prior_train, counts_train = get_class_prior(train_set, num_classes=7)
    prior_val,   _            = get_class_prior(val_set,   num_classes=7)
    prior_test,  _            = get_class_prior(test_set,  num_classes=7)
    print(f"[PRIOR] train={counts_train.tolist()} -> {np.round(prior_train,4).tolist()}")
    # siapkan tensor log(prior) di device
    log_prior_train = torch.log(torch.tensor(prior_train + EPS, dtype=torch.float32, device=device))
    log_prior_val   = torch.log(torch.tensor(prior_val   + EPS, dtype=torch.float32, device=device))
    log_prior_test  = torch.log(torch.tensor(prior_test  + EPS, dtype=torch.float32, device=device))

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Cek label
    check_label_range("TRAIN", train_loader)
    check_label_range("VAL",   val_loader)
    check_label_range("TEST",  test_loader)

    # Model, Loss, Optimizer, Scheduler
    model = DualInputRULAModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # <- label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_kappa = -1e9  # simpan berdasar QWK
    train_losses, val_losses, val_accs, val_kappas = [], [], [], []

    for epoch in range(EPOCHS):
        t0 = time.time()

        # ----- Train -----
        model.train()
        running_train_loss = 0.0
        for img, key, label in train_loader:
            img, key, label = img.to(device), key.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img, key)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # ----- Val -----
        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for img, key, label in val_loader:
                img, key, label = img.to(device), key.to(device), label.to(device)

                logits = model(img, key)

                # Loss pakai logits asli (konsisten dgn training)
                val_ce = criterion(logits, label)
                running_val_loss += val_ce.item()

                # ===== Prior correction (VALIDASI) =====
                # logits_adj = logits + TAU*(log P_val - log P_train)
                logits_adj = logits + TAU * (log_prior_val - log_prior_train)

                preds = torch.argmax(logits_adj, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

                y_true.extend(label.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

        avg_train_loss = running_train_loss / max(1, len(train_loader))
        avg_val_loss = running_val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)
        val_kappa = cohen_kappa_score(np.array(y_true), np.array(y_pred), weights='quadratic')

        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        val_kappas.append(val_kappa)

        dur = time.time() - t0
        saved = ""
        if val_kappa > best_val_kappa + 1e-4:  # simpan berdasar QWK
            best_val_kappa = val_kappa
            torch.save(model.state_dict(), model_save_path)
            saved = "(saved)"
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} "
              f"| ValAcc {val_acc:.2f}% | ValQWK {val_kappa:.3f} | ⏱️ {dur:.2f}s {saved}")

    # =========================
    # TEST EVALUATION
    # =========================
    print("\n[TEST] Evaluasi model terbaik:")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, key, label in test_loader:
            img, key, label = img.to(device), key.to(device), label.to(device)
            logits = model(img, key)

            # ===== Prior correction (TEST) =====
            logits_adj = logits + TAU * (log_prior_test - log_prior_train)

            preds = torch.argmax(logits_adj, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

            y_true.extend(label.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    test_acc = 100.0 * correct / max(1, total)
    test_kappa = cohen_kappa_score(np.array(y_true), np.array(y_pred), weights='quadratic')
    print(f"🎯 Test Accuracy: {test_acc:.2f}% | QWK: {test_kappa:.3f}")

    # Tambahan: ±1 accuracy (ordinale)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    plusminus1 = np.mean(np.abs(y_true_np - y_pred_np) <= 1) * 100.0
    print(f"±1 Accuracy: {plusminus1:.2f}%")

    # Confusion Matrix (pakai label 1–7 saat tampil)
    y_true_cls = y_true_np + 1
    y_pred_cls = y_pred_np + 1
    labels_ = [1,2,3,4,5,6,7]

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels_)
    plt.figure(figsize=(6,5))
    sns.heatmap(pd.DataFrame(cm, index=labels_, columns=labels_), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (RULA 1–7) [prior-corrected]")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    target_names = [f"RULA-{i}" for i in labels_]
    print(classification_report(y_true_cls, y_pred_cls, labels=labels_, target_names=target_names, digits=2))

    # Curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.legend(); plt.title("Loss per Epoch")

    plt.subplot(1,2,2)
    plt.plot(val_accs,   label='Val Accuracy')
    plt.plot(val_kappas, label='Val QWK')
    plt.legend(); plt.title("Validation Accuracy & QWK per Epoch")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
