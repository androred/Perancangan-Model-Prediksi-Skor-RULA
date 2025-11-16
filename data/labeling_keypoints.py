import os
import sys
import csv
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ====== KONFIGURASI ======
IMG_DIR = r"D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\image"   # folder gambar input
OUT_CSV = r"D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\keypoints_rula_FULL_data.csv"
BATCH_SIZE = 500
SAVE_EVERY = 100            # simpan setiap N gambar diproses (di luar checkpoint batch)
USE_TQDM = True             # butuh tqdm untuk progress bar? set False jika ingin log sederhana
CONF_THRESHOLD = 0.5        # min visibility pose landmark agar dianggap valid
# ==========================


try:
    from tqdm import tqdm
except Exception:
    USE_TQDM = False

# MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("❌ mediapipe belum terpasang. Install dengan: pip install mediapipe==0.10.14")
    sys.exit(1)

mp_pose = mp.solutions.pose

# Indeks landmark MediaPipe Pose (33 titik)
# https://developers.google.com/mediapipe/solutions/vision/pose
LMS = mp.solutions.pose.PoseLandmark
# Shortcut indeks (kita gunakan sisi kiri & kanan untuk upper-limb)
L_SHOULDER, R_SHOULDER = LMS.LEFT_SHOULDER.value, LMS.RIGHT_SHOULDER.value
L_ELBOW, R_ELBOW       = LMS.LEFT_ELBOW.value, LMS.RIGHT_ELBOW.value
L_WRIST, R_WRIST       = LMS.LEFT_WRIST.value, LMS.RIGHT_WRIST.value
L_HIP, R_HIP           = LMS.LEFT_HIP.value, LMS.RIGHT_HIP.value

# Ekstensi gambar yang didukung
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------- Util sudut & vektor ----------
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    Sudut ABC (derajat) dengan titik B sebagai vertex.
    a,b,c: (x,y) dalam koordinat normalisasi [0..1].
    Return None jika tidak valid.
    """
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cosang = np.dot(ba, bc) / (na * nc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _is_valid(pt: Tuple[float,float,float], conf_th=CONF_THRESHOLD) -> bool:
    # pt = (x, y, visibility)
    if pt is None: return False
    x, y, vis = pt
    return (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (vis is not None) and (vis >= conf_th)


def _get_xy(row: List[Tuple[float,float,float]], idx: int) -> Optional[np.ndarray]:
    # ambil (x,y) jika visibility memadai
    if idx < 0 or idx >= len(row): return None
    x, y, v = row[idx]
    if (x is None) or (y is None) or (v is None): return None
    if v < CONF_THRESHOLD: return None
    return np.array([x, y], dtype=np.float32)


# ---------- Skor RULA (versi worksheet-inspired) ----------
def rula_upper_arm_score(upper_angle: float) -> int:
    """
    Upper arm flexion/abduction proxy.
    Threshold referensi dari berbagai implementasi populer (bukan tabel asli 1:1).
    """
    if upper_angle is None:
        return -1
    # sudut di bahu: elbow-shoulder-hip (semakin besar -> lengan lebih terangkat)
    if upper_angle < 20:   return 1
    if upper_angle < 45:   return 2
    if upper_angle < 90:   return 3
    if upper_angle < 120:  return 4
    if upper_angle < 160:  return 5
    return 6  # sangat terangkat


def rula_lower_arm_score(lower_angle: float) -> int:
    """
    Lower arm (siku) flexion.
    """
    if lower_angle is None:
        return -1
    # siku ideal 60-100 derajat
    if 60 <= lower_angle <= 100: return 1
    if 30 <= lower_angle < 60 or 100 < lower_angle <= 120: return 2
    return 3  # jauh dari rentang nyaman


def rula_wrist_score(wrist_angle: float) -> int:
    """
    Wrist (pergelangan) flexion proxy.
    Karena kita tidak punya rotasi pronation/supination yang akurat di 2D,
    gunakan proxy sederhana dari garis elbow-wrist vs sumbu horizontal lokal.
    """
    if wrist_angle is None:
        return -1
    # makin besar deviasi -> skor lebih tinggi
    if wrist_angle < 15:  return 1
    if wrist_angle < 30:  return 2
    if wrist_angle < 45:  return 3
    return 4


def rula_groupA_score(ua: int, la: int, w: int, wrist_twist_adj: int = 0) -> int:
    """
    Gabungkan skor Table A secara sederhana (worksheet-inspired).
    RULA asli menggunakan lookup table;di sini dikonservasi dengan:
        base = ua + la - 1
        base = base + w - 1
        + wrist_twist_adj (0 atau 1)
        clamp 1..7
    """
    if ua < 1 or la < 1 or w < 1:
        return -1
    base = (ua + la - 1) + (w - 1) + wrist_twist_adj
    base = max(1, min(7, base))
    return base


def compute_side_scores(lms_row: List[Tuple[float,float,float]], side: str) -> int:
    """
    Hitung skor RULA Table A untuk satu sisi (left/right).
    """
    if side == "left":
        s_idx, e_idx, w_idx, h_idx = L_SHOULDER, L_ELBOW, L_WRIST, L_HIP
    else:
        s_idx, e_idx, w_idx, h_idx = R_SHOULDER, R_ELBOW, R_WRIST, R_HIP

    s = _get_xy(lms_row, s_idx)
    e = _get_xy(lms_row, e_idx)
    w = _get_xy(lms_row, w_idx)
    h = _get_xy(lms_row, h_idx)

    # Upper arm angle: elbow - shoulder - hip
    upper = _angle(e, s, h)
    # Lower arm angle: wrist - elbow - shoulder
    lower = _angle(w, e, s)
    # Wrist angle: approx sudut elbow - wrist - (titik sedikit ke kanan dari wrist)
    wrist = None
    if (w is not None) and (e is not None):
        pseudo = np.array([w[0] + 0.01, w[1]], dtype=np.float32)
        wrist = _angle(e, w, pseudo)

    ua = rula_upper_arm_score(upper) if upper is not None else -1
    la = rula_lower_arm_score(lower) if lower is not None else -1
    ws = rula_wrist_score(wrist) if wrist is not None else -1

    if ua < 0 or la < 0 or ws < 0:
        return -1

    # Wrist twist adjustment: tidak tersedia → set 0
    twist_adj = 0
    return rula_groupA_score(ua, la, ws, wrist_twist_adj=twist_adj)


def categorize_rula(final_score: int) -> str:
    if final_score < 0:
        return "Tidak valid"
    if final_score <= 2: return "Baik"
    if final_score <= 4: return "Perlu investigasi"
    if final_score <= 6: return "Perlu perubahan secepatnya"
    return "Sangat buruk"


# ---------- MediaPipe Pose extractor ----------
def extract_pose_keypoints(image_path: Path, pose_model) -> Optional[List[Tuple[float,float,float]]]:
    """
    Kembalikan list [(x,y,visibility)] untuk 33 landmark (normalized) atau None jika gagal.
    """
    import cv2  # pakai OpenCV untuk baca gambar (mediapipe butuh np.ndarray BGR)
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose_model.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lms = res.pose_landmarks.landmark
    out = []
    for lm in lms:
        x = float(lm.x) if (lm.x is not None) else None
        y = float(lm.y) if (lm.y is not None) else None
        v = float(lm.visibility) if (lm.visibility is not None) else None
        out.append((x, y, v))
    # Pastikan panjang 33
    if len(out) != 33:
        return None
    return out


# ---------- Pipeline utama ----------
def list_images(root: Path) -> List[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def resume_read_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if "image_path" in df.columns:
            return set(df["image_path"].astype(str).tolist())
    except Exception:
        pass
    return set()


def main():
    img_root = Path(IMG_DIR)
    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_images = list_images(img_root)
    if not all_images:
        print("❌ Tidak ada gambar ditemukan di folder input.")
        return

    done = resume_read_done(out_csv)
    pending = [p for p in all_images if str(p) not in done]

    print(f"Total gambar: {len(all_images)} | Sudah ada di CSV: {len(done)} | Akan diproses: {len(pending)}")

    # Siapkan header CSV
    key_cols = []
    for i in range(33):
        key_cols += [f"x{i}", f"y{i}"]
    columns = ["image_path"] + key_cols + ["rula_left", "rula_right", "rula_score_final", "rula_category"]

    # Jika CSV belum ada, buat file baru dengan header
    if not out_csv.exists():
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(columns)

    # MediaPipe Pose (recommended settings untuk gambar statis)
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,    # 0,1,2 → 1 = moderate, cukup cepat & stabil
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:

        total = len(pending)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(batches):
            start = b * BATCH_SIZE
            end = min(total, (b + 1) * BATCH_SIZE)
            batch_files = pending[start:end]
            t0 = time.time()

            if USE_TQDM:
                iterator = tqdm(batch_files, desc=f"[Batch {b+1}/{batches}]")
            else:
                iterator = batch_files

            rows_to_append = []
            save_counter = 0

            for img_path in iterator:
                lms = extract_pose_keypoints(img_path, pose)
                if lms is None:
                    # landmark gagal → isi -1
                    xy = []
                    for i in range(33):
                        xy += [-1.0, -1.0]
                    left_score = -1
                    right_score = -1
                else:
                    # ambil hanya (x,y); visibility dipakai untuk validasi
                    xy = []
                    for (x, y, v) in lms:
                        if (x is None) or (y is None) or (v is None) or (v < CONF_THRESHOLD):
                            xy += [-1.0, -1.0]
                        else:
                            xy += [float(x), float(y)]

                    # Skor kiri/kanan
                    left_score  = compute_side_scores(lms, "left")
                    right_score = compute_side_scores(lms, "right")

                # final: worst arm (maks)
                final_score = max(left_score, right_score)
                category = categorize_rula(final_score)

                row = [str(img_path)] + xy + [left_score, right_score, final_score, category]
                rows_to_append.append(row)

                # Tulis sebagian periodik (SAVE_EVERY) agar aman kalo proses lama
                save_counter += 1
                if save_counter >= SAVE_EVERY:
                    df_part = pd.DataFrame(rows_to_append, columns=columns)
                    df_part.to_csv(out_csv, mode="a", header=False, index=False)
                    rows_to_append = []
                    save_counter = 0

            # Tulis sisa batch
            if rows_to_append:
                df_part = pd.DataFrame(rows_to_append, columns=columns)
                df_part.to_csv(out_csv, mode="a", header=False, index=False)

            dt = time.time() - t0
            print(f"✅ Batch {b+1}/{batches} selesai: {end-start} gambar | ⏱️ {dt:.2f}s")

    print(f"\n🎉 Selesai. Hasil disimpan ke: {out_csv}")


if __name__ == "__main__":
    main()
