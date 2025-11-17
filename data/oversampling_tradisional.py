# make_splits_moderate_oversample_safe.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ========= KONFIGURASI =========
seed = 42
np.random.seed(seed)

# Sumber data (boleh 1 atau gabungan beberapa CSV)
INPUT_CSVS = [
    r"D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\keypoints_rula_FULL_clean.csv"
]

# Kolom penting
IMG_COL   = "image_path"
LABEL_COL = "rula_score_final"

# Output folder
OUT_DIR = r"D:\Kuliah\Sem 7\TA1\New Env Tugas Akhir\output\balanced_splits"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_OVER = os.path.join(OUT_DIR, "train_over.csv")
VAL_REAL   = os.path.join(OUT_DIR, "val_real.csv")
TEST_REAL  = os.path.join(OUT_DIR, "test_real.csv")
VAL_BAL    = os.path.join(OUT_DIR, "val_bal.csv")   # opsional evaluasi
TEST_BAL   = os.path.join(OUT_DIR, "test_bal.csv")  # opsional evaluasi

# Rasio split global
TEST_PCT = 0.10
VAL_PCT  = 0.10

# Minimum per kelas untuk VAL/TEST (agar minoritas tetap ada representasi)
MIN_VAL_PER_CLASS  = 20
MIN_TEST_PER_CLASS = 20

# Target oversampling moderat di TRAIN
TARGET_PER_CLASS = 3000        # 3000 atau 8000 
MAX_DUP_PER_IMAGE = 40        # batas duplikasi tiap image di TRAIN

# Apabila ada image yang muncul ganda di beberapa CSV dengan label beda → ambil modus
RESOLVE_CONFLICT_LABEL = True

# ========= UTIL =========
def show_dist(name, df):
    vc = df[LABEL_COL].value_counts().sort_index()
    print(f"\n=== Distribusi {name} ===")
    for k, v in vc.items():
        print(f"Kelas {k}: {v}")
    print("TOTAL:", len(df))

def stratified_split_with_min_per_class(df, test_pct, val_pct, min_test, min_val):
    """Split  (train_base, val, test)
       - Stratified by LABEL_COL
       - Enforce minimum per class for VAL and TEST
    """
    # Bekerja per kelas
    train_parts, val_parts, test_parts = [], [], []
    for c in range(1, 8):
        sub = df[df[LABEL_COL] == c]
        n = len(sub)
        if n == 0:
            continue

        # hitung target jumlah
        n_test = int(round(test_pct * n))
        n_val  = int(round(val_pct * n))

        # enforce minimum
        n_test = max(n_test, min( n-1, min_test)) if n > min_test else max(1, min(n, int(round(test_pct*n))))
        # sisa setelah test
        remaining = n - n_test
        n_val = max(n_val, min(remaining-1, min_val)) if remaining > min_val else max(1, min(remaining, int(round(val_pct*n))))

        # kalau tetap tidak muat, turunkan min agar selalu ada train
        if n_test + n_val >= n:
            # fallback: jaga minimal 1 untuk train
            overflow = (n_test + n_val) - (n - 1)
            # kurangi dari n_val dulu
            reduce_val = min(overflow, max(0, n_val - 1))
            n_val -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                n_test = max(1, n_test - overflow)

        # sampling
        sub = sub.sample(frac=1.0, random_state=seed)
        test_c = sub.iloc[:n_test]
        val_c  = sub.iloc[n_test:n_test+n_val]
        train_c = sub.iloc[n_test+n_val:]

        test_parts.append(test_c)
        val_parts.append(val_c)
        train_parts.append(train_c)

    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    val_df  = pd.concat(val_parts,  ignore_index=True) if val_parts  else pd.DataFrame(columns=df.columns)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df

def oversample_moderate(train_df, target_per_class, max_dup_per_image):
    """Oversample ke target moderat dengan batas duplikasi per image.
       Return: train_over (dengan kolom tambahan 'is_dupe': 0 asli, 1 hasil duplikasi)
    """
    parts = []
    for c in range(1, 8):
        sub = train_df[train_df[LABEL_COL] == c].copy()
        n = len(sub)
        if n == 0:
            print(f"[W] Kelas {c} kosong di TRAIN_BASE → tidak di-oversample.")
            continue

        # target efektif tak boleh lebih dari n * max_dup_per_image
        effective_target = min(target_per_class, n * max_dup_per_image)

        if effective_target <= n:
            sub["is_dupe"] = 0
            parts.append(sub)
            continue

        repeats = effective_target // n
        rest    = effective_target - repeats * n

        repeats = min(repeats, max_dup_per_image)
        effective_target = repeats * n + rest

        # satu kopi 'asli' + (repeats-1) kopi dupe
        chunk = [sub.assign(is_dupe=0)]
        for _ in range(repeats - 1):
            chunk.append(sub.copy().assign(is_dupe=1))
        if rest > 0:
            chunk.append(sub.sample(rest, replace=True, random_state=seed).assign(is_dupe=1))
        parts.append(pd.concat(chunk, ignore_index=True))

    train_over = pd.concat(parts, ignore_index=True) if parts else train_df.copy()
    train_over = train_over.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_over

def make_balanced_eval(df, per_class):
    """Undersample setiap kelas ke per_class untuk evaluasi seimbang."""
    chunks = []
    for c in range(1, 8):
        sub = df[df[LABEL_COL] == c]
        if len(sub) == 0:
            continue
        if len(sub) <= per_class:
            chunks.append(sub)
        else:
            chunks.append(sub.sample(per_class, random_state=seed))
    if not chunks:
        return df.copy()
    out = pd.concat(chunks, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ========= MAIN =========
def main():
    # gabung sumber
    dfs = [pd.read_csv(p) for p in INPUT_CSVS]
    data = pd.concat(dfs, ignore_index=True)

    # bersihkan
    data = data.dropna(subset=[IMG_COL, LABEL_COL]).copy()
    data = data[data[LABEL_COL].between(1, 7)].copy()
    data[LABEL_COL] = data[LABEL_COL].astype(int)

    # resolusi konflik label per image_path (opsional)
    if RESOLVE_CONFLICT_LABEL:
        mode_label = data.groupby(IMG_COL)[LABEL_COL].agg(lambda x: x.mode().iloc[0])
        data = data.merge(mode_label.rename("mode_label"), left_on=IMG_COL, right_index=True)
        data = data[data[LABEL_COL] == data["mode_label"]].drop(columns=["mode_label"])

    # drop duplikat exact image
    data = data.drop_duplicates(subset=[IMG_COL]).reset_index(drop=True)

    show_dist("DATA (raw, valid 1..7)", data)

    # stratified split dengan minimum per kelas
    train_base, val_df, test_df = stratified_split_with_min_per_class(
        data, TEST_PCT, VAL_PCT, MIN_TEST_PER_CLASS, MIN_VAL_PER_CLASS
    )

    show_dist("TRAIN_BASE (sebelum OS)", train_base)
    show_dist("VAL_REAL", val_df)
    show_dist("TEST_REAL", test_df)

    # oversample moderat hanya di TRAIN
    train_over = oversample_moderate(train_base, TARGET_PER_CLASS, MAX_DUP_PER_IMAGE)
    show_dist("TRAIN_OVER (moderate OS)", train_over)

    # simpan
    train_over.to_csv(TRAIN_OVER, index=False)
    val_df.to_csv(VAL_REAL, index=False)
    test_df.to_csv(TEST_REAL, index=False)
    print(f"\n✅ Simpan:")
    print("  -", TRAIN_OVER)
    print("  -", VAL_REAL)
    print("  -", TEST_REAL)

    # opsional: balanced eval (undersampling)
    val_bal  = make_balanced_eval(val_df,  min(100, val_df[LABEL_COL].value_counts().min() if len(val_df) else 0))
    test_bal = make_balanced_eval(test_df, min(150, test_df[LABEL_COL].value_counts().min() if len(test_df) else 0))
    if len(val_bal):
        val_bal.to_csv(VAL_BAL, index=False);  show_dist("VAL_BAL (opsional)",  val_bal)
        print("  -", VAL_BAL)
    if len(test_bal):
        test_bal.to_csv(TEST_BAL, index=False); show_dist("TEST_BAL (opsional)", test_bal)
        print("  -", TEST_BAL)

if __name__ == "__main__":
    main()
