import os
import json
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class DualInputRULADataset(Dataset):
    """
    Dual-input dataset:
      - Gambar (RGB)
      - 66-dim keypoints (x0..x32, y0..y32) ATAU satu kolom 'keypoints' berisi 66 angka
      - Label kelas 0..6 (RULA 1..7 -> 0..6), dengan prioritas sumber: rula_score_final > label_idx > label
    """
    def __init__(self, csv_file, image_dir, transform=None, strict=True, debug=True):
        """
        Args:
            csv_file (str): path CSV split (train/val/test)
            image_dir (str): root folder gambar (image_path relatif ke sini)
            transform: torchvision transforms untuk gambar
            strict (bool): True = raise error jika ada baris invalid
            debug (bool): print ringkasan distribusi label
        """
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.strict = strict
        self.debug = debug

        # --- Tentukan kolom keypoints ---
        xy_cols = [f'{a}{i}' for i in range(33) for a in ['x', 'y']]
        has_xy = all(c in self.df.columns for c in xy_cols)
        has_packed = ('keypoints' in self.df.columns)

        if has_xy:
            self.mode = 'xy'
            self.keypoint_cols = xy_cols
            # Pastikan numerik
            self.df[self.keypoint_cols] = self.df[self.keypoint_cols].apply(pd.to_numeric, errors='coerce')
        elif has_packed:
            self.mode = 'packed'
        else:
            raise ValueError(
                "CSV tidak punya kolom keypoints yang dikenali. "
                "Butuh 66 kolom (x0..x32,y0..y32) ATAU satu kolom 'keypoints'."
            )

        # --- Siapkan label 0..6 ---
        labs = None
        used_source = None

        # 1) PRIORITAS: rula_score_final (1..7 → 0..6)
        if 'rula_score_final' in self.df.columns:
            labs_rs = pd.to_numeric(self.df['rula_score_final'], errors='coerce').astype('Int64')
            if labs_rs.notna().any():
                labs = (labs_rs.fillna(-999) - 1).astype(int)
                used_source = 'rula_score_final (mapped 1..7 → 0..6)'

        # 2) fallback: label_idx (sudah 0..6)
        if labs is None and 'label_idx' in self.df.columns:
            labs_li = pd.to_numeric(self.df['label_idx'], errors='coerce').astype('Int64')
            if labs_li.notna().any():
                labs = labs_li.fillna(-999).astype(int)
                used_source = 'label_idx (0..6)'

        # 3) fallback terakhir: label (diasumsikan 0..6 – tetapi beri warning kalau uniknya hanya 0..2)
        if labs is None and 'label' in self.df.columns:
            labs_l = pd.to_numeric(self.df['label'], errors='coerce').astype('Int64')
            if labs_l.notna().any():
                uniq_tmp = sorted(labs_l.dropna().unique().tolist())
                if self.debug and max(uniq_tmp) <= 2:
                    print("⚠️  WARNING: Kolom 'label' terlihat hanya 0..2. "
                          "Pastikan ini memang skor 0..6, bukan kategori 3 kelas.")
                labs = labs_l.fillna(-999).astype(int)
                used_source = 'label (diasumsikan 0..6)'

        if labs is None:
            raise ValueError("CSV harus punya salah satu: 'rula_score_final' (1..7), 'label_idx' (0..6), atau 'label' (0..6).")

        # Filter valid label (0..6)
        mask_label = (labs >= 0) & (labs <= 6)
        if not mask_label.all():
            bad = (~mask_label).sum()
            msg = f"Ada {bad} baris dengan label di luar 0..6. Baris ini akan dibuang."
            if self.strict:
                raise ValueError(msg)
            else:
                print("[WARN]", msg)
        self.df = self.df.loc[mask_label].copy()
        self.df['label_idx'] = labs[mask_label].astype(int).values

        # Pastikan kolom image_path ada
        if 'image_path' not in self.df.columns:
            raise ValueError("CSV tidak memiliki kolom 'image_path'.")

        # Bangun full path; dukung path absolut maupun relatif
        def _full_path(p):
            p = str(p)
            if os.path.isabs(p):
                return p
            return os.path.join(self.image_dir, p)

        full_paths = [ _full_path(p) for p in self.df['image_path'].astype(str).values ]
        exists = np.array([os.path.isfile(p) for p in full_paths])

        if not exists.all():
            missing = (~exists).sum()
            sample_missing = [full_paths[i] for i, ok in enumerate(exists) if not ok][:5]
            msg = f"Ada {missing} file gambar tidak ditemukan. Contoh:\n- " + "\n- ".join(sample_missing)
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                print("[WARN]", msg)

        # Simpan final list path & label
        self.df = self.df.loc[exists].reset_index(drop=True)
        self.img_paths = [ _full_path(p) for p in self.df['image_path'].astype(str).values ]
        self.labels = self.df['label_idx'].astype(int).values

        # Debug ringkasan
        if self.debug:
            uniq = sorted(pd.Series(self.labels).unique().tolist())
            cnts = pd.Series(self.labels).value_counts().sort_index()
            print(f"[DATASET] Sumber label: {used_source}")
            print(f"[DATASET] Range label: min={min(self.labels)}, max={max(self.labels)}, unik={uniq}")
            print(f"[DATASET] Distribusi (0..6): " + " ".join(f"{i}:{int(cnts.get(i,0))}" for i in range(7)))

    def __len__(self):
        return len(self.df)

    def _parse_packed_keypoints(self, s):
        """Parse 'keypoints' (66 angka) dari string/JSON ke np.float32[66]."""
        if pd.isna(s):
            return None
        if isinstance(s, (list, tuple, np.ndarray)):
            arr = np.asarray(s, dtype=np.float32)
        else:
            txt = str(s).strip()
            try:
                # coba JSON list
                obj = json.loads(txt)
                arr = np.asarray(obj, dtype=np.float32)
            except Exception:
                try:
                    # coba python literal (list)
                    obj = ast.literal_eval(txt)
                    arr = np.asarray(obj, dtype=np.float32)
                except Exception:
                    # coba CSV "v1,v2,...,v66"
                    parts = [p for p in txt.split(",") if p.strip() != ""]
                    arr = np.asarray(parts, dtype=np.float32)
        if arr.shape[0] != 66:
            return None
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load image ---
        img_path = self.img_paths[idx]  # sudah full path
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # --- Load keypoints 66-dim ---
        if self.mode == 'xy':
            vals = row.loc[[f'{a}{i}' for i in range(33) for a in ['x', 'y']]]
            vals = pd.to_numeric(vals, errors='coerce').values.astype('float32')
        else:
            vals = self._parse_packed_keypoints(row['keypoints'])
            if vals is None:
                raise ValueError(f"Baris {idx}: kolom 'keypoints' tidak valid (harus 66 angka).")

        if not np.isfinite(vals).all():
            bad_count = np.sum(~np.isfinite(vals))
            raise ValueError(f"Baris {idx}: keypoints mengandung NaN/inf ({bad_count} nilai).")

        keypoints = torch.tensor(vals, dtype=torch.float32)

        # --- Label 0..6 (long) ---
        label = torch.tensor(int(row['label_idx']), dtype=torch.long)

        return img, keypoints, label
