from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from tqdm import tqdm
import typer

from mental_health_classification.config import (
    CSV_FILENAME,
    FEATURES,
    PROJ_ROOT,
    RAW_DATA_DIR,
    SEED,
    TARGET,
)

app = typer.Typer()

_label_encoder: LabelEncoder | None = None  # reused for class names in report()
_scaler: StandardScaler | None = None  # reused for inference on raw (unscaled) features


def resolve_csv(csv_path: Path | str | None = None) -> Path:
    """Find the source CSV: explicit path > data/raw > repo root (notebook CWD)."""
    if csv_path is not None:
        return Path(csv_path)
    raw = RAW_DATA_DIR / CSV_FILENAME
    if raw.exists():
        return raw
    root = PROJ_ROOT / CSV_FILENAME
    if root.exists():
        return root
    return raw  # default (will raise a clear FileNotFoundError on read)


def load_data(
    csv_path: Path | str | None = None,
    test_size=0.15,
    val_size=0.15,
    random_state=SEED,
):
    """Reload the CSV -> return 6 tensors: (Xtr, ytr, Xval, yval, Xte, yte).

    Steps (mirrors notebook section 9):
      a. Encode text labels -> integers 0..9
      b. Stratified 70% train / 15% val / 15% test split
      c. Standardize features (scaler fit on TRAIN only)
      d. Convert to float32 / long tensors
    """
    global _label_encoder, _scaler

    path = resolve_csv(csv_path)
    data = pd.read_csv(path, low_memory=False)
    X_raw = data[FEATURES].to_numpy(dtype=np.float32)
    y_raw = data[TARGET].to_numpy()

    # a. Encode text labels -> integers 0..9
    _label_encoder = LabelEncoder()
    y_encoded = _label_encoder.fit_transform(y_raw)
    logger.info(f"Daftar kelas: {list(_label_encoder.classes_)}")

    # b1. Split off the test set first (15% of total)
    X_rest, X_te, y_rest, y_te = train_test_split(
        X_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    # b2. Split validation from the remainder (15% of total)
    val_rel = val_size / (1 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_rel,
        random_state=random_state,
        stratify=y_rest,
    )
    logger.info(f"Bagi data -> latih {len(X_tr)}, validasi {len(X_val)}, uji {len(X_te)}")

    # c. Scaling (fit on train only!)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    _scaler = scaler  # kept for predict.save_model()/inference on raw features

    # d. To PyTorch tensors
    return (
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr).long(),
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long(),
        torch.from_numpy(X_te),
        torch.from_numpy(y_te).long(),
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / CSV_FILENAME,
):
    """Load + split the raw CSV and log the resulting tensor shapes."""
    logger.info("Processing dataset...")
    # tqdm placeholder retained from template for CLI parity; real work below.
    for i in tqdm(range(1), total=1):
        if i == 5:  # pragma: no cover - kept for template parity
            logger.info("Something happened for iteration 5.")
    Xtr, ytr, Xval, yval, Xte, yte = load_data(csv_path=input_path)
    logger.info(
        f"Latih: {tuple(Xtr.shape)}, validasi: {tuple(Xval.shape)}, uji: {tuple(Xte.shape)}"
    )
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
