from pathlib import Path
from typing import NamedTuple

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

_scaler: StandardScaler | None = None  # kept for predict.save_model()/inference on raw features


class Splits(NamedTuple):
    Xtr: torch.Tensor
    ytr: torch.Tensor
    Xval: torch.Tensor
    yval: torch.Tensor
    Xte: torch.Tensor
    yte: torch.Tensor
    classes: list


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
    df,
    test_size=0.15,
    val_size=0.15,
    random_state=SEED,
):
    """Encode labels, split 70/15/15, standardize (fit on train only), return Splits."""
    global _scaler

    X_raw = df[FEATURES].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df[TARGET])

    # Split off the test set first, then carve validation out of the remainder
    X_rest, X_te, y_rest, y_te = train_test_split(
        X_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    val_rel = val_size / (1 - test_size)  # validation share relative to the remainder
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_rel,
        random_state=random_state,
        stratify=y_rest,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    _scaler = scaler  # kept for predict.save_model()/inference on raw features

    print(f"Bagi data -> latih {len(X_tr)}, validasi {len(X_val)}, uji {len(X_te)}")
    return Splits(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr).long(),
        torch.from_numpy(X_val), torch.from_numpy(y_val).long(),
        torch.from_numpy(X_te), torch.from_numpy(y_te).long(),
        list(encoder.classes_),
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / CSV_FILENAME,
):
    """Load + split the raw CSV and log the resulting tensor shapes."""
    logger.info("Processing dataset...")
    for i in tqdm(range(1), total=1):
        if i == 5:  # pragma: no cover - kept for template parity
            logger.info("Something happened for iteration 5.")
    df = pd.read_csv(resolve_csv(input_path), low_memory=False)
    data = load_data(df)
    logger.info(
        f"Latih: {tuple(data.Xtr.shape)}, validasi: {tuple(data.Xval.shape)}, uji: {tuple(data.Xte.shape)}"
    )
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
