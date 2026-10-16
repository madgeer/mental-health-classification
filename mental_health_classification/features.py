from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from mental_health_classification.config import (
    CSV_FILENAME,
    FEATURES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET,
)

app = typer.Typer()


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if expected feature/label columns are missing."""
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom hilang di dataset: {missing}")


def get_X_y(df: pd.DataFrame):
    """Return (X: DataFrame[n x 12], y: Series[n]) views over the raw frame."""
    validate_columns(df)
    return df[FEATURES], df[TARGET]


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / CSV_FILENAME,
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """Select the 12 audio features + label from the raw CSV and write features.csv."""
    logger.info("Generating features from dataset...")
    for i in tqdm(range(1), total=1):
        if i == 5:  # pragma: no cover - kept for template parity
            logger.info("Something happened for iteration 5.")
    df = pd.read_csv(input_path, low_memory=False)
    X, y = get_X_y(df)
    out = pd.concat([X, y], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.success(f"Features generation complete: {output_path} {tuple(out.shape)}")


if __name__ == "__main__":
    app()
