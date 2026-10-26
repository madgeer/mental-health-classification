from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
from tqdm import tqdm
import typer

from mental_health_classification.config import (
    CSV_FILENAME,
    FEATURES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    TARGET,
)
from mental_health_classification.modeling.model import MLP

app = typer.Typer()


@torch.no_grad()
def predict(model, X):
    """Return the predicted classes (0..9) for a tensor batch."""
    return model(X).argmax(dim=1).cpu().numpy()


def metric_summary(y_true, y_pred):
    """Return an {accuracy, balanced_accuracy, macro_f1} dict from numpy arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def report(model, Xval, yval, Xte, yte, save_path: Path | None = None, class_names=None):
    """Print val & test metrics and plot the test confusion matrix.

    `class_names` defaults to the classes from the last `load_data()` call; when
    neither is available a numeric fallback is used (so seaborn never gets None).
    """
    yval_true = np.asarray(yval)
    yte_true = np.asarray(yte)
    yval_pred, yte_pred = predict(model, Xval), predict(model, Xte)

    for name, yt, yp in [("validasi", yval_true, yval_pred), ("uji", yte_true, yte_pred)]:
        m = metric_summary(yt, yp)
        logger.info(
            f"{name:8s} -> acc={m['accuracy']:.4f}, "
            f"balanced={m['balanced_accuracy']:.4f}, macro-F1={m['macro_f1']:.4f}"
        )

    names = list(class_names) if class_names is not None else get_class_names()
    if names is None:
        n_classes = int(max(yte_true.max(initial=0), yte_pred.max(initial=0))) + 1
        names = [str(i) for i in range(n_classes)]
    labels = list(range(len(names)))

    # Confusion matrix (test only — the final one)
    cm = confusion_matrix(yte_true, yte_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.title("Confusion matrix (data uji)")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix disimpan ke: {save_path}")
    plt.show()
    return classification_report(
        yte_true, yte_pred, labels=labels, target_names=names, digits=4, zero_division=0
    )


def save_model(
    model,
    path: Path = MODELS_DIR / "model.pkl",
    *,
    mean=None,
    scale=None,
    classes=None,
    sizes=None,
    init="he",
    activation="relu",
):
    """Persist model weights + everything inference needs (scaler stats, classes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sizes is None:
        sizes = [model.layers[0].in_features, *[layer.out_features for layer in model.layers]]
    bundle = {
        "state_dict": model.state_dict(),
        "sizes": list(sizes),
        "init": init,
        "activation": activation,
        "mean": None if mean is None else np.asarray(mean, dtype=np.float32),
        "scale": None if scale is None else np.asarray(scale, dtype=np.float32),
        "classes": None if classes is None else list(classes),
        "features": list(FEATURES),
    }
    torch.save(bundle, path)
    logger.info(f"Model disimpan ke: {path}")
    return path


def load_model(path: Path = MODELS_DIR / "model.pkl"):
    """Rebuild a model saved by `save_model()`; return (model, bundle)."""
    bundle = torch.load(Path(path), weights_only=False)
    model = MLP(tuple(bundle["sizes"]), bundle["init"], SEED, bundle["activation"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def _standardize(X: np.ndarray, bundle: dict) -> np.ndarray:
    mean, scale = bundle.get("mean"), bundle.get("scale")
    if mean is None or scale is None:
        logger.warning(
            "Bundle tanpa mean/scale; fitur dipakai apa adanya (tidak distandardisasi)."
        )
        return X
    return (X - mean) / scale


@app.command()
def main(
    features_path: Path = RAW_DATA_DIR / CSV_FILENAME,
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Run inference on a CSV of features using a model saved by `save_model()`.

    Writes the input rows plus a `prediction` column. If the label column is
    present, the honest metrics (accuracy/balanced/macro-F1) are logged too.
    """
    logger.info("Performing inference for model...")
    for i in tqdm(range(1), total=1):
        if i == 5:  # pragma: no cover - kept for template parity
            logger.info("Something happened for iteration 5.")

    model, bundle = load_model(model_path)
    df = pd.read_csv(features_path, low_memory=False)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom fitur hilang di {features_path}: {missing}")

    X = _standardize(df[FEATURES].to_numpy(dtype=np.float32), bundle)
    preds = predict(model, torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)))

    classes = bundle.get("classes")
    out = df.copy()
    out["prediction"] = [classes[p] if classes is not None else int(p) for p in preds]

    if TARGET in df.columns:
        y_true = df[TARGET].to_numpy()
        if classes is not None and y_true.dtype.kind in "OUS":
            lookup = {name: i for i, name in enumerate(classes)}
            y_true = np.array([lookup.get(v, -1) for v in y_true])
        m = metric_summary(y_true, preds)
        logger.info(
            f"Metrik vs {TARGET}: acc={m['accuracy']:.4f}, "
            f"balanced={m['balanced_accuracy']:.4f}, macro-F1={m['macro_f1']:.4f}"
        )

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictions_path, index=False)
    logger.info(f"Prediksi disimpan ke: {predictions_path} {tuple(out.shape)}")
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
