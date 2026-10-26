from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import typer

from mental_health_classification.config import (
    CSV_FILENAME,
    FEATURES,
    FIGURES_DIR,
    RAW_DATA_DIR,
    TARGET,
)

app = typer.Typer()


def plot_label_distribution(y: pd.Series, save_path: Path | None = None):
    """Horizontal log-scale bar chart of class counts (notebook section 7)."""
    counts = y.value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, orient="h", color="steelblue")
    plt.xscale("log")
    for i, v in enumerate(counts.values):
        plt.text(v * 1.08, i, str(v), va="center", fontsize=9)
    plt.xlabel("Jumlah sampel (skala log)")
    plt.ylabel("Mental_Health_Label")
    plt.title("Distribusi label (log-scale: mayoritas vs minoritas terlihat semua)")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return counts


def plot_feature_histograms(df: pd.DataFrame, save_path: Path | None = None):
    """4x3 histograms + KDE for the 12 audio features."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], bins=50, kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Sebaran 12 fitur audio (cek skala & kemiringan)", fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_correlation(df: pd.DataFrame, save_path: Path | None = None):
    """Pearson correlation heatmap across the 12 features (notebook section 8)."""
    corr = df[FEATURES].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Korelasi antar fitur musik")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return corr


def plot_training_curve(epoch_curves: dict, save_path: Path | None = None):
    """Training curves (train vs val loss/acc) over epochs (notebook section 16)."""
    epochs_axis = range(1, len(epoch_curves["val_loss"]) + 1)
    plt.figure()
    plt.plot(epochs_axis, epoch_curves["train_loss"], label="Train", color="#1f77b4", linewidth=2)
    plt.plot(epochs_axis, epoch_curves["val_loss"], label="Val", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Kurva training (train vs validasi)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return epoch_curves


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path | None = None):
    """Test confusion matrix (notebook section 20)."""
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.title("Confusion Matrix — Adam (data uji)")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return cm


def plot_learning_curves(curves_bank: dict, epochs: int, save_path: Path | None = None):
    """2xN grid of train-vs-val loss/accuracy curves, one column per optimizer."""
    names = list(curves_bank)
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 8), sharex=True)
    if len(names) == 1:
        axes = np.array(axes).reshape(2, 1)
    epochs_axis = range(1, epochs + 1)
    for idx, name in enumerate(names):
        c = curves_bank[name]
        ax_loss = axes[0, idx]
        ax_loss.plot(epochs_axis, c["train_loss"], label="Train", linewidth=2)
        ax_loss.plot(epochs_axis, c["val_loss"], label="Val", linewidth=2)
        ax_loss.set_title(f"Loss - {name}", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)

        ax_acc = axes[1, idx]
        ax_acc.plot(epochs_axis, np.array(c["train_acc"]) * 100, label="Train", linewidth=2)
        ax_acc.plot(epochs_axis, np.array(c["val_acc"]) * 100, label="Val", linewidth=2)
        ax_acc.set_title(f"Accuracy - {name}", fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(loc="lower right")
        ax_acc.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / CSV_FILENAME,
    output_path: Path = FIGURES_DIR / "plot.png",
):
    """Regenerate the EDA figures (sections 7-8) straight from the raw CSV."""
    logger.info("Generating plot from data...")
    for i in tqdm(range(1), total=1):
        if i == 5:  # pragma: no cover - kept for template parity
            logger.info("Something happened for iteration 5.")
    df = pd.read_csv(input_path, low_memory=False)
    output_path = Path(output_path)
    plot_label_distribution(df[TARGET], output_path.parent / "distribusi_label.png")
    plot_feature_histograms(df, output_path.parent / "distribusi_fitur.png")
    plot_correlation(df, output_path.parent / "korelasi_fitur.png")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
