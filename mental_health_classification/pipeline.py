"""End-to-end runner: reproduce notebook sections 7-8 & 16-20 in one command.

Usage (from the repo root):

    python -m mental_health_classification.pipeline

Runs, in order: EDA figures -> 1x Adam validation curve -> the 4-optimizer
summary table -> learning curves -> per-class report + confusion matrix, and
finally saves the reported model (with scaler stats) to models/model.pkl so
`python -m mental_health_classification.modeling.predict` works on raw data.
"""

from loguru import logger
import pandas as pd
import typer

from mental_health_classification import dataset as dataset_mod
from mental_health_classification.config import (
    CSV_FILENAME,
    FIGURES_DIR,
    HYPERPARAMS,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
    TARGET,
)
from mental_health_classification.dataset import load_data
from mental_health_classification.modeling.optimizers import make_adam
from mental_health_classification.modeling.predict import report, save_model
from mental_health_classification.modeling.train import OPTIMIZERS, fit_status, train, train_epochs
from mental_health_classification.plots import (
    plot_correlation,
    plot_feature_histograms,
    plot_label_distribution,
    plot_learning_curves,
    plot_training_curve,
)

app = typer.Typer()


@app.command()
def main(
    eda: bool = typer.Option(
        True, help="Buat grafik EDA (distribusi label, histogram, korelasi)."
    ),
    adam_curve: bool = typer.Option(
        True, help="Latih 1 model Adam (300 step) + kurva loss validasi (section 16)."
    ),
    report_optimizer: str = typer.Option(
        "Adam", help="Optimizer untuk laporan per kelas + confusion matrix (section 19-20)."
    ),
):
    """Jalankan alur notebook end-to-end dan simpan gambar/tabel/model."""
    logger.info("Menyiapkan data (load_data)...")
    data = load_data()
    Xval, yval, Xte, yte = data[2], data[3], data[4], data[5]

    # --- EDA (sections 7-8) straight from the raw CSV -------------------------
    if eda:
        raw = pd.read_csv(RAW_DATA_DIR / CSV_FILENAME, low_memory=False)
        plot_label_distribution(raw[TARGET], FIGURES_DIR / "distribusi_label.png")
        plot_feature_histograms(raw, FIGURES_DIR / "distribusi_fitur.png")
        plot_correlation(raw, FIGURES_DIR / "korelasi_fitur.png")

    # --- 1x Adam + validation-loss curve (section 16) -------------------------
    if adam_curve:
        loss, acc_val, acc_test, history = train(
            lambda: make_adam(lr=1e-2),
            steps=HYPERPARAMS["steps"],
            sizes=HYPERPARAMS["sizes"],
            seed=SEED,
            data=data,
            track=True,
        )
        logger.info(f"Adam 1x latih: loss={loss:.4f}, val={acc_val:.4f}, uji={acc_test:.4f}")
        plot_training_curve(history, FIGURES_DIR / "kurva_training.png")

    # --- 4 optimizers, same hyperparameters (section 18) ----------------------
    rows, curves_bank, models_bank = [], {}, {}
    for name, factory, _lr_str in OPTIMIZERS:
        tr_loss, va_loss, tr_acc, va_acc, te_acc, curves, model = train_epochs(
            factory,
            epochs=HYPERPARAMS["epochs"],
            batch_size=HYPERPARAMS["batch_size"],
            sizes=HYPERPARAMS["sizes"],
            init=HYPERPARAMS["init"],
            activation=HYPERPARAMS["activation"],
            seed=SEED,
            data=data,
            return_model=True,
        )
        curves_bank[name] = curves
        models_bank[name] = model
        rows.append(
            {
                "Optimizer": name,
                "Train Loss": tr_loss,
                "Val Loss": va_loss,
                "Train Acc (%)": tr_acc * 100,
                "Val Acc (%)": va_acc * 100,
                "Test Acc (%)": te_acc * 100,
                "Status": fit_status(tr_acc, va_acc, tr_loss, va_loss),
            }
        )

    summary = pd.DataFrame(rows).set_index("Optimizer")
    logger.info("\n" + summary.to_string(float_format="%.4f"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "tabel_ringkasan.csv")

    # --- Learning curves: loss & accuracy vs epoch (section 20) ---------------
    plot_learning_curves(curves_bank, HYPERPARAMS["epochs"], FIGURES_DIR / "learning_curves.png")

    # --- Per-class report + confusion matrix (sections 19-20) -----------------
    if report_optimizer not in models_bank:
        raise typer.BadParameter(
            f"report_optimizer harus salah satu dari {list(models_bank)} (dapat '{report_optimizer}')"
        )
    encoder = dataset_mod._label_encoder
    classes = list(encoder.classes_) if encoder is not None else None
    text = report(
        models_bank[report_optimizer],
        Xval,
        yval,
        Xte,
        yte,
        save_path=FIGURES_DIR / f"confusion_matrix_{report_optimizer.lower()}.png",
        class_names=classes,
    )
    logger.info("Classification report:\n" + str(text))
    (REPORTS_DIR / f"classification_report_{report_optimizer.lower()}.txt").write_text(
        str(text), encoding="utf-8"
    )

    # --- Persist the model so the inference CLI can reuse it ------------------
    scaler = dataset_mod._scaler
    save_model(
        models_bank[report_optimizer],
        MODELS_DIR / "model.pkl",
        mean=scaler.mean_ if scaler is not None else None,
        scale=scaler.scale_ if scaler is not None else None,
        classes=classes,
        init=HYPERPARAMS["init"],
        activation=HYPERPARAMS["activation"],
    )
    logger.success("Pipeline complete.")


if __name__ == "__main__":
    app()
