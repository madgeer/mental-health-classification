import math
from typing import NamedTuple

from loguru import logger
import numpy as np
import torch
from torch import nn
import typer

from mental_health_classification.config import HYPERPARAMS, SEED
from mental_health_classification.dataset import Splits, load_data
from mental_health_classification.modeling.model import MLP
from mental_health_classification.modeling.optimizers import (
    make_adam,
    make_momentum,
    make_rmsprop,
    make_sgd,
)

app = typer.Typer()

OPTIMIZERS = [
    ("Adam", lambda: make_adam(lr=1e-3), "1e-3"),
    ("RMSprop", lambda: make_rmsprop(lr=1e-3), "1e-3"),
    ("Momentum", lambda: make_momentum(lr=1e-2), "1e-2"),
    ("SGD", lambda: make_sgd(lr=1e-2), "1e-2"),
]


class TrainResult(NamedTuple):
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    test_acc: float
    epoch_curves: dict[str, list[float]]  # train/val loss & acc per epoch
    model: nn.Module
    diverged: bool


def _make_model(data: Splits, sizes, init: str, activation, seed: int) -> MLP:
    """Build an MLP whose first/last dims are inferred from the data."""
    sizes = [data.Xtr.shape[1], *sizes[1:-1], int(data.ytr.max().item()) + 1]
    return MLP(tuple(sizes), init, seed, activation)


def _minibatches(n, batch_size, rng):
    """Infinite shuffled minibatch index generator; reshuffles at each epoch."""
    while True:
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            yield torch.as_tensor(order[s:s + batch_size])


def _diverged(batch_loss, params):
    return not math.isfinite(batch_loss) or not all(torch.isfinite(p).all() for p in params)


def train(
    opt_factory,
    data: Splits,
    *,
    epochs=20,
    batch_size=64,
    sizes=(12, 128, 64, 10),
    init="he",
    activation="relu",
    seed=SEED,
) -> TrainResult:
    """Train one model for `epochs` full passes over the TRAIN data.

    Minibatches come from TRAIN data only; test data stays untouched until the end.
    Returns a TrainResult with per-epoch curves and a `diverged` flag (NaN/inf loss or weights).
    """
    model = _make_model(data, sizes, init, activation, seed)
    step_fn = opt_factory()
    rng = np.random.default_rng(1000 + seed)
    n, Xtr, ytr = len(data.Xtr), data.Xtr, data.ytr
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def fail():
        return TrainResult(math.inf, math.inf, float("nan"), float("nan"), float("nan"), curves, model, True)

    batches = _minibatches(n, batch_size, rng)
    for _ in range(epochs):
        for _ in range(math.ceil(n / batch_size)):
            idx = next(batches)
            loss, grads = model.loss_and_grads(Xtr[idx], ytr[idx])
            step_fn(model.params, grads)
            if _diverged(loss, model.params):
                return fail()
        curves["train_loss"].append(model.loss(Xtr, ytr))
        curves["val_loss"].append(model.loss(data.Xval, data.yval))
        curves["train_acc"].append(model.accuracy(Xtr, ytr))
        curves["val_acc"].append(model.accuracy(data.Xval, data.yval))

    train_loss = model.loss(Xtr, ytr)
    if not math.isfinite(train_loss):
        return fail()
    return TrainResult(
        train_loss, model.loss(data.Xval, data.yval),
        model.accuracy(Xtr, ytr), model.accuracy(data.Xval, data.yval),
        model.accuracy(data.Xte, data.yte),
        curves, model, False,
    )


def fit_status(train_acc, val_acc, train_loss, val_loss, gap_tol=0.03):
    """Klasifikasi Fit/Overfit sederhana dari gap train-val."""
    if not all(math.isfinite(v) for v in [train_acc, val_acc, train_loss, val_loss]):
        return "Diverged"
    gap = train_acc - val_acc
    if gap > gap_tol or val_loss > train_loss + 0.10:
        return "Overfit"
    if train_acc < 0.60 and val_acc < 0.60:
        return "Underfit"
    return "Fit"


@app.command()
def main():
    """Train one model per optimizer (notebook section 18) and print the summary."""
    data = load_data()
    runs = {}
    for name, factory, _ in OPTIMIZERS:
        runs[name] = train(
            factory, data,
            epochs=HYPERPARAMS["epochs"],
            batch_size=HYPERPARAMS["batch_size"],
            sizes=HYPERPARAMS["sizes"],
            init=HYPERPARAMS["init"],
            activation=HYPERPARAMS["activation"],
        )

    import pandas as pd  # noqa: E402

    # Full summary: train/val loss & acc, honest metrics (balanced acc, macro-F1), status
    rows = []
    from mental_health_classification.modeling.predict import metric_summary, predict  # noqa: E402

    for name, r in runs.items():
        preds = predict(r.model, data.Xte)
        m = metric_summary(data.yte, preds)
        rows.append({
            "Optimizer": name,
            "Train Loss": r.train_loss,
            "Val Loss": r.val_loss,
            "Train Acc (%)": r.train_acc * 100,
            "Val Acc (%)": r.val_acc * 100,
            "Test Acc (%)": r.test_acc * 100,
            "Bal. Acc (%)": m["balanced_accuracy"] * 100,
            "Macro-F1 (%)": m["macro_f1"] * 100,
            "Status": fit_status(r.train_acc, r.val_acc, r.train_loss, r.val_loss),
        })
    df = pd.DataFrame(rows).set_index("Optimizer")
    logger.info("\n" + df.to_string(float_format="%.4f"))
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
