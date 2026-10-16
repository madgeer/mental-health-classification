import math

from loguru import logger
import numpy as np
import torch
import typer

from mental_health_classification.config import HYPERPARAMS, SEED
from mental_health_classification.dataset import load_data
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


def train(
    opt_factory,
    steps=300,
    batch_size=64,
    sizes=(12, 64, 32, 10),
    init="he",
    activation="relu",
    seed=0,
    data=None,
    track=False,
    return_model=False,
):
    """Train one model; return (train loss, val accuracy, test accuracy, history).

    - Minibatches are reshuffled each epoch, drawn from TRAIN data only.
    - track=True records VALIDATION loss each step (for model selection);
      history[0] = before training. Test data stays untouched during tuning.
    - A run with NaN/inf loss or weights is marked diverged (loss = inf).
    - return_model=True appends the trained model as the last element,
      for analysis (F1, confusion matrix) without retraining.
    """
    Xtr, ytr, Xval, yval, Xte, yte = data if data is not None else load_data()
    # Auto-fix the input/output dims if they mismatch the data
    sizes = list(sizes)
    sizes[0], sizes[-1] = Xtr.shape[1], int(ytr.max().item()) + 1
    model = MLP(tuple(sizes), init, seed, activation)
    step = opt_factory()

    rng = np.random.default_rng(1000 + seed)
    history = [model.loss(Xval, yval)] if track else []
    order, pos = rng.permutation(len(Xtr)), 0

    for _ in range(steps):
        if pos + batch_size > len(Xtr):  # epoch exhausted -> reshuffle
            order, pos = rng.permutation(len(Xtr)), 0
        idx = torch.as_tensor(order[pos : pos + batch_size])
        pos += batch_size

        batch_loss, grads = model.loss_and_grads(Xtr[idx], ytr[idx])
        step(model.params, grads)

        if not math.isfinite(batch_loss) or not all(torch.isfinite(p).all() for p in model.params):
            failed = (math.inf, float("nan"), float("nan"), history + [math.inf])
            return (*failed, model) if return_model else failed
        if track:
            history.append(model.loss(Xval, yval))

    final_loss = model.loss(Xtr, ytr)
    if not math.isfinite(final_loss):
        failed = (math.inf, float("nan"), float("nan"), history + [math.inf])
        return (*failed, model) if return_model else failed
    result = (final_loss, model.accuracy(Xval, yval), model.accuracy(Xte, yte), history)
    return (*result, model) if return_model else result


def evaluate(opt_factory, trials=5, **kwargs):
    """Repeat training over several seeds; return (loss, val acc, test acc, diverged)."""
    data = load_data()
    losses, val_accs, test_accs, diverged = [], [], [], 0
    for t in range(trials):
        loss, av, at, _ = train(opt_factory, seed=t, data=data, **kwargs)
        if math.isfinite(loss):
            losses.append(loss)
            val_accs.append(av)
            test_accs.append(at)
        else:
            diverged += 1
    if not losses:
        return math.inf, float("nan"), float("nan"), diverged
    return float(np.mean(losses)), float(np.mean(val_accs)), float(np.mean(test_accs)), diverged


def train_epochs(
    opt_factory,
    epochs=20,
    batch_size=64,
    sizes=(12, 64, 32, 10),
    init="he",
    activation="relu",
    seed=SEED,
    data=None,
    return_model=False,
):
    """Latih per-epoch; return (train_loss, val_loss, train_acc, val_acc, test_acc, curves[, model]).

    curves = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[]) per epoch.
    1 epoch = satu pass penuh atas data latih (reshuffle tiap epoch).
    """
    Xtr, ytr, Xval, yval, Xte, yte = data if data is not None else load_data()
    sizes = list(sizes)
    sizes[0], sizes[-1] = Xtr.shape[1], int(ytr.max().item()) + 1
    model = MLP(tuple(sizes), init, seed, activation)
    step_fn = opt_factory()
    rng = np.random.default_rng(1000 + seed)
    n = len(Xtr)
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = torch.as_tensor(order[s : s + batch_size])
            batch_loss, grads = model.loss_and_grads(Xtr[idx], ytr[idx])
            step_fn(model.params, grads)
            if not math.isfinite(batch_loss) or not all(
                torch.isfinite(p).all() for p in model.params
            ):
                out = (math.inf, math.inf, float("nan"), float("nan"), float("nan"), curves)
                return (*out, model) if return_model else out
        curves["train_loss"].append(model.loss(Xtr, ytr))
        curves["val_loss"].append(model.loss(Xval, yval))
        curves["train_acc"].append(model.accuracy(Xtr, ytr))
        curves["val_acc"].append(model.accuracy(Xval, yval))
    out = (
        curves["train_loss"][-1],
        curves["val_loss"][-1],
        curves["train_acc"][-1],
        curves["val_acc"][-1],
        model.accuracy(Xte, yte),
        curves,
    )
    return (*out, model) if return_model else out


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
    import pandas as pd

    data = load_data()
    rows = []
    for name, factory, _ in OPTIMIZERS:
        tr_loss, va_loss, tr_acc, va_acc, te_acc, _ = train_epochs(
            factory,
            epochs=HYPERPARAMS["epochs"],
            batch_size=HYPERPARAMS["batch_size"],
            sizes=HYPERPARAMS["sizes"],
            init=HYPERPARAMS["init"],
            activation=HYPERPARAMS["activation"],
            seed=SEED,
            data=data,
        )
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
    df = pd.DataFrame(rows).set_index("Optimizer")
    logger.info("\n" + df.to_string(float_format="%.4f"))
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
