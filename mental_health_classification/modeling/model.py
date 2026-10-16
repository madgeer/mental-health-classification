import math

import torch
from torch import nn
import torch.nn.functional as F

from mental_health_classification.config import SEED


def init_layer(layer: nn.Linear, init: str = "he"):
    """Re-initialize one nn.Linear in place.

    Options for `init`:
      default = keep PyTorch's default init, small = N(0, 0.01),
      const = 0.1, zero = 0, large = N(0, 1.0),
      xavier / he = Gaussian scaled by layer size.
    Bias is always zeroed (except in "default" mode).
    """
    W, b = layer.weight, layer.bias
    fan_out, fan_in = W.shape  # PyTorch stores weights as [out, in]

    with torch.no_grad():
        if init == "default":
            return
        elif init == "zero":
            W.zero_()
        elif init == "const":
            W.fill_(0.1)
        elif init == "small":
            W.normal_(0, 0.01)
        elif init == "large":
            W.normal_(0, 1.0)
        elif init == "xavier":
            W.normal_(0, math.sqrt(2.0 / (fan_in + fan_out)))
        elif init == "he":
            W.normal_(0, math.sqrt(2.0 / fan_in))
        else:
            raise ValueError(f"Unknown init: {init}")
        b.zero_()


def get_activation(name):
    options = {
        "relu": F.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "leaky_relu": F.leaky_relu,
        "gelu": F.gelu,
        "elu": F.elu,
    }
    if callable(name):  # a custom function (e.g. F.silu) is also accepted
        return name
    try:
        return options[name]
    except KeyError:
        raise ValueError(f"Unknown activation: {name}. Choose from: {sorted(options)}")


class MLP(nn.Module):
    def __init__(self, sizes=(12, 64, 32, 10), init="he", seed=SEED, activation="relu"):
        super().__init__()
        torch.manual_seed(seed)  # for reproducibility
        self.act = get_activation(activation)
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])
        )
        for layer in self.layers:
            init_layer(layer, init)
        # Flat parameter list for the manual optimizers below
        self.params = [p for layer in self.layers for p in (layer.weight, layer.bias)]

    def forward(self, x):
        """Return (logits, per-layer activations). No activation on the last layer."""
        acts = [x]
        h = x
        for i, layer in enumerate(self.layers):
            z = layer(h)
            h = z if i == len(self.layers) - 1 else self.act(z)
            acts.append(h)
        return h, acts

    def _compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)  # mean cross-entropy

    def loss_and_grads(self, x, y):
        """Compute the loss and its gradients (used at every training step)."""
        logits, _ = self.forward(x)
        loss = self._compute_loss(logits, y)
        grads = torch.autograd.grad(loss, self.params)
        return loss.item(), list(grads)

    @torch.no_grad()
    def loss(self, x, y):
        logits, _ = self.forward(x)
        return self._compute_loss(logits, y).item()

    @torch.no_grad()
    def accuracy(self, x, y):
        logits, _ = self.forward(x)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()
