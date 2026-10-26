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


def _get_activation(name):
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
    def __init__(self, sizes=(12, 128, 64, 10), init="he", seed=SEED, activation="relu"):
        super().__init__()
        torch.manual_seed(seed)  # for reproducibility
        self.act = _get_activation(activation)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])
        )
        for layer in self.layers:
            init_layer(layer, init)

    @property
    def params(self):
        """Flat parameter list (in registration order) for autograd.grad / manual optimizers."""
        return list(self.parameters())

    def forward(self, x):
        """Return logits only; the last layer has no activation (softmax lives in cross_entropy)."""
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    def loss_and_grads(self, x, y):
        """Compute the loss and its gradients (used at every training step)."""
        loss = F.cross_entropy(self.forward(x), y)
        return loss.item(), list(torch.autograd.grad(loss, self.params))

    @torch.no_grad()
    def loss(self, x, y):
        return F.cross_entropy(self.forward(x), y).item()

    @torch.no_grad()
    def accuracy(self, x, y):
        preds = self.forward(x).argmax(dim=1)
        return (preds == y).float().mean().item()
