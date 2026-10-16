import torch


def make_sgd(lr=0.1):
    """Plain SGD: w -= lr * gradient."""

    def step(params, grads):
        with torch.no_grad():
            for p, g in zip(params, grads):
                p -= lr * g

    return step


def make_momentum(lr=0.01, beta=0.9):
    """SGD + momentum: reuse the previous direction to reduce oscillation."""
    velocity = {}

    def step(params, grads):
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                velocity[i] = beta * velocity.get(i, 0) + g
                p -= lr * velocity[i]

    return step


def make_rmsprop(lr=0.001, rho=0.99, eps=1e-8):
    """RMSprop: a per-weight adaptive learning rate."""
    sq_avg = {}

    def step(params, grads):
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                sq_avg[i] = rho * sq_avg.get(i, 0) + (1 - rho) * g * g
                p -= lr * g / (torch.sqrt(sq_avg[i]) + eps)

    return step


def make_adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """Adam = momentum + RMSprop + bias correction for the early steps."""
    m, v = {}, {}
    step_count = 0

    def step(params, grads):
        nonlocal step_count
        step_count += 1
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m.get(i, 0) + (1 - beta1) * g
                v[i] = beta2 * v.get(i, 0) + (1 - beta2) * g * g
                if bias_correction:
                    m_hat = m[i] / (1 - beta1**step_count)
                    v_hat = v[i] / (1 - beta2**step_count)
                else:
                    m_hat, v_hat = m[i], v[i]
                p -= lr * m_hat / (torch.sqrt(v_hat) + eps)

    return step
