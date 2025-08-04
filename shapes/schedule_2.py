# schedule.py
import torch

# --- Diffusion Parameters (Continuous Time t in [0, 1]) ---
# These parameters define the noise schedule and are consistent with the JAX notebook.
beta_0 = 0.1
beta_1 = 20.0


def log_alpha(t):
    """
    Calculates the log of the alpha term (signal rate).
    This is a faithful implementation from the JAX reference.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * t * beta_0 - 0.25 * t.pow(2) * (beta_1 - beta_0)


def alpha(t):
    """Calculates the alpha term (signal rate)."""
    return torch.exp(log_alpha(t))


def log_sigma(t):
    """
    Calculates the log of the sigma term (noise rate).
    This uses the standard Variance Preserving (VP) SDE definition, where
    alpha_t^2 + sigma_t^2 = 1. This is a common and valid choice for diffusion models.
    An epsilon is added for numerical stability near t=0.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    # This is equivalent to log(sqrt(1 - alpha(t)^2))
    return torch.log(1.0 - torch.exp(2 * log_alpha(t)) + 1e-9) / 2


def sigma(t):
    """Calculates the sigma term (noise rate)."""
    return torch.exp(log_sigma(t))


def dlog_alphadt(t):
    """
    Analytical derivative of log_alpha w.r.t. t.
    This is a faithful and correct implementation.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * beta_0 - 0.5 * t * (beta_1 - beta_0)


def g2(t):
    """
    [FIXED] Calculates the diffusion coefficient g_t^2, which is essential for the
    reverse ODE (probability flow) solver.

    For a Variance-Preserving (VP) SDE where sigma_t^2 = 1 - alpha_t^2, the
    forward SDE is: dx_t = (dlog_alpha/dt * x_t) dt + sqrt(-2 * dlog_alpha/dt) dW_t.
    The term g_t^2 is therefore -2 * dlog_alphadt(t).

    The previous `beta(t)` function was incorrect and led to a physically
    inconsistent reverse ODE. This function provides the correct term.
    """
    return -2 * dlog_alphadt(t)


def q_t(x0, t, eps=None):
    """
    Performs the forward diffusion process (noising).
    This correctly adds noise to data x0 at time t according to the VP-SDE schedule.
    """
    if eps is None:
        eps = torch.randn_like(x0)

    # Reshape alpha and sigma to be broadcastable with the image tensor
    alpha_t = alpha(t).view(-1, 1, 1, 1)
    sigma_t = sigma(t).view(-1, 1, 1, 1)

    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps
