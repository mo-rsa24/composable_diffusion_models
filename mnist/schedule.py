# schedules.py
import torch

# --- Diffusion Parameters (Continuous Time t in [0, 1]) ---
beta_0 = 0.1
beta_1 = 20.0


def log_alpha(t):
    """Calculates the log of the alpha term."""
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * t * beta_0 - 0.25 * t.pow(2) * (beta_1 - beta_0)


def alpha(t):
    """Calculates the alpha term."""
    return torch.exp(log_alpha(t))


def log_sigma(t):
    """Calculates the log of the sigma term.
    Todo: Be watchful of this function
    Before it was: t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return torch.log(t_tensor + 1e-9)  # Add epsilon to avoid log(0)
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return torch.log(1 - torch.exp(2 * log_alpha(t)) + 1e-9) / 2


def sigma(t):
    """Calculates the sigma term."""
    return torch.exp(log_sigma(t))


def dlog_alphadt(t):
    """Analytical derivative of log_alpha w.r.t. t."""
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * beta_0 - 0.5 * t * (beta_1 - beta_0)


def beta(t):
    """Calculates the beta term, related to the diffusion coefficient.
    Todo: Be watchful of this one too
    Before it was:t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return (1 + 0.5 * t_tensor * beta_0 + 0.5 * t_tensor ** 2 * (beta_1 - beta_0))
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return -2 * dlog_alphadt(t) * sigma(t) ** 2


def q_t(x0, t, eps=None):
    """
    Performs the forward diffusion process to add noise to the data x0 at time t.
    """
    if eps is None:
        eps = torch.randn_like(x0)

    alpha_t = alpha(t).view(-1, 1, 1, 1)
    sigma_t = sigma(t).view(-1, 1, 1, 1)

    xt = alpha_t * x0 + sigma_t * eps
    return xt, eps