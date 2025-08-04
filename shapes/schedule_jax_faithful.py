# schedule_jax_faithful.py
import torch

# --- Diffusion Parameters (Consistent with JAX notebook) ---
beta_0 = 0.1
beta_1 = 20.0


def log_alpha(t):
    """Calculates the log of the alpha term (signal rate)."""
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * t * beta_0 - 0.25 * t.pow(2) * (beta_1 - beta_0)


def alpha(t):
    """Calculates the alpha term."""
    return torch.exp(log_alpha(t))


# --- [FIXED] JAX-Faithful Sigma Schedule ---
def log_sigma(t):
    """
    Calculates the log of the sigma term.
    This is a direct translation of the JAX notebook's schedule: `log(t)`.
    An epsilon is added for numerical stability near t=0.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return torch.log(t + 1e-9)


def sigma(t):
    """Calculates the sigma term."""
    return torch.exp(log_sigma(t))


def dlog_alphadt(t):
    """Analytical derivative of log_alpha w.r.t. t."""
    t = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * beta_0 - 0.5 * t * (beta_1 - beta_0)

def beta(t):
    """
    Custom beta(t) term from the JAX notebook. This is used in the
    notebook's specific reverse ODE solver.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return 1 + 0.5 * t * beta_0 + 0.5 * t.pow(2) * (beta_1 - beta_0)

# --- [FIXED] JAX-Faithful g2(t) ---
# The reverse ODE depends on both alpha and sigma. Since we changed sigma,
# we must derive the corresponding g2(t) for the probability flow ODE.
# The general form of the ODE is: dx = [f(x,t) - 0.5*g(t)^2*s(x,t)] dt
# For the Ornstein-Uhlenbeck process, f(x,t) = dlog_alpha/dt * x
# and g(t)^2 = 2 * sigma * d/dt(sigma) + 2 * sigma^2 * dlog_alpha/dt
def g2(t):
    """
    Calculates the diffusion coefficient g^2(t) corresponding to the
    JAX-faithful schedule where sigma(t) = t.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    # d/dt(sigma) for sigma(t)=t is 1.
    d_sigma_dt = 1.0
    sigma_t = sigma(t)
    dlog_alpha_dt_t = dlog_alphadt(t)

    return 2 * sigma_t * d_sigma_dt + 2 * sigma_t.pow(2) * dlog_alpha_dt_t
