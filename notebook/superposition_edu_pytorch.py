import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def sample_data(bs, up=True, device='cpu'):
    """
    Generates sample data based on two distinct cluster locations,
    controlled by the 'up' parameter.

    Args:
      bs (int): The batch size.
      up (bool): If True, samples from the upper cluster; otherwise,
                 samples from the lower cluster.
      device (str): The device to place the tensors on.

    Returns:
      torch.Tensor: A tensor of shape (bs, 2) representing the sampled data.
    """
    if up:
        # Samples from a line segment at y=1
        min_vals = torch.tensor([0, 1], device=device)
        max_vals = torch.tensor([2, 2], device=device)
    else:
        # Samples from a line segment at y=0
        min_vals = torch.tensor([0, 0], device=device)
        max_vals = torch.tensor([2, 1], device=device)

    # Create a range for randint
    # Note: torch.randint's high is exclusive, so we add 1
    x_1 = torch.stack([
        torch.randint(min_vals[0], max_vals[0], (bs,), device=device),
        torch.randint(min_vals[1], max_vals[1], (bs,), device=device)
    ], dim=1).float()

    x_1 = 3 * (x_1 - 0.5)
    x_1 += 4e-1 * torch.randn_like(x_1)
    return x_1


# --- Diffusion Parameters and Functions ---
ndim = 2
t_0, t_1 = 0.0, 1.0
beta_0 = 0.1
beta_1 = 20.0


def log_alpha(t):
    """Calculates the log of the alpha term."""
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * t_tensor * beta_0 - 0.25 * t_tensor ** 2 * (beta_1 - beta_0)


def log_sigma(t):
    """Calculates the log of the sigma term."""
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return torch.log(t_tensor + 1e-9)  # Add epsilon to avoid log(0)


def dlog_alphadt(t):
    """Analytical derivative of log_alpha w.r.t. t."""
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * beta_0 - 0.5 * t_tensor * (beta_1 - beta_0)


def beta(t):
    """Calculates the beta term."""
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return (1 + 0.5 * t_tensor * beta_0 + 0.5 * t_tensor ** 2 * (beta_1 - beta_0))


def q_t(data, t):
    """Performs the forward diffusion process."""
    eps = torch.randn_like(data)
    if isinstance(t, (float, int)) or (isinstance(t, torch.Tensor) and t.ndim == 0):
        t = torch.full((data.shape[0], 1), t, device=data.device, dtype=torch.float32)

    log_alpha_t = log_alpha(t).view(-1, 1)
    log_sigma_t = log_sigma(t).view(-1, 1)
    x_t = torch.exp(log_alpha_t) * data + torch.exp(log_sigma_t) * eps
    return eps, x_t

# --- PyTorch Model Definition ---
class MLP(nn.Module):
    def __init__(self, num_hid, num_out):
        super(MLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1 + num_out, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_out)
        )

    def forward(self, t, x):
        h = torch.cat([t, x], dim=1)
        return self.main(h)


# --- Training Function ---
def train_model(data_generator, ndim: int = 2, device='cuda'):
    model = MLP(num_hid=512, num_out=ndim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    num_iterations = 20_000

    model.train()
    for iter in trange(num_iterations, desc=f"Training on {data_generator.keywords['up']} data"):
        optimizer.zero_grad()

        data = data_generator(bs=bs)
        t = torch.rand(bs, 1, device=device) * (t_1 - 0.001) + 0.001

        eps, x_t = q_t(data, t)
        predicted_noise = model(t, x_t)

        loss = ((eps + predicted_noise) ** 2).mean()

        loss.backward()
        optimizer.step()

    return model


# --- Generation Function ---
def generate_samples(model, bs:int = 512, ndim: int = 2, device: str='cuda'):
    dt = 1e-2
    t_end = 1.0
    n = int(t_end / dt)

    x_gen = torch.randn(bs, ndim).to(device)
    x_gen_history = torch.zeros(bs, n + 1, ndim, device=device)
    x_gen_history[:, 0, :] = x_gen

    model.eval()
    with torch.no_grad():
        for i in trange(n, desc="Generating Samples"):
            t = t_end - i * dt
            t_tensor = torch.full((bs, 1), t, device=device)

            # Vector field for the SDE
            predicted_noise = model(t_tensor, x_gen)
            dxdt = dlog_alphadt(t).to(device) * x_gen - 2 * beta(t).to(device) * predicted_noise

            # Euler-Maruyama step
            noise = torch.randn_like(x_gen)
            dx = -dt * dxdt + torch.sqrt(2 * torch.exp(log_sigma(t)).to(device) * beta(t).to(device) * dt) * noise
            x_gen = x_gen + dx
            x_gen_history[:, i + 1, :] = x_gen

    return x_gen_history


def vector_field(model, t, x):
    """
    Computes the score (model output) and the divergence of the score.
    The divergence is computed using the Hutchinson's trace estimator via a JVP.
    """
    x_clone = x.clone().requires_grad_(True)
    t_in = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32)

    # Random vector for JVP
    eps = torch.randn_like(x_clone)

    # Define the function for JVP: model output w.r.t. input x
    model_fn = lambda _x: model(t_in, _x)

    # Compute JVP
    score_val, jvp_val = torch.autograd.functional.jvp(model_fn, x_clone, eps, create_graph=False)

    # Divergence is E[eps^T * J * eps]. We use a single sample.
    divergence = (jvp_val * eps).sum(dim=1, keepdim=True)

    return score_val.detach(), divergence.detach()


def get_dll(t, x, sdlogdx_val, divlog_val, dxdt, device):
    """Calculates the change in log-likelihood (Fokker-Planck equation)."""
    v = dlog_alphadt(t).to(device) * x - beta(t).to(device) * sdlogdx_val
    dlldt = -dlog_alphadt(t).to(device) * ndim + beta(t).to(device) * divlog_val
    dlldt += -((sdlogdx_val / torch.exp(log_sigma(t)).to(device)) * (v - dxdt)).sum(1, keepdim=True)
    return dlldt


def get_kappa(t, divlogs, sdlogdxs, device):
    """Calculates the weighting factor kappa."""
    divlog_1, divlog_2 = divlogs
    sdlogdx_1, sdlogdx_2 = sdlogdxs

    log_sigma_t = log_sigma(t).to(device)

    kappa_num = torch.exp(log_sigma_t) * (divlog_1 - divlog_2) + (sdlogdx_1 * (sdlogdx_1 - sdlogdx_2)).sum(1,
                                                                                                           keepdim=True)
    kappa_den = ((sdlogdx_1 - sdlogdx_2) ** 2).sum(1, keepdim=True)

    # Add a small epsilon to the denominator to avoid division by zero
    kappa = kappa_num / (kappa_den + 1e-9)
    return kappa

# --- Main Execution ---
seed = 0
bs = 512
print(f"Using device: {device}")

torch.manual_seed(seed)
np.random.seed(seed)

# --- Visualize Forward Process ---
t_axis = np.linspace(0.001, 1.0, 6)
plt.figure(figsize=(23, 5))
for i, t_val in enumerate(t_axis):
    plt.subplot(1, len(t_axis), i + 1)
    _, x_t_up = q_t(sample_data(bs // 2, up=True, device=device), t_val)
    _, x_t_down = q_t(sample_data(bs // 2, up=False, device=device), t_val)
    plt.scatter(x_t_up.cpu().numpy()[:, 0], x_t_up.cpu().numpy()[:, 1], alpha=0.3)
    plt.scatter(x_t_down.cpu().numpy()[:, 0], x_t_down.cpu().numpy()[:, 1], alpha=0.3)
    plt.title(f't={t_val:.2f}')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
plt.suptitle("Forward Diffusion of Two Distributions", fontsize=16)
plt.show()

# --- Train Models ---
sample_data_up = partial(sample_data, up=True, device=device)
sample_data_down = partial(sample_data, up=False, device=device)

model_up = train_model(sample_data_up)
model_down = train_model(sample_data_down)

# --- Generate and Visualize Reverse Process ---
x_gen_history = generate_samples(model_up, ndim=ndim, device=device)

plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    t_forward = t_axis[len(t_axis) - 1 - i]

    # Get noised data from forward process
    _, x_t = q_t(sample_data(bs, up=True, device=device), t_forward)

    # Get generated data from reverse process history
    gen_idx = int(x_gen_history.shape[1] * t_axis[i])
    gen_idx = min(max(gen_idx, 0), x_gen_history.shape[1] - 1)
    x_gen_at_t = x_gen_history[:, gen_idx, :].cpu().numpy()

    plt.scatter(x_t.cpu().numpy()[:, 0], x_t.cpu().numpy()[:, 1], label='noise_data', alpha=0.3)
    plt.scatter(x_gen_at_t[:, 0], x_gen_at_t[:, 1], label='gen_data', color='green')
    plt.title(f't={t_forward:.2f}')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
plt.suptitle("Reverse Diffusion using the 'Up' Model", fontsize=16)
plt.show()


# --- Generate and Visualize Reverse Process ---
x_gen_history = generate_samples(model_down, ndim=ndim, device=device)

plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    t_forward = t_axis[len(t_axis) - 1 - i]

    # Get noised data from forward process
    _, x_t = q_t(sample_data(bs, up=False, device=device), t_forward)

    # Get generated data from reverse process history
    gen_idx = int(x_gen_history.shape[1] * t_axis[i])
    gen_idx = min(max(gen_idx, 0), x_gen_history.shape[1] - 1)
    x_gen_at_t = x_gen_history[:, gen_idx, :].cpu().numpy()

    plt.scatter(x_t.cpu().numpy()[:, 0], x_t.cpu().numpy()[:, 1], label='noise_data', alpha=0.3)
    plt.scatter(x_gen_at_t[:, 0], x_gen_at_t[:, 1], label='gen_data', color='green')
    plt.title(f't={t_forward:.2f}')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
plt.suptitle("Reverse Diffusion using the 'Down' Model", fontsize=16)
plt.show()

# --- Combined Generation and Log-Likelihood Calculation ---
dt = 1e-3
t_end = 1.0
n = int(t_end / dt)

x_gen = torch.randn(bs, ndim).to(device)
x_gen_history = torch.zeros(bs, n + 1, ndim, device=device)
x_gen_history[:, 0, :] = x_gen

ll_1 = torch.zeros(bs, n + 1, device=device)
ll_2 = torch.zeros(bs, n + 1, device=device)

model_up.eval()
model_down.eval()
with torch.no_grad():
    for i in trange(n, desc="Generating with Combined Models"):
        t = t_end - i * dt
        x_t = x_gen_history[:, i, :]

        # Calculate scores and divergences for both models
        sdlogdx_1, divdlog_1 = vector_field(model_up, t, x_t)
        sdlogdx_2, divdlog_2 = vector_field(model_down, t, x_t)

        # Calculate kappa and the combined dynamics
        kappa = get_kappa(t, (divdlog_1, divdlog_2), (sdlogdx_1, sdlogdx_2), device)
        dxdt = dlog_alphadt(t).to(device) * x_t - beta(t).to(device) * (sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2))

        # Update sample
        x_gen_history[:, i + 1, :] = x_t - dt * dxdt

        # Update log-likelihoods
        ll_1[:, i + 1] = ll_1[:, i] - dt * get_dll(t, x_t, sdlogdx_1, divdlog_1, dxdt, device).squeeze()
        ll_2[:, i + 1] = ll_2[:, i] - dt * get_dll(t, x_t, sdlogdx_2, divdlog_2, dxdt, device).squeeze()

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot((ll_1 - ll_2).cpu().numpy()[:20, :].T)
plt.title("Log-Likelihood Difference (p1/p2) Over Time")
plt.xlabel("Time Step (reversed)")
plt.ylabel("Log(p1/p2)")
plt.grid()
plt.show()

plt.figure(figsize=(23, 5))
t_axis = np.linspace(0.001, 1.0, 6)
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    t_forward = t_axis[len(t_axis) - 1 - i]

    # Get noised data from forward process for visualization
    _, x_t_up = q_t(sample_data(bs // 2, up=True, device=device), t_forward)
    _, x_t_down = q_t(sample_data(bs // 2, up=False, device=device), t_forward)
    plt.scatter(x_t_up.cpu().numpy()[:, 0], x_t_up.cpu().numpy()[:, 1], label='noise_data_up', alpha=0.3)
    plt.scatter(x_t_down.cpu().numpy()[:, 0], x_t_down.cpu().numpy()[:, 1], label='noise_data_down', alpha=0.3)

    # Get generated data from reverse process history
    gen_idx = int(n * t_axis[i])
    gen_idx = min(max(gen_idx, 0), n)
    x_gen_at_t = x_gen_history[:, gen_idx, :].cpu().numpy()
    plt.scatter(x_gen_at_t[:, 0], x_gen_at_t[:, 1], label='gen_data', color='green', alpha=0.5)

    plt.title(f't={t_forward:.2f}')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
plt.suptitle("Reverse Diffusion using Combined Models", fontsize=16)
plt.show()