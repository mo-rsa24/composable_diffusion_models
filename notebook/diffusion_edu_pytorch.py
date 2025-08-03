import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def sample_data(bs):
  """
  Generates sample data based on a uniform distribution over 4 cluster centers.

  Args:
    bs (int): The batch size.

  Returns:
    torch.Tensor: A tensor of shape (bs, 2) representing the sampled data.
  """
  x_1 = torch.randint(0, 2, (bs, 2)).float()
  x_1 = 3 * (x_1 - 0.5)
  x_1 += 4e-1 * torch.randn_like(x_1)
  return x_1

def log_alpha(t, beta_0=0.1, beta_1 = 20.0):
    """
    Calculates the log of the alpha term in the diffusion process.

    Args:
    t (torch.Tensor or float): The time step.

    Returns:
    torch.Tensor: The log of alpha_t.
    """
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * t_tensor * beta_0 - 0.25 * t_tensor ** 2 * (beta_1 - beta_0)

def log_sigma(t):
  """
  Calculates the log of the sigma term in the diffusion process.

  Args:
    t (torch.Tensor or float): The time step.

  Returns:
    torch.Tensor: The log of sigma_t.
  """
  # Ensure t is a tensor for torch.log
  t_tensor = torch.as_tensor(t, dtype=torch.float32)
  return torch.log(t_tensor)


def dlog_alphadt(t):
    """
    Analytical derivative of log_alpha with respect to t.
    This is used during sampling, which happens in a no_grad context,
    so we use the analytical formula instead of autograd.
    """
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return -0.5 * beta_0 - 0.5 * t_tensor * (beta_1 - beta_0)


def beta(t, beta_0=0.1, beta_1=20.0):
    """
    Calculates the beta term, which is related to the diffusion schedule.
    """
    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    return (1 + 0.5 * t_tensor * beta_0 + 0.5 * t_tensor ** 2 * (beta_1 - beta_0))


def q_t(data, t):
    """
    Performs the forward diffusion process (q_t) to add noise to the data.

    Args:
      data (torch.Tensor): The input data tensor.
      t (float or torch.Tensor): The time step.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: A tuple containing the noise and the noisy data.
    """
    eps = torch.randn_like(data)
    # Ensure t can be broadcasted if it's a scalar
    if isinstance(t, (float, int)) or (isinstance(t, torch.Tensor) and t.ndim == 0):
        t = torch.full((data.shape[0], 1), t, device=data.device, dtype=torch.float32)

    log_alpha_t = log_alpha(t).view(-1, 1)
    log_sigma_t = log_sigma(t).view(-1, 1)
    x_t = torch.exp(log_alpha_t) * data + torch.exp(log_sigma_t) * eps
    return eps, x_t

def visualize_process(bs=512):
    t_axis = np.linspace(0.001, 1.0, 6)  # Start from a small non-zero value for log_sigma
    plt.figure(figsize=(23, 5))
    for i, t_val in enumerate(t_axis):
        plt.subplot(1, len(t_axis), i + 1)

        # Generate data and apply the forward process
        data_sample = sample_data(bs)
        _, x_t = q_t(data_sample, t_val)

        # Scatter plot the results
        plt.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), alpha=0.3)
        plt.title(f't={t_val:.2f}')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True)

    plt.suptitle("Forward Diffusion Process at Different Timesteps (PyTorch)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


class MLP(nn.Module):
    def __init__(self, num_hid, num_out):
        super(MLP, self).__init__()
        self.num_hid = num_hid
        self.num_out = num_out

        self.main = nn.Sequential(
            nn.Linear(1 + num_out, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),  # Swish activation
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),  # Swish activation
            nn.Linear(num_hid, num_out)
        )

    def forward(self, t, x):
        # Concatenate time and data tensors
        h = torch.cat([t, x], dim=1)
        return self.main(h)

def train(model, optimizer, num_iterations=20_000, device='cuda'):
    loss_plot = np.zeros(num_iterations)
    model.train()
    for iter in trange(num_iterations):
        optimizer.zero_grad()

        # Sample data and time
        data = sample_data(bs).to(device)
        t = torch.rand(bs, 1, device=device) * (t_1 - 0.001) + 0.001  # Sample t from [0.001, 1.0]

        # Forward diffusion process
        eps, x_t = q_t(data, t)

        # Predict noise using the model
        predicted_noise = model(t, x_t)

        # Calculate loss
        loss = ((eps + predicted_noise) ** 2).mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss_plot[iter] = loss.item()
    return loss_plot

def visualize_train_plot(loss_plot):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_plot)
    plt.title("Training Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

# --- Visualize Forward Process (optional, from before) ---
def visualize_forward_process(bs=512):
  t_axis = np.linspace(0.001, 1.0, 6)
  plt.figure(figsize=(23, 5))
  for i, t_val in enumerate(t_axis):
    plt.subplot(1, len(t_axis), i + 1)

    data_sample = sample_data(bs)
    _, x_t = q_t(data_sample, t_val)

    plt.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), alpha=0.3)
    plt.title(f't={t_val:.2f}')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)

  plt.suptitle("Forward Diffusion Process at Different Timesteps (PyTorch)", fontsize=16)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()


def vector_field(t, x, xi=0.0):
    """
    Computes the vector field for the reverse SDE.
    v_t(x) = dlog(alpha)/dt * x - beta(t) * model(t, x) - ...
    """
    # Ensure t is correctly shaped for the model
    if isinstance(t, (float, int)) or (isinstance(t, torch.Tensor) and t.ndim == 0):
        t_in = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float32)
    else:
        t_in = t

    predicted_noise = model(t_in, x)

    # Move scalar tensors to the correct device before operations
    dlog_alphadt_t = dlog_alphadt(t).to(x.device)
    beta_t = beta(t).to(x.device)
    log_sigma_t = log_sigma(t).to(x.device)

    dxdt = dlog_alphadt_t * x - beta_t * predicted_noise - xi * beta_t / torch.exp(log_sigma_t) * predicted_noise
    return dxdt

def get_samples(sample_x, model, xi = 1.0, dt = 1e-2, t_end = 1.0,  bs=512, n:int= int(1 / 1e-2), device='cuda'):
    x_gen = torch.randn(bs, sample_x.shape[1], device=device)
    x_gen_history = torch.zeros(bs, n + 1, sample_x.shape[1], device=device)
    x_gen_history[:, 0, :] = x_gen
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i in trange(n, desc="Generating Samples"):
            t = t_end - i * dt
            t_tensor = torch.full((bs, 1), t, device=device)

            # Euler-Maruyama step
            dx = -dt * vector_field(t, x_gen, xi) + torch.sqrt(2 * xi * beta(t) * dt) * torch.randn_like(x_gen)
            x_gen = x_gen + dx
            x_gen_history[:, i + 1, :] = x_gen
    return x_gen, x_gen_history

def visualize_reverse_process(x_gen_history, bs=512, device='cuda', n=int(1.0/1e-2)):
    # --- Visualize Reverse Process ---
    t_axis = np.linspace(0.001, 1.0, 6)
    plt.figure(figsize=(23, 5))

    for i in range(len(t_axis)):
        plt.subplot(1, len(t_axis), i + 1)

        # Time for forward process
        t_forward = t_axis[len(t_axis) - 1 - i]

        # Get noised data from forward process q_t
        _, x_t = q_t(sample_data(bs).to(device), t_forward)

        # Get generated data from reverse process history
        # This matches the logic from the original JAX code
        gen_idx = int(n * t_axis[i])
        x_gen_at_t = x_gen_history[:, gen_idx, :].cpu().numpy()

        plt.scatter(x_t.cpu().numpy()[:, 0], x_t.cpu().numpy()[:, 1], label='noise_data', alpha=0.3)
        plt.scatter(x_gen_at_t[:, 0], x_gen_at_t[:, 1], label='gen_data', alpha=0.5)

        plt.title(f't={t_forward:.2f}')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid(True)
        if i == 0:
            plt.legend(fontsize=15)

    plt.suptitle("Reverse Diffusion Process at Different Timesteps (PyTorch)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Hyperparameters
seed = 0
bs = 512
learning_rate = 2e-4
num_iterations = 20_000


# Time and beta parameters for the diffusion process
t_0, t_1 = 0.0, 1.0
beta_0 = 0.1
beta_1 = 20.0


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize model and optimizer
# We need to know the output dimension, let's get it from a sample
sample_x = sample_data(1)
model = MLP(num_hid=512, num_out=sample_x.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Model Architecture:")
print(model)

# --- Training Loop ---
visualize_process(bs)
loss_plot = train(model, optimizer, num_iterations=20_000, device='cuda')
visualize_train_plot(loss_plot)
visualize_forward_process(bs)

# Generation parameters
dt = 1e-2
xi = 1.0
t_end = 1.0
n = int(t_end / dt)

# Initial noise
x_gen = torch.randn(bs, sample_x.shape[1], device=device)
x_gen_history = torch.zeros(bs, n + 1, sample_x.shape[1], device=device)
x_gen_history[:, 0, :] = x_gen

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for i in trange(n, desc="Generating Samples"):
        t = t_end - i * dt

        # Euler-Maruyama step
        beta_t = beta(t).to(device)
        dx = -dt * vector_field(t, x_gen, xi) + torch.sqrt(2 * xi * beta_t * dt) * torch.randn_like(x_gen)
        x_gen = x_gen + dx
        x_gen_history[:, i + 1, :] = x_gen

# --- Visualize Reverse Process ---
t_axis = np.linspace(0.001, 1.0, 6)
plt.figure(figsize=(23, 5))

for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)

    # Time for forward process
    t_forward = t_axis[len(t_axis) - 1 - i]

    # Get noised data from forward process q_t
    _, x_t = q_t(sample_data(bs).to(device), t_forward)

    # Get generated data from reverse process history
    # This matches the logic from the original JAX code
    # We step backwards in time during generation, so to get the state at t,
    # we look at step n*(1-t)
    gen_idx = int(n * (1.0 - t_forward))
    # Ensure index is within bounds
    gen_idx = min(max(gen_idx, 0), n)
    x_gen_at_t = x_gen_history[:, gen_idx, :].cpu().numpy()

    plt.scatter(x_t.cpu().numpy()[:, 0], x_t.cpu().numpy()[:, 1], label='noise_data', alpha=0.3)
    plt.scatter(x_gen_at_t[:, 0], x_gen_at_t[:, 1], label='gen_data', alpha=0.5)

    plt.title(f't={t_forward:.2f}')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    if i == 0:
        plt.legend(fontsize=15)

plt.suptitle("Reverse Diffusion Process at Different Timesteps (PyTorch)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#
# x_gen, x_gen_history = get_samples(sample_x, model, xi = xi, dt = dt, t_end = t_end,  bs=bs, n=n, device='cuda')
# visualize_reverse_process(x_gen_history, bs=bs, device='cuda', n=n)
