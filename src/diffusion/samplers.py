import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

class SuperDiffSampler:
    """Implements the SUPERDIFF algorithm for composing two pre-trained models."""

    def __init__(self, sde):
        self.sde = sde

    @torch.no_grad()
    def sample(self, model1, model2, batch_size, shape, device, operation='OR', temp=1.0, bias=0.0):
        model1.eval()
        model2.eval()
        x = torch.randn((batch_size, *shape), device=device)
        log_q1 = torch.zeros(batch_size, device=device)
        log_q2 = torch.zeros(batch_size, device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc=f"SUPERDIFF Sampling ({operation})", leave=False):
            t_idx = timesteps[i]
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise1, noise2 = model1(x, t.float()), model2(x, t.float())
            score1, score2 = -noise1 / sqrt_one_minus_alpha_bar_t, -noise2 / sqrt_one_minus_alpha_bar_t
            if operation.upper() == 'OR':
                logits = torch.stack([log_q1, log_q2], dim=1)
                kappas = F.softmax(temp * logits + bias, dim=1)
                kappa1, kappa2 = kappas[:, 0].view(-1, 1, 1, 1), kappas[:, 1].view(-1, 1, 1, 1)
            elif operation.upper() == 'AND':
                # Heuristic to balance log-densities, pushing towards an equal density state.
                # A rigorous implementation solves the linear system in Prop. 6 of the paper [cite: 207-210].
                probs = F.softmax(torch.stack([-log_q1, -log_q2], dim=1), dim=1)
                kappa1, kappa2 = probs[:, 0].view(-1, 1, 1, 1), probs[:, 1].view(-1, 1, 1, 1)
            else:
                kappa1, kappa2 = 0.5, 0.5
            combined_score = kappa1 * score1 + kappa2 * score2
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            mean = (1 / sqrt_alpha_t) * (x + beta_t * combined_score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            dx = x_prev - x
            dtau = 1.0 / self.sde.num_timesteps
            d = x.shape[1] * x.shape[2] * x.shape[3]
            div_f = -0.5 * beta_t.squeeze() * d

            def update_log_q(log_q, score):
                term1 = torch.sum(dx * score, dim=[1, 2, 3])
                f_term = -0.5 * beta_t * x
                g_sq_term = beta_t
                inner_prod_term = torch.sum((f_term - 0.5 * g_sq_term * score) * score, dim=[1, 2, 3])
                return log_q + term1 + (div_f + inner_prod_term) * dtau

            log_q1, log_q2 = update_log_q(log_q1, score1), update_log_q(log_q2, score2)
            x = x_prev
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_single_model(self, model, batch_size, shape, device):
        model.eval()
        x = torch.randn((batch_size, *shape), device=device)
        timesteps = torch.arange(self.sde.num_timesteps - 1, -1, -1, device=device)
        for i in tqdm(range(self.sde.num_timesteps), desc="Single Model Sampling", leave=False):
            t_idx = timesteps[i]
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            beta_t = self.sde.betas[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(self.sde.alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sde.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = model(x, t.float())
            score = -noise / sqrt_one_minus_alpha_bar_t
            mean = (1 / sqrt_alpha_t) * (x + beta_t * score)
            if i < self.sde.num_timesteps - 1:
                posterior_variance = self.sde.posterior_variance[t].view(-1, 1, 1, 1)
                x_prev = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            else:
                x_prev = mean
            x = x_prev
        return x.clamp(-1, 1)

