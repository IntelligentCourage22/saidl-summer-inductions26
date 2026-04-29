import torch
import torch.nn.functional as F


def extract(values, timesteps, shape):
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (
            1.0 - torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        ) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        return GaussianDiffusion(
            self.num_timesteps,
            float(self.betas[0].detach().cpu()),
            float(self.betas[-1].detach().cpu()),
            device,
        )

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x_t, timesteps, eps):
        return (
            x_t - extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape) * eps
        ) / extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)

    def training_loss(self, model, x_start):
        timesteps = torch.randint(
            0, self.num_timesteps, (x_start.shape[0],), device=x_start.device
        )
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        pred = model(x_t, timesteps)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, timesteps):
        pred_noise = model(x_t, timesteps)
        return self.p_sample_from_pred_noise(pred_noise, x_t, timesteps)

    @torch.no_grad()
    def p_sample_from_pred_noise(self, pred_noise, x_t, timesteps):
        beta_t = extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, timesteps, x_t.shape)
        model_mean = sqrt_recip_alpha * (x_t - beta_t * pred_noise / sqrt_one_minus)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.dim() - 1)))
        variance = extract(self.posterior_variance, timesteps, x_t.shape)
        return model_mean + nonzero_mask * torch.sqrt(variance.clamp(min=1e-20)) * noise

    @torch.no_grad()
    def ddim_step_from_pred_noise(self, pred_noise, x_t, timestep, prev_timestep):
        bsz = x_t.shape[0]
        t = torch.full((bsz,), int(timestep), device=x_t.device, dtype=torch.long)
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        x0 = self.predict_x0_from_eps(x_t, t, pred_noise)
        if prev_timestep < 0:
            alpha_prev = torch.ones_like(alpha_t)
        else:
            prev = torch.full((bsz,), int(prev_timestep), device=x_t.device, dtype=torch.long)
            alpha_prev = extract(self.alphas_cumprod, prev, x_t.shape)
        direction = torch.sqrt((1.0 - alpha_prev).clamp(min=0.0)) * pred_noise
        return torch.sqrt(alpha_prev) * x0 + direction

    @torch.no_grad()
    def add_noise_to_timestep(self, x_start, t_start, mask=None):
        timesteps = torch.full((x_start.shape[0],), t_start, device=x_start.device, dtype=torch.long)
        noised = self.q_sample(x_start, timesteps)
        if mask is None:
            return noised
        return mask * noised + (1 - mask) * x_start
