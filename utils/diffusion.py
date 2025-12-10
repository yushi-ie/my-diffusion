import torch
import torch.nn.functional as F


def make_beta_schedule(T: int, start: float = 1e-4, end: float = 0.02):
    """Create a linear beta schedule (DDPM Appendix B)."""
    return torch.linspace(start, end, T)


class Diffusion:
    """
    Minimal DDPM helper.
    - Forward:   x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps
    - Reverse:   predict eps, compute mean of p(x_{t-1}|x_t), add noise if t>0
    """

    def __init__(self, T: int = 1000, device: str = "cuda"):
        self.device = device
        self.T = T
        betas = make_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """Forward noising: sample x_t from q(x_t|x0)."""
        noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def p_sample(self, model, x_t: torch.Tensor, cond_img: torch.Tensor, t: int):
        """
        One reverse step x_t -> x_{t-1} using predicted noise.
        - model: UNet2D
        - t: int (0-indexed)
        """
        betas_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

        noise_level = torch.full((x_t.shape[0],), torch.sqrt(alpha_bar_t), device=x_t.device)
        eps_hat = model(x_t, cond_img, noise_level)

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = betas_t / sqrt_one_minus_ab
        mean = coef1 * (x_t - coef2 * eps_hat)
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(betas_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        cond_img: torch.Tensor,
        shape,
        start_x: torch.Tensor | None = None,
        start_t: int | None = None,
    ):
        """
        Run reverse diffusion to obtain x_0.
        - If start_x/start_t are provided, start from a noised source image x_t.
        """
        b = cond_img.size(0)
        if start_x is None:
            x_t = torch.randn(b, *shape, device=cond_img.device)
            t_start = self.T - 1
        else:
            x_t = start_x
            t_start = self.T - 1 if start_t is None else int(start_t)
        for t in reversed(range(t_start + 1)):
            x_t = self.p_sample(model, x_t, cond_img, t)
        return x_t


def noise_prediction_loss(model, diffusion: Diffusion, x0: torch.Tensor, cond_img: torch.Tensor):
    """
    MSE between predicted noise and true noise at a random timestep t.
    """
    b = x0.size(0)
    t = torch.randint(0, diffusion.T, (b,), device=x0.device)
    x_t, noise = diffusion.q_sample(x0, t)
    noise_level = torch.sqrt(diffusion.alpha_bar[t])
    eps_hat = model(x_t, cond_img, noise_level)
    return F.mse_loss(eps_hat, noise)
