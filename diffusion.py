import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        # # 确保梯度跟踪
        # x_t.requires_grad_()
        # noise.requires_grad_()
        # # 直接去噪
        # pred_x0 = (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise) / extract(self.sqrt_alphas_bar, t, x_0.shape)
        #
        # # 损失计算：对比去噪后的 x_0 和真实的 x_0
        # loss = F.mse_loss(pred_x0, x_0, reduction='none')

        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class GaussianDiffusionSamplerDDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta=0.):
        super().__init__()

        self.model = model
        self.T = T
        self.eta = eta  # DDIM 参数，控制随机性

        # 初始化 beta 和相关参数
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        使用预测的噪声 eps 来计算 t 时刻的均值。
        """
        alpha_t = extract(self.alphas_bar, t, x_t.shape)
        alpha_prev = extract(self.alphas_bar_prev, t, x_t.shape)

        # DDIM 的去噪公式
        return (
            torch.sqrt(alpha_prev) * (
                (x_t - eps * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)
            )
        )

    def forward(self, x_T):
        """
        DDIM 采样过程：支持确定性（eta=0）或半随机采样（eta>0）
        """
        x_t = x_T

        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

            # 提取 alpha 和 beta
            alpha_t = extract(self.alphas_bar, t, x_t.shape)
            alpha_prev = extract(self.alphas_bar_prev, t, x_t.shape)
            beta_t = extract(self.betas, t, x_t.shape)

            # 模型预测噪声
            eps = self.model(x_t, t)

            # 根据 eps 预测 x0 和 xt-1
            x0_pred = (x_t - eps * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)
            mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)

            # 计算噪声项
            if time_step > 0:
                sigma_t = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * beta_t)
                noise = torch.randn_like(x_t) if self.eta > 0 else 0
                x_t = mean + sigma_t * noise
            else:
                x_t = mean  # 最后一步无噪声

        return torch.clip(x_t, -1, 1)
