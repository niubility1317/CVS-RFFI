"""Source-only paired clean-versus-restored discrimination, without TX or F."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip

WEIGHT = .01
SEED = 2026110715
OBJECTIVE = 'wide_conditional_adversary'
CRITIC_KIND = 'paired_iq_hinge_spectral_conv'


class ConditionalIQCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([spectral_norm(nn.Conv1d(7, 16, 7, stride=2, padding=3)),
            spectral_norm(nn.Conv1d(16, 32, 5, stride=2, padding=2)),
            spectral_norm(nn.Conv1d(32, 32, 5, stride=2, padding=2))])
        self.head = nn.Conv1d(32, 1, 1)

    def forward(self, received, anchor, candidate, variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or candidate.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching paired source IQ required')
        scale = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        noise = torch.log1p(variance.clamp_min(0.)[:, None, None]/scale.square())
        h = torch.cat((received/scale, anchor/scale, candidate/scale, noise.expand(-1, 1, 256)), 1)
        for layer in self.layers:
            h = F.leaky_relu(layer(h), .2)
        return self.head(h).mean((1, 2))


def update_critic(critic, optimizer, received, anchor, restored, variance, clean):
    """One critic-only update, then a differentiable frozen-critic generator loss."""
    critic.train().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    real = critic(received.detach(), anchor.detach(), clean.detach(), variance.detach())
    fake = critic(received.detach(), anchor.detach(), restored.detach(), variance.detach())
    critic_loss = F.relu(1-real).mean()+F.relu(1+fake).mean()
    checked_backward_and_clip(critic_loss, critic, (critic.head.weight,))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    critic.eval().requires_grad_(False)
    adversarial = -critic(received.detach(), anchor.detach(), restored, variance.detach()).mean()
    return critic_loss.detach(), adversarial
