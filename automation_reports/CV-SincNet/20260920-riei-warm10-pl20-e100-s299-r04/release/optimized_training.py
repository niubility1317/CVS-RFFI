"""Source-only online channel augmentation and detached clean-to-SG supervision."""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def augment_pairs(pairs, seed, epoch, step, role, curriculum_epochs=50):
    if pairs.ndim != 4 or tuple(pairs.shape[1:]) != (2, 2, 2048):
        raise ValueError('Online augmentation requires paired (B,2,2,2048) input')
    generator = torch.Generator(device=pairs.device)
    # No draws from model/global RNG. Same realization for matched seed/epoch/step/role.
    draw_seed = int(seed) * 100000000 + int(epoch) * 100000 + int(step) * 2 + int(role)
    generator.manual_seed(draw_seed)
    clean = pairs[:, 0]
    wave = torch.complex(clean[:, 0], clean[:, 1])
    raw = torch.stack([wave[:, :256], wave[:, 512:768]], dim=1).reshape(-1, 256)
    n, length = raw.shape
    random = torch.rand((n, 2), generator=generator, device=pairs.device)
    rho = torch.where(random[:, :1] < .5, torch.zeros_like(random[:, :1]), .8 + .19 * random[:, 1:])
    innovations = torch.randn((n, length), generator=generator, device=pairs.device)
    powers = rho ** torch.arange(length, device=pairs.device)[None, :]
    impulses = innovations.clone()
    impulses[:, 0] = 0
    # Stationary AR(1), computed as causal convolution; zero-pad to avoid wraparound.
    conv = torch.fft.irfft(torch.fft.rfft(impulses, n=512) * torch.fft.rfft(powers, n=512), n=512)[:, :length]
    gaussian = powers * innovations[:, :1] + torch.sqrt(1 - rho.square()) * conv
    progress = min(1., max(0., (epoch - 1) / max(1, curriculum_epochs - 1)))
    sigma = 1. + 2. * progress
    los = torch.exp(sigma * gaussian)
    scatter = torch.randn((n, length, 2), generator=generator, device=pairs.device) * math.sqrt(.0049 / 2.)
    gain = torch.complex(los + scatter[..., 0], scatter[..., 1])
    sg_raw = raw * gain
    input_power = raw.abs().square().mean(1, keepdim=True)
    output_power = sg_raw.abs().square().mean(1, keepdim=True)
    sg_raw = sg_raw * torch.sqrt((input_power + 1e-12) / (output_power + 1e-12))
    padded = F.pad(sg_raw, (0, 256)).reshape(-1, 2, 512)
    freq = torch.fft.fft(padded, dim=-1) / math.sqrt(512.)
    representation = torch.cat([padded[:, 0], padded[:, 1], freq[:, 0], freq[:, 1]], dim=1)
    sg = torch.stack([representation.real, representation.imag], dim=1)
    if not torch.isfinite(sg).all():
        raise FloatingPointError('Non-finite online channel realization')
    result = torch.stack([clean, sg], dim=1)
    power = sg_raw.abs().square()
    energy = power.sum(1)
    diagnostics = {'mode': 'dynamic', 'draw_seed': draw_seed, 'sigma': sigma,
                   'iid_fraction': float((rho == 0).float().mean().item()),
                   'rho_mean': float(rho.mean().item()),
                   'energy_effective_samples_mean': float((energy.square() / power.square().sum(1).clamp_min(1e-30)).mean().item()),
                   'top1_energy_fraction_mean': float((power.max(1).values / energy.clamp_min(1e-30)).mean().item()),
                   'rms_ratio_mean': float(torch.sqrt(power.mean(1) / input_power[:, 0].clamp_min(1e-30)).mean().item()),
                   'clean_max_abs_change': float((result[:, 0] - clean).abs().max().item())}
    return result, diagnostics


def pseudo_objective(logits, policy, threshold, active):
    """Never accepts ground-truth labels. Masks and CE targets are detached."""
    if logits.shape[0] % 2:
        raise ValueError('Clean/SG logits must be interleaved pairs')
    probabilities = logits.detach().softmax(1)
    confidence, predictions = probabilities.max(1)
    self_mask = confidence >= threshold
    if policy == 'self':
        candidate = self_mask
        pseudo = predictions
        teacher_conf = confidence
        denominator = self_mask.sum().clamp_min(1)
    elif policy == 'clean_only':
        candidate = self_mask.clone()
        candidate[1::2] = False
        pseudo = predictions
        teacher_conf = confidence
        # Drop SG numerator while keeping original selected-view denominator.
        # Thus the clean gradient contribution is unchanged for the same logits.
        denominator = self_mask.sum().clamp_min(1)
    elif policy == 'clean_to_sg':
        pseudo = predictions[0::2].repeat_interleave(2)
        teacher_conf = confidence[0::2].repeat_interleave(2)
        candidate = (confidence[0::2] >= threshold).repeat_interleave(2)
        denominator = candidate.sum().clamp_min(1)
    else:
        raise ValueError(policy)
    used = candidate & bool(active)
    loss = (F.cross_entropy(logits, pseudo, reduction='none') * used).sum() / denominator
    meta = {'labels': pseudo, 'teacher_confidence': teacher_conf, 'candidate': candidate,
            'used': used, 'self_candidate': self_mask, 'denominator': int(denominator.item()),
            'policy': policy}
    return loss, meta
