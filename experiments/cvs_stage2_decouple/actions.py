"""Small, independent state-to-physical-action networks for fixed-parent fitting.

Only source L IQ and its labels enter preparation.  In identity training the
artifact is frozen and contains no identity model, virtual feature predictor,
EMA teacher, receiver identifier, or online fit operation.
"""
from contextlib import contextmanager
import math
import time

import torch
from torch import nn

from experiments.cvs_multi_action_risk.actions import signal_state
from experiments.cvs_multi_action_risk.physics import (
    apply_action, phase_process, PHASE_LIMIT_RAD, CFO_LIMIT_HZ,
    PHASE_NOISE_LIMIT_RAD,
)


KINDS = ("linear", "temporal")
STATE_DIM = 40
CODE_DIM = 8
HIDDEN = 48
FORMAT_VERSION = 1
DELAY_LIMIT_SAMPLES = .25
DELAY_PAD_SAMPLES = 256


def action_contract():
    return dict(
        format_version=FORMAT_VERSION, input_state_dim=STATE_DIM,
        random_code_dim=CODE_DIM, hidden=HIDDEN, network="LayerNorm(state)+code; 48-SiLU-48-SiLU-4",
        inputs_excluded=["identity_embedding", "label", "RX", "day", "student_loss"],
        objective="maximize fixed-parent true-IQ logit-margin drop; fixed classifier scale 30",
        linear=dict(delays_samples=[1, 3], complex_tap_component_limit=.06,
                    boundary="causal zero padding", rms_renormalization=False),
        temporal=dict(phase_limit_rad=PHASE_LIMIT_RAD, cfo_limit_hz=CFO_LIMIT_HZ,
                      phase_noise_limit_rad=PHASE_NOISE_LIMIT_RAD,
                      curvature_hz_s=0., phase_process="three smooth sine modes; clipped Gaussian coefficients",
                      parameter_schema=["phase_rad", "CFO_Hz", "phase_noise_rad", "curvature_Hz_s", "delay_samples"]),
        fractional_delay=dict(enabled=True, limit_samples=DELAY_LIMIT_SAMPLES,
                              padding_each_side_samples=DELAY_PAD_SAMPLES,
                              operator="zero-pad 256 each side, FFT phase shift, crop original central 256",
                              zero_delay="exact value identity with differentiable delay path",
                              boundary_assumption="IQ outside observed 256-sample window is zero; sinc interpolation may ring near truncated edges",
                              scope="new fixed mild timing perturbation, not satellite/channel simulation or calibrated source error"),
        random="explicit private CPU generators; no global RNG consumption",
        fit=dict(passes_per_network=2, batch_size=128, optimizer="AdamW", lr=2e-4, weight_decay=1e-4),
        online_fit=False, frozen_during_identity_training=True,
    )


def _generator(value):
    if isinstance(value, int):
        return torch.Generator(device="cpu").manual_seed(value)
    if not isinstance(value, torch.Generator) or str(value.device) != "cpu":
        raise ValueError("Supply an integer seed or explicit private CPU torch.Generator")
    return value


def logit_margin(logits, labels):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Parent must return [batch, classes] fixed-scale logits")
    if labels.shape != (len(logits),) or labels.dtype != torch.long:
        raise ValueError("Visible L labels must be a long vector")
    actual = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf)
    return actual - other.max(1).values


class PhysicalActionNet(nn.Module):
    def __init__(self, kind, hidden=HIDDEN):
        super().__init__()
        if kind not in KINDS:
            raise ValueError("Unknown L/T action kind")
        self.kind = kind
        self.state_norm = nn.LayerNorm(STATE_DIM)
        self.net = nn.Sequential(nn.Linear(STATE_DIM + CODE_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, state, code):
        if state.shape != (len(code), STATE_DIM) or code.shape != (len(state), CODE_DIM):
            raise ValueError("Action inputs must be physical [B,40] state and random [B,8] code")
        raw = self.net(torch.cat((self.state_norm(state), code), 1))
        if self.kind == "linear":
            return raw.tanh()
        return torch.stack((PHASE_LIMIT_RAD * raw[:, 0].tanh(),
                            CFO_LIMIT_HZ * raw[:, 1].tanh(),
                            PHASE_NOISE_LIMIT_RAD * raw[:, 2].sigmoid(),
                            torch.zeros_like(raw[:, 3]),
                            DELAY_LIMIT_SAMPLES * raw[:, 3].tanh()), 1)


def _new_net(kind, generator, device):
    # nn.Linear's constructor uses global CPU RNG; isolate it, then initialize
    # every trainable element from the declared private generator explicitly.
    with torch.random.fork_rng(devices=[]):
        net = PhysicalActionNet(kind)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Linear):
                bound = math.sqrt(6. / (module.in_features + module.out_features))
                module.weight.uniform_(-bound, bound, generator=generator)
                module.bias.zero_()
        net.net[-1].weight.mul_(.1)
    return net.to(device)


def _code(batch, generator, like):
    return (2 * torch.rand(batch, CODE_DIM, generator=generator) - 1).to(like)


def fractional_delay(x, delay_samples):
    """Differentiable local timing error with explicit zero extension.

    Positive delay implements y[n] = x[n-delay]. Padding makes the circular
    FFT boundary 256 zero samples away from both observed endpoints; it does
    not claim knowledge of unobserved IQ or absence of truncation ringing.
    """
    if tuple(x.shape[1:]) != (2, 256) or delay_samples.shape != (len(x),):
        raise ValueError("Fractional delay requires IQ [B,2,256] and delay [B]")
    z = torch.complex(x[:, 0], x[:, 1])
    padded = torch.nn.functional.pad(z, (DELAY_PAD_SAMPLES, DELAY_PAD_SAMPLES))
    frequencies = torch.fft.fftfreq(padded.shape[1], device=x.device, dtype=x.dtype)
    angle = -2 * math.pi * delay_samples.to(x)[:, None] * frequencies[None]
    shifted = torch.fft.ifft(torch.fft.fft(padded) * torch.complex(angle.cos(), angle.sin()))
    shifted = shifted[:, DELAY_PAD_SAMPLES:DELAY_PAD_SAMPLES + 256]
    result = torch.stack((shifted.real, shifted.imag), 1)
    # Remove FFT roundoff at exactly zero while retaining the derivative of
    # the true interpolation operation with respect to delay and input IQ.
    exact_zero = result + (x - result).detach()
    return torch.where((delay_samples == 0)[:, None, None], exact_zero, result)


def apply_physical_action(x, parameters, kind, noise=None):
    if kind == "linear":
        return apply_action(x, parameters, kind)
    if kind != "temporal" or parameters.shape != (len(x), 5):
        raise ValueError("T requires phase/CFO/noise/zero-curvature/delay parameters")
    return apply_action(fractional_delay(x, parameters[:, 4]), parameters[:, :4], kind, noise)


def _apply(net, x, state, generator):
    p = net(state, _code(len(x), generator, x))
    noise = phase_process(len(x), generator, x) if net.kind == "temporal" else None
    return apply_physical_action(x, p, net.kind, noise), p


def _parameter_summary(parameters, kind):
    p = parameters.detach().float()
    if kind == "linear":
        valid = bool((p.abs() <= 1.000001).all())
        physical = .06 * p
    else:
        limits = p.new_tensor([PHASE_LIMIT_RAD, CFO_LIMIT_HZ, PHASE_NOISE_LIMIT_RAD, 0., DELAY_LIMIT_SAMPLES])
        valid = bool((p.abs() <= limits + 1e-6).all() and (p[:, 2] >= 0).all())
        physical = p
    return dict(kind=kind, samples=len(p), finite=bool(torch.isfinite(p).all()),
                within_registered_bounds=valid,
                parameter_min=p.min(0).values.cpu().tolist(),
                parameter_max=p.max(0).values.cpu().tolist(),
                physical_parameter_abs_mean=physical.abs().mean(0).cpu().tolist())


@contextmanager
def _frozen_reference(parent):
    flags = [(parameter, parameter.requires_grad) for parameter in parent.parameters()]
    modes = [(module, module.training) for module in parent.modules()]
    try:
        parent.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training


def _validate_cache(cache):
    x, y, state = cache["x"], cache["y"], cache["state"]
    if x.device.type != "cpu" or tuple(x.shape[1:]) != (2, 256) or len(x) == 0:
        raise ValueError("Action preparation requires nonempty CPU source L IQ [N,2,256]")
    if y.device.type != "cpu" or y.shape != (len(x),) or y.dtype != torch.long:
        raise ValueError("Action preparation requires visible source L long labels on CPU")
    if state.device.type != "cpu" or state.shape != (len(x), STATE_DIM):
        raise ValueError("Action preparation requires cached CPU physical state [N,40]")
    ids = list(cache["ids"])
    if len(ids) != len(x) or len(set(ids)) != len(x):
        raise ValueError("Action cache requires one unique physical ID per source L packet")
    if not bool(torch.isfinite(x).all() and torch.isfinite(state).all()):
        raise ValueError("Nonfinite source L IQ/state")
    return x, y, state


def fit_actions(parent, cache, seed, device, passes=2, batch_size=128):
    """Fit L and T separately on exactly ``passes`` complete source L traversals.

    The caller has strictly loaded the fixed same-seed parent before this call.
    A cached ``u0`` (aliases: ``parent_margin``, ``margin``) avoids all clean
    reference forwards here.  Only a missing margin cache triggers one clean
    pass.  Parent parameters/buffers/modes/gradient flags are preserved.
    """
    x, y, states = _validate_cache(cache)
    if not isinstance(passes, int) or passes < 1 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive integer passes and batch_size required")
    if any(parameter.device != torch.device(device) for parameter in parent.parameters()):
        # torch.device('cuda') has no explicit index; compare actual location.
        desired = torch.empty(0, device=device).device
        if any(parameter.device != desired for parameter in parent.parameters()):
            raise ValueError("Caller must place the fixed parent on the fitting device")
    started = time.perf_counter()
    parent_before = {name: value.detach().cpu().clone() for name, value in parent.state_dict().items()}
    diagnostics = dict(seed=int(seed), source_packets=len(x), unique_physical_ids=len(cache["ids"]),
                       parent_clean_forward_batches=0, parent_augmented_forward_batches=0,
                       passes_per_network=passes, batch_size=batch_size, branches={})
    networks = {}
    with _frozen_reference(parent):
        u0 = next((cache[key] for key in ("u0", "parent_margin", "margin") if key in cache), None)
        if u0 is None:
            with torch.no_grad():
                u0 = torch.cat([logit_margin(parent(x[start:start + batch_size].to(device)),
                                           y[start:start + batch_size].to(device)).cpu()
                                for start in range(0, len(x), batch_size)])
            diagnostics["parent_clean_forward_batches"] = math.ceil(len(x) / batch_size)
        u0 = torch.as_tensor(u0).detach().cpu().float()
        if u0.shape != (len(x),) or not bool(torch.isfinite(u0).all()):
            raise ValueError("Cached parent logit margin must be finite [N]")
        for kind_index, kind in enumerate(KINDS):
            generator = _generator(int(seed) + 17041 * (kind_index + 1))
            net = _new_net(kind, generator, device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
            branch_start = time.perf_counter()
            epochs = []
            for pass_index in range(passes):
                order = torch.randperm(len(x), generator=generator)
                drop_sum = 0.
                gradient_sum = 0.
                steps = 0
                width = 4 if kind == "linear" else 5
                minimum = torch.full((width,), float("inf"))
                maximum = torch.full((width,), -float("inf"))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    batch = x[indices].to(device)
                    labels = y[indices].to(device)
                    state = states[indices].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    augmented, parameters = _apply(net, batch, state, generator)
                    augmented_margin = logit_margin(parent(augmented), labels)
                    drop = u0[indices].to(device) - augmented_margin
                    loss = -drop.mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite action fitting loss")
                    loss.backward()
                    norm = torch.stack([p.grad.detach().square().sum()
                                        for p in net.parameters() if p.grad is not None]).sum().sqrt()
                    if not bool(torch.isfinite(norm)):
                        raise ValueError("Nonfinite action network gradient")
                    gradient_sum += float(norm)
                    optimizer.step()
                    drop_sum += float(drop.detach().sum())
                    minimum = torch.minimum(minimum, parameters.detach().min(0).values.cpu())
                    maximum = torch.maximum(maximum, parameters.detach().max(0).values.cpu())
                    diagnostics["parent_augmented_forward_batches"] += 1
                    steps += 1
                epochs.append(dict(pass_index=pass_index + 1, samples=len(order),
                                   unique_physical_ids=len(order), steps=steps,
                                   mean_parent_margin_drop=drop_sum / len(order),
                                   mean_action_gradient_norm=gradient_sum / steps,
                                   parameter_min=minimum.tolist(), parameter_max=maximum.tolist()))
            net.eval().requires_grad_(False)
            networks[kind] = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            diagnostics["branches"][kind] = dict(passes=epochs, optimizer_steps=passes * math.ceil(len(x) / batch_size),
                                                total_samples=passes * len(x),
                                                elapsed_seconds=time.perf_counter() - branch_start,
                                                parameters=sum(p.numel() for p in net.parameters()))
    unchanged = all(torch.equal(value, parent.state_dict()[name].detach().cpu())
                    for name, value in parent_before.items())
    if not unchanged:
        raise RuntimeError("Frozen parent model changed during action preparation")
    diagnostics.update(parent_parameters_and_buffers_unchanged=True,
                       elapsed_seconds=time.perf_counter() - started)
    return dict(format_version=FORMAT_VERSION, seed=int(seed), contract=action_contract(),
                fit_settings=dict(passes=passes, batch_size=batch_size, lr=2e-4, weight_decay=1e-4),
                networks=networks, diagnostics=diagnostics)


class FrozenActions:
    def __init__(self, payload, device):
        if payload.get("format_version") != FORMAT_VERSION or payload.get("contract") != action_contract():
            raise ValueError("Unsupported or changed fixed action artifact contract")
        self.device = torch.empty(0, device=device).device
        self.networks = nn.ModuleDict()
        for index, kind in enumerate(KINDS):
            net = _new_net(kind, _generator(int(payload["seed"]) + index), device)
            net.load_state_dict(payload["networks"][kind], strict=True)
            net.eval().requires_grad_(False)
            self.networks[kind] = net
        self.networks.eval()

    @torch.no_grad()
    def transform(self, x, kind, seed_or_generator):
        if kind not in KINDS:
            raise ValueError("Unknown physical L/T action kind")
        if x.device != self.device or tuple(x.shape[1:]) != (2, 256) or len(x) == 0:
            raise ValueError("Action IQ must be nonempty [B,2,256] on artifact device")
        generator = _generator(seed_or_generator)
        augmented, parameters = _apply(self.networks[kind], x, signal_state(x), generator)
        stats = _parameter_summary(parameters, kind)
        if not stats["finite"] or not stats["within_registered_bounds"]:
            raise ValueError("Physical action escaped its registered limits")
        return augmented.detach(), stats


def load_actions(payload, device):
    return FrozenActions(payload, device)
