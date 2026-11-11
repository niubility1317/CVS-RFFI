import torch


def channel_view(x, angle, *, kind, delay, echo_amplitude):
    if x.ndim != 3 or x.shape[1] != 2 or angle.shape != (len(x),) or angle.device != x.device:
        raise ValueError('Aligned two-channel IQ and per-packet parameter required')
    if kind not in ('cfo', 'echo', 'cfo_echo'):
        raise ValueError('Unknown channel view')
    if kind == 'cfo' and (delay != 0 or echo_amplitude != 0.0):
        raise ValueError('Pure CFO has no echo')
    if kind != 'cfo' and (delay not in (1, 3) or echo_amplitude != 0.15 or x.shape[-1] <= delay):
        raise ValueError('Registered bounded echo required')
    y = x
    if kind in ('cfo', 'cfo_echo'):
        ramp = torch.linspace(-0.5, 0.5, x.shape[-1], device=x.device, dtype=x.dtype)
        theta = angle[:, None] * ramp[None, :]
        co, si = theta.cos(), theta.sin()
        y = torch.stack((co*x[:, 0]-si*x[:, 1], si*x[:, 0]+co*x[:, 1]), dim=1)
    if kind in ('echo', 'cfo_echo'):
        shifted = torch.nn.functional.pad(y[..., :-delay], (delay, 0))
        # A single uniformly distributed echo phase over [-pi, pi].
        theta = 8*angle[:, None]
        co, si = theta.cos(), theta.sin()
        echo = torch.stack((co*shifted[:, 0]-si*shifted[:, 1],
                            si*shifted[:, 0]+co*shifted[:, 1]), dim=1)
        mixed = y + echo_amplitude*echo
        before = y.square().mean(dim=(1,2), keepdim=True)
        after = mixed.square().mean(dim=(1,2), keepdim=True)
        y = mixed * torch.sqrt(before / after.clamp_min(1e-12))
    return y
