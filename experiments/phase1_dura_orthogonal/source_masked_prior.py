"""Train a clean-source masked IQ prior; no target inference or classifier."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import load_source_training, WIDE_FIT_SIZE

SPAN = 32
SEED = 2026110707


def masked_inputs(clean, starts):
    if clean.ndim != 3 or clean.shape[1:] != (2, 256) or starts.shape != (len(clean),):
        raise ValueError('Clean IQ and one mask start per record required')
    if bool(((starts < 1) | (starts + SPAN >= 256)).any()):
        raise ValueError('Mask requires both adjacent samples')
    positions = torch.arange(256, device=clean.device)[None, :]
    mask = (positions >= starts[:, None]) & (positions < starts[:, None] + SPAN)
    observed = clean.masked_fill(mask[:, None], 0.)
    scale = (observed.square().sum((1, 2), keepdim=True) / (2 * (256-SPAN))).clamp_min(1e-12).sqrt()
    rows = torch.arange(len(clean), device=clean.device)
    left = clean[rows, :, starts - 1]
    right = clean[rows, :, starts + SPAN]
    weight = ((positions - starts[:, None] + 1) / (SPAN + 1)).clamp(0, 1)
    linear = left[:, :, None]*(1-weight[:, None]) + right[:, :, None]*weight[:, None]
    return observed, mask, scale, linear


class MaskedIQPrior(nn.Module):
    """A residual dilated-convolution inpainting model; accepts no identity data."""
    def __init__(self, width=48):
        super().__init__()
        self.entry = nn.Conv1d(3, width, 7, padding=3)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d)
                                    for d in (1, 2, 4, 8, 16, 32))
        self.head = nn.Conv1d(width, 2, 1)

    def forward(self, observed, mask, scale):
        h = F.silu(self.entry(torch.cat((observed/scale, mask[:, None].to(observed.dtype)), 1)))
        for block in self.blocks:
            h = h + .2 * F.silu(block(h))
        return self.head(h) * scale


def errors(prediction, clean, mask):
    if prediction.shape != clean.shape or mask.shape != (len(clean), 256):
        raise ValueError('Matching prediction and fixed mask required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    delta = (prediction-clean).square() * mask[:, None]
    return delta.sum((1, 2)) / (2*SPAN*energy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source-only output required')
    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    _, _, clean_fit, fit = load_source_training(args.root, ids, WIDE_FIT_SIZE)
    smoke_iq = core.tensor(received[ids[:8]], 'cuda:0')
    with torch.no_grad():
        smoke = frozen_dura(args.root, 'cuda:0')(smoke_iq)
    if smoke.shape != smoke_iq.shape or not torch.isfinite(smoke).all():
        raise ValueError('Real DURA checkpoint source smoke failed')
    print('MASKED_PRIOR_DURA_SMOKE_PASS', len(smoke), flush=True)
    del smoke, smoke_iq
    model = MaskedIQPrior().to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    random = np.random.default_rng(SEED)
    train_data = core.tensor(clean_fit[fit], 'cuda:0')
    started = time.monotonic()
    losses = []
    model.train()
    for epoch in range(5):
        order = random.permutation(len(fit))
        total = torch.zeros((), device='cuda:0')
        for offset in range(0, len(order), 64):
            indices = order[offset:offset+64]
            y = train_data[torch.tensor(indices.tolist(), device='cuda:0')]
            starts = torch.tensor(random.integers(1, 256-SPAN, size=len(indices)).tolist(), device='cuda:0')
            observed, mask, scale, _ = masked_inputs(y, starts)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observed, mask, scale)
            loss = errors(prediction, y, mask).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite masked source loss required')
            loss.backward()
            optimizer.step()
            total += loss.detach()*len(indices)
        losses.append(float(total.cpu())/len(fit))
        print('MASKED_PRIOR_EPOCH', epoch+1, 'loss', losses[-1], 'elapsed', round(time.monotonic()-started, 2), flush=True)
    model.eval()
    pilot = core.tensor(clean[ids], 'cuda:0')
    generator = np.random.default_rng(SEED+1)
    starts = torch.tensor(generator.integers(1, 256-SPAN, size=len(ids)).tolist(), device='cuda:0')
    values = []
    with torch.no_grad():
        for offset in range(0, len(ids), 64):
            y, begin = pilot[offset:offset+64], starts[offset:offset+64]
            observed, mask, scale, linear = masked_inputs(y, begin)
            prediction = model(observed, mask, scale)
            values.append(torch.stack((errors(prediction, y, mask), errors(linear, y, mask),
                                       errors(torch.zeros_like(y), y, mask)), 1).cpu())
    values = np.from_dlpack(torch.cat(values).contiguous()).copy()
    if not np.isfinite(values).all():
        raise FloatingPointError('Finite masked source metrics required')
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            selected = values[np.isin(rx, receivers) & (scene == view)]
            if len(selected) != 48:
                raise ValueError('48 records per source pilot group required')
            groups[name+'/'+view] = dict(count=len(selected), model=float(selected[:, 0].mean()),
                linear=float(selected[:, 1].mean()), zero=float(selected[:, 2].mean()),
                model_beats_linear=float((selected[:, 0] < selected[:, 1]).mean()))
    args.out.mkdir(parents=True)
    torch.save(dict(model=model.state_dict(), fit_ids=fit, commit=args.commit, seed=SEED), args.out/'final.pt')
    core.write(args.out/'result.json', dict(status='SOURCE_MASKED_PRIOR_COMPLETE', commit=args.commit,
        fit_ids=len(fit), evaluation_ids=len(ids), epochs=5, span=SPAN, batch=64, losses=losses,
        elapsed_seconds=time.monotonic()-started, groups=groups,
        parameters=sum(p.numel() for p in model.parameters()), TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0,
        caveat='Source-only masked clean IQ reconstruction, not restoration or identity accuracy.'))
    print('MASKED_PRIOR_COMPLETE', len(fit), len(ids), flush=True)


if __name__ == '__main__':
    main()
