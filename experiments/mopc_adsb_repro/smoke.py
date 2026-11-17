"""Real-training-record, no-query GPU smoke including checkpoint reload."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mopc_adsb_repro.run import regularization, correct
from experiments.pamf_phase2_practical.mopc_author import MoPCAuthorWiSig256


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); data, out = Path(a.data), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    x = np.load(data / 'train_x.npy', mmap_mode='r')
    y = np.load(data / 'train_labels.npy')
    torch.manual_seed(1731); torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    model = MoPCAuthorWiSig256(num_classes=100).cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0002)
    xb = torch.from_numpy(np.array(x[:16], copy=True)).cuda()
    yb = torch.as_tensor(y[:16], device='cuda', dtype=torch.long)
    loss = F.cross_entropy(model.fc(model.feature_extract_stage2(xb))[:, :50], yb)
    if not torch.isfinite(loss): raise FloatingPointError('Base smoke loss')
    loss.backward(); opt.step()
    torch.save(model.state_dict(), out / 'smoke_checkpoint.pt')
    original = MoPCAuthorWiSig256(num_classes=100).cuda()
    original.load_state_dict(torch.load(out / 'smoke_checkpoint.pt')); original.eval()
    with torch.no_grad(): prototype = original.feature_extract_stage2(xb).mean(0, keepdim=True).repeat(50, 1)
    results = {}
    for version in ('author',):
        candidate = copy.deepcopy(original).train(); prev = copy.deepcopy(original).eval().requires_grad_(False)
        z = candidate.feature_extract_stage2(xb)
        logits = candidate.fc(z)[:, :75]
        p = candidate.fc(prototype[:16])[:, :75]
        p = p / 2
        objective = F.cross_entropy(logits, yb) + F.cross_entropy(p, torch.arange(16, device='cuda')) + regularization(candidate, prev, version)
        objective.backward()
        finite = torch.stack([torch.isfinite(v.grad).all() for v in candidate.parameters() if v.grad is not None]).all()
        if not torch.isfinite(objective) or not finite: raise FloatingPointError(version)
        result = correct(prototype, prototype[:2], prototype[:2] + .1, version)
        if not torch.isfinite(result).all(): raise FloatingPointError(version + ' prototype')
        results[version] = float(objective.detach())
    (out / 'smoke_verified.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(), 'losses': results, 'query_read': False}), encoding='utf-8')


if __name__ == '__main__': main()
