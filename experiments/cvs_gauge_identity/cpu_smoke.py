"""Disposable CPU/runtime validation; no formal dataset or checkpoint."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.cvs_gauge_identity.model import VARIANTS, build, frozen_synthetic_diagnostics


def run():
    torch.set_num_threads(2)
    rows = []
    for seed in range(2026092701, 2026092705):
        for variant in VARIANTS:
            torch.manual_seed(seed)
            model = build(variant).cpu()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            losses = []
            for kind in ('random', 'weak', 'constant'):
                x = torch.randn(4, 2, 256)
                if kind == 'weak':
                    x *= 1e-9
                    x[0].zero_()
                elif kind == 'constant':
                    x.fill_(0.2)
                x.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), torch.arange(4))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                optimizer.step()
                losses.append(dict(input_kind=kind, ce=float(loss.detach())))
            diagnostic = frozen_synthetic_diagnostics(model)
            assert diagnostic['phase_logit_max_abs_error'] < 1e-3
            if diagnostic['affine_invariance_claimed']:
                assert diagnostic['affine_logit_max_abs_error'] < 1e-3
            assert all(math.isfinite(v) for v in diagnostic['unit_embedding_distances'].values())
            rows.append(dict(seed=seed, variant=variant, contract=model.contract(),
                             losses=losses, frozen_synthetic=diagnostic))
    return dict(status='PASS', python=sys.executable, torch_version=torch.__version__,
                device='cpu', disposable_models=8, disposable_ce_updates=24,
                formal_data_access=False, target_access=False, checkpoint_access=False,
                claim='Runtime validation only; no formal training or hardware identification.', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    with args.output.open('x', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
