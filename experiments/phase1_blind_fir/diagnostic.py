"""Source-only line-oracle diagnostic for data-fitted FIR perturbations."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_innovation_probe.metrics import oracle_line_probe
from experiments.phase1_blind_fir.fit import fir_design, fit_fir_delta

COUNTS = (8, 32, 64)
RIDGE = .1
MAX_RELATIVE = .25


def _pair(x):
    return torch.stack((x.real, x.imag), 1)


@torch.no_grad()
def diagnose(dura, received, clean):
    from experiments.pamf_action_restore.operators import Action
    context = dura.estimate_context(received)
    anchor = dura.stage_outputs(received, context)[0][-1]
    action = context[1]
    x = torch.complex(anchor[:, 0], anchor[:, 1])
    y = torch.complex(received[:, 0], received[:, 1])
    residual = y-action(x)
    result = {'baseline': oracle_line_probe(anchor, clean, _pair(action.adjoint(residual)))}
    before = residual.abs().square().sum(-1).clamp_min(1e-12)
    for count in COUNTS:
        delta = fit_fir_delta(x, residual, action.main, action.image, action.kernel,
                              count=count, ridge=RIDGE, max_relative=MAX_RELATIVE)
        padded = torch.nn.functional.pad(delta, (0, action.kernel.shape[1]-count))
        updated = Action(action.kernel+padded, action.main, action.image)
        prediction_change = updated(x)-action(x)
        coeff = torch.cat((delta.real, delta.imag), -1).to(x.dtype)
        via_design = torch.einsum('bnk,bk->bn', fir_design(x, action.main, action.image, count), coeff)
        torch.testing.assert_close(prediction_change, via_design, rtol=3e-4, atol=2e-5)
        after = y-updated(x)
        ratio = after.abs().square().sum(-1)/before
        if not torch.isfinite(ratio).all() or (ratio > 1.0001).any():
            raise ValueError('Fitted FIR must not increase the observed residual')
        probe = oracle_line_probe(anchor, clean, _pair(updated.adjoint(after)))
        probe['residual_ratio'] = ratio
        probe['relative_kernel_delta'] = delta.abs().square().sum(-1).sqrt()/action.kernel.abs().square().sum(-1).sqrt().clamp_min(1e-12)
        result[str(count)] = probe
    return result


def run(root, out, commit):
    if out.exists():
        raise FileExistsError('A new source diagnostic output root is required')
    if not torch.cuda.is_available():
        raise ValueError('An explicitly selected GPU is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(root)
    dura = frozen_dura(root, 'cuda:0')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('DURA must remain frozen')
    x = core.tensor(received[ids[:2]], 'cuda:0')
    y = core.tensor(clean[ids[:2]], 'cuda:0')
    diagnose(dura, x, y)
    print('BLIND_FIR_REAL_SOURCE_SMOKE_PASS 2', flush=True)
    out.mkdir(parents=True)
    chunks = {}
    started = time.monotonic()
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        values = diagnose(dura, core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0'))
        for model, metrics in values.items():
            for key, tensor in metrics.items():
                if tensor.dtype == torch.bool:
                    tensor = tensor.to(torch.float32)
                chunks.setdefault((model, key), []).append(tensor.detach().cpu().to(torch.float64))
        if (start+len(selected)) % 168 == 0:
            print('BLIND_FIR_SOURCE_PROGRESS', start+len(selected), flush=True)
    arrays = {key: np.from_dlpack(torch.cat(value).contiguous()).copy() for key, value in chunks.items()}
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if int(mask.sum()) != 48:
                raise ValueError('Each fixed source receiver and view requires 48 records')
            rows = {}
            for model in ('baseline', *(str(c) for c in COUNTS)):
                rows[model] = {key: float(np.mean(arrays[(model, key)][mask]))
                    for key in ('anchor_error', 'positive_error', 'signed_error', 'optimal_alpha')}
                if model != 'baseline':
                    rows[model].update({key: float(np.mean(arrays[(model, key)][mask]))
                        for key in ('residual_ratio', 'relative_kernel_delta')})
            groups[name+'/'+view] = dict(count=48, models=rows)
    core.write(out/'result.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', commit=commit,
        records=len(ids), groups=groups, counts=list(COUNTS), ridge=RIDGE, max_relative=MAX_RELATIVE,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0, source_only=True,
        caveat='Oracle step uses source clean IQ and is not deployable or an identity-accuracy bound.'))
    print('BLIND_FIR_SOURCE_DIAGNOSTIC_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
