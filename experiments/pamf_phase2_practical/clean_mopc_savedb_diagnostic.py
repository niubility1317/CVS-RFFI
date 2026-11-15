"""Diagnose a clean MoPC failure from its saved B, without query or truth."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import group2_cil_bridge as cil
from . import group2_cross_fit as cross
from .clean_mopc_diagnostic import support_only_clean
from .external_execution import seed_for
from .fast_transfer import to_tensor
from .group2_b_state import load_b_state
from .group2_dadda_mopc_diagnostic import trace_registration
from .group2_savedb_diagnostic import observe_support_forward
from .io import activate, read, require


def run(rt, job, k, n, device, variant, emit, *, repeats=1):
    require(variant in ('original', 'clip1', 'clip10', 'original_uninstrumented', 'affine_bridge'),
            'Unsupported diagnostic variant')
    require(type(repeats) is int and 1 <= repeats <= 20 and
            (variant == 'original_uninstrumented' or repeats == 1), 'Invalid diagnostic repeat count')
    require(job.get('profile') in ('protonet_cda','dadda_sda') and job.get('methods') == ['mopc_hr']
            and job.get('save_B_for_mopc') and [k, n] in job['points'], 'Wrong diagnostic job')
    require(variant != 'affine_bridge' or job['profile'] == 'protonet_cda', 'Affine bridge requires ProtoNet')
    require(job.get('mopc_bridge', 'original') == 'original', 'Diagnostic requires original bridge B state')
    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = Path(rt['data_root']) / job['capsule']
    splits = [read(p) for p in (cap / 'splits').glob('*.json')]
    old = [x for x in splits if x['k'] == k and x['new_count'] == 0]
    new = [x for x in splits if x['k'] == k and x['new_count'] == n]
    require(len(old) == len(new) == 1 and old[0]['support_indices'] == new[0]['support_indices'][:6*k],
            'Wrong support identity')
    b_seed = seed_for(int(job['seed']), job['capsule'], k, 'B')
    c_seed = seed_for(int(job['seed']), job['capsule'], k, n, 'C')
    saved = Path(job['output']) / f'B-K{k}.pt'
    payload = load_b_state(saved, job, k, old[0]['split_id'], b_seed, device)
    anchor = payload['anchor']
    if variant == 'affine_bridge':
        from .group2_proto_mopc_affine import ProtoMoPCAffine, require_equivalent_b
        old_x, old_y = support_only_clean(rt['data_root'], rt['clean_root'], job['capsule'],
                                          old[0]['split_id'], new_only=False)
        old_x = to_tensor(old_x, device, np.float32)
        old_y = to_tensor(old_y, device, np.int64).long()
        candidate = ProtoMoPCAffine(payload['model'], payload['old_prototypes']).to(device).eval()
        with torch.no_grad():
            baseline = anchor.model(old_x)['tx_logits']
            alternative = candidate(old_x)['tx_logits']
            require_equivalent_b(baseline, alternative)
        anchor = cil.make_anchor(candidate, old_x, old_y, method='mopc_hr')
        emit(dict(event='AFFINE_B_SUPPORT_MATCH', centered_logits_match=True, decisions_match=True,
                  old_classes=6, query_read=False, truth_read=False))
    x, y = support_only_clean(rt['data_root'], rt['clean_root'], job['capsule'], new[0]['split_id'])
    x = to_tensor(x, device, np.float32)
    y = to_tensor(y, device, np.int64).long()
    clip = {'original': None, 'clip1': 1.0, 'clip10': 10.0, 'original_uninstrumented': None,
            'affine_bridge': None}[variant]
    emit(dict(event='SAVED_CLEAN_B_SUPPORT_READY', variant=variant, key=job['key'], k=k, n=n,
              b_seed=b_seed, c_seed=c_seed, query_read=False, truth_read=False, predictions_written=False))

    def register(prepared, inputs, labels, *, seed, methods):
        require(tuple(methods) == ('mopc_hr',), 'Wrong diagnostic methods')
        return {'mopc_hr': cil.register(prepared['mopc_hr'], inputs, labels, seed=seed,
                                        mopc_semantics='public_code', mopc_max_grad_norm=clip)}

    if variant == 'original_uninstrumented':
        for i in range(repeats):
            try:
                state = cil.register(anchor, x, y, seed=c_seed, mopc_semantics='public_code')
                emit(dict(event='REPEAT_COMPLETE', repeat=i+1, updates=len(state.loss_trace),
                          query_read=False, truth_read=False, predictions_written=False))
            except FloatingPointError as exc:
                emit(dict(event='DIAGNOSTIC_RESULT', variant=variant, status='NONFINITE', repeat=i+1,
                          error=str(exc), optimizer_changed=False, query_read=False,
                          truth_read=False, predictions_written=False))
                return
        emit(dict(event='DIAGNOSTIC_RESULT', variant=variant, status='COMPLETE', repeats=repeats,
                  optimizer_changed=False, query_read=False, truth_read=False, predictions_written=False))
        return

    try:
        with patch.object(cross, 'register_c', register), observe_support_forward(emit):
            trace_registration({'mopc_hr': anchor}, x, y, c_seed, emit)
        status, error = 'COMPLETE', None
    except FloatingPointError as exc:
        status, error = 'NONFINITE', str(exc)
    emit(dict(event='DIAGNOSTIC_RESULT', variant=variant, status=status, error=error,
              optimizer_changed=clip is not None, adapter_changed=variant=='affine_bridge',
              query_read=False, truth_read=False,
              predictions_written=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'job', 'variant'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--device', choices=['cuda:0'], required=True)
    p.add_argument('--repeats', type=int, default=1)
    a = p.parse_args()
    run(read(a.runtime), read(a.job), a.k, a.n, a.device, a.variant,
        lambda event: print(json.dumps(event, allow_nan=False), flush=True), repeats=a.repeats)


if __name__ == '__main__':
    main()
