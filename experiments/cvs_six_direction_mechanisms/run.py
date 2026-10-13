"""Six prespecified source mechanism checks; no target entry point."""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import torch

from . import design as d
from .common import load_model, load_roles, context, preserve_rng, snapshot, assert_unchanged
from experiments.cvs_conditional_rx_probe_health.run import _rng_state, _assert_rng, _write_new

MODULES = dict(matching='matching', sampling='sampling', update='updates',
               transform='transforms', directed='directed', sets='sets')


def verify_registration():
    spec = d.read(d.ROOT/'registration/experiment.json')
    expected = {r['row_id']: d.config(r) for r in d.rows()}
    if spec.get('run_id') != d.RUN or len(spec.get('rows', [])) != 3:
        raise ValueError('Three fixed source model rows required')
    if {r['row_id'] for r in spec['rows']} != set(expected):
        raise ValueError('Source diagnostic row matrix differs')
    for r in spec['rows']:
        if r['config'] != expected[r['row_id']]:
            raise ValueError('Source diagnostic config differs')
    return spec


def preflight(output, device='cpu'):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_source_pseudotarget_health.probes import capture, descriptors
    output = Path(output)
    if output.exists():
        raise FileExistsError('No-query smoke already exists; reconcile before retry')
    state = _rng_state(device); rows = []
    with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            model, proof = load_model(row, device); before = snapshot(model)
            with torch.no_grad(), capture(model) as maps:
                out = model.forward_identity_only(torch.zeros(2, 2, 256, device=device))
            features = descriptors(maps, out['id_feat_joint'])
            if out['tx_logits'].shape != (2, 6) or not torch.isfinite(out['tx_logits']).all():
                raise ValueError('Frozen CE30 real-checkpoint smoke failed')
            rows.append(dict(row_id=row['row_id'], checkpoint=proof['source_checkpoint'],
                             feature_shapes={k:list(v.shape) for k,v in features.items()},
                             **assert_unchanged(model, before)))
            del model
    _assert_rng(state, device)
    result = dict(status='PASS', run_id=d.RUN, real_checkpoints=3, rows=rows,
                  no_query=True, source_IQ_opened=False, U_truth_read=False,
                  target_access=False, identity_updates=0, RNG_preserved=True)
    _write_new(output, result)
    print('SIX_DIRECTION_REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS checkpoints=3', flush=True)
    return result


def run(output, device='cuda:0', registration=True):
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    if registration: verify_registration()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); torch.set_num_threads(2); state = _rng_state(device)
    records = []
    try:
        with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
            roles, contract = load_roles(d.source_config(d.rows()[0]))
            _write_new(output/'source_contract.json', contract)
            _write_new(output/'resolved_config.json', dict(run_id=d.RUN, pid=os.getpid(),
                cwd=os.getcwd(), python=sys.executable, device=device,
                commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                config=[d.config(r) for r in d.rows()], roles=dict(d.ROLE_COUNTS),
                backend_flags=actual_flags(), identity_updates=0, V_updates=0,
                U_truth_read=False, target_access=False, source_V='health only'))
            for row in d.rows():
                rowpath = output/row['row_id']; rowpath.mkdir(exist_ok=False)
                model, proof = load_model(row, device); before = snapshot(model)
                if proof['source_contract'] != contract:
                    raise ValueError('CE30 source contract differs from current role handles')
                _write_new(rowpath/'model_contract.json', proof)
                print('SIX_DIRECTION_SOURCE_SCAN '+row['row_id'], flush=True)
                ctx = context(row, model, roles, device)
                raw = dict(features=ctx['features'], probabilities=ctx['probabilities'],
                           labels={k:{n:v for n,v in cache.items() if n!='x'} for k,cache in roles.items()},
                           folds=ctx['folds'], fold_info=ctx['fold_info'], state_q=ctx['state_q'],
                           state_mean=ctx['state_mean'], state_scale=ctx['state_scale'],
                           target_access=False, U_truth_read=False, raw_IQ_persisted=False)
                with (rowpath/'raw_source.pt').open('xb') as stream: torch.save(raw,stream)
                results = {}
                for direction in d.DIRECTIONS:
                    print('SIX_DIRECTION_BEGIN '+row['row_id']+' '+direction, flush=True)
                    module = importlib.import_module('.'+MODULES[direction], __package__)
                    t = time.perf_counter()
                    value = module.diagnose(ctx, rowpath/direction, device)
                    if not isinstance(value,dict) or not isinstance(value.get('applicability',{}).get('passed'),bool):
                        raise ValueError('Fixed source applicability conclusion missing: '+direction)
                    value.update(direction=direction, row_id=row['row_id'],
                                 elapsed_seconds=time.perf_counter()-t,
                                 U_truth_read=False, target_access=False, V_updates=0,
                                 **assert_unchanged(model,before))
                    _write_new(rowpath/(direction+'.json'),value); results[direction]=value
                    print('SIX_DIRECTION_END '+row['row_id']+' '+direction+' pass='+str(value['applicability']['passed']), flush=True)
                done = dict(status='SOURCE_MECHANISMS_COMPLETE', **row, directions=list(d.DIRECTIONS),
                            applicability={k:v['applicability'] for k,v in results.items()},
                            **assert_unchanged(model,before), target_access=False,U_truth_read=False)
                _write_new(rowpath/'completion.json',done); records.append(done)
                del ctx,raw,model,before
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
            analysis = dict(status='SOURCE_MECHANISMS_ANALYZED',run_id=d.RUN,rows=records,
                directions={k:dict(seed_passes=[r['applicability'][k]['passed'] for r in records],
                    eligible_for_preregistered_scratch30=all(r['applicability'][k]['passed'] for r in records),
                    selection='all three fixed primary source checks must pass; V accuracy and target scores excluded')
                    for k in d.DIRECTIONS}, identity_updates=0,U_truth_read=False,target_access=False,
                claim_scope='Source applicability only; no identity improvement or unknown-RX guarantee',
                next_identity_budget='scratch30 explicitly confirmed by user; independent matched controls')
            _write_new(output/'analysis.json',analysis)
        _assert_rng(state,device)
        done = dict(status='SOURCE_MECHANISMS_COMPLETE',run_id=d.RUN,rows=3,direction_cells=18,
                    completed=records,identity_updates=0,V_updates=0,U_truth_read=False,target_access=False,
                    parameters_buffers_exactly_unchanged=True,modes_unchanged=True,identity_gradient_count=0,
                    RNG_preserved=True,elapsed_seconds=time.perf_counter()-started)
        _write_new(output/'completion.json',done)
        print('SIX_DIRECTION_SOURCE_COMPLETE rows=3 cells=18 identity_updates=0',flush=True)
        return done
    except Exception as exc:
        _write_new(output/'failure.json',dict(status='FAILED',run_id=d.RUN,error=repr(exc),
                    completed_rows=[r['row_id'] for r in records],no_automatic_retry=True))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:preflight(a.output,a.device)
    else:run(a.output,a.device)


if __name__=='__main__':main()
