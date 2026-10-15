"""Fixed source V, frozen real default head, independent virtual RX responses."""
import gzip
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from . import probe_design as d
from .augmentation import apply_response, session_response


def validate_source(V, contract):
    if (len(V['ids']) != 27000 or len(set(V['ids'])) != 27000
            or sorted(V['ids']) != sorted(contract['role_ids']['V'])
            or set(V['ids']) & (set(contract['role_ids']['L_s']) | set(contract['role_ids']['U_s']))
            or V['x'].shape != (27000, 2, 256) or V['y'].shape != (27000,)
            or set(V['rx'].tolist()) != {1, 3, 4, 6, 8}
            or set(V['day'].tolist()) != {1, 2, 3}
            or set(V['y'].tolist()) != set(range(6))
            or not bool(torch.isfinite(V['x']).all())):
        raise ValueError('Exact full source V27000 with legal TX/RX/day required')
    return V


def summary(pred, y, margin, reference):
    count = len(y)
    cm = torch.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
    tp = cm.diagonal().float()
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
    correct = pred == y
    fixed = reference == y
    return dict(count=count, correct=int(correct.sum()), accuracy=int(correct.sum())/count,
                macro_f1=float(f1.mean()), confusion=cm.tolist(), margin=float(margin.mean()),
                reference_clean_correct=int(fixed.sum()),
                clean_correct_retention=float(correct[fixed].float().mean()) if bool(fixed.any()) else None,
                repairs=int((correct & ~fixed).sum()), new_errors=int((~correct & fixed).sum()))


def gate(results, configuration=None):
    c = configuration or d.config()
    base = results['RAW']['overall']['accuracy']
    drops = {k:100 * (base - results[k]['overall']['accuracy']) for k in c['source_gate']['candidates']}
    return dict(status='PASS' if all(v <= c['source_gate']['max_accuracy_drop_pp'] + 1e-9 for v in drops.values()) else 'FAIL',
                accuracy_drop_pp=drops, threshold_pp=c['source_gate']['max_accuracy_drop_pp'],
                source_only=True, target_metrics_read=False,
                interpretation='Operator safety, not training improvement or target performance')


@torch.no_grad()
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This registered source probe is CPU-only')
    if d.BASE.exists():
        raise FileExistsError('Preserve and reconcile an existing source run')
    torch.set_num_threads(2)
    from experiments.cvs_sigtime_rdrop_legacy import design as legacy
    from experiments.cvs_sigtime_rdrop_legacy.provenance import own_model, source_proof
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context
    c = legacy.config(next(r for r in legacy.rows() if r['row_id'] == 'E20_RDROP-s2026092701'))
    if c['output_root'] != d.PARENT or c['epochs'] != 20:
        raise ValueError('Only the predeclared fixed E20 RDrop parent permitted')
    source_proof(c)  # Exact source roles, scratch ancestry and final-only, before checkpoint loading.
    started = time.perf_counter()
    d.BASE.mkdir()
    d.write(d.BASE / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    try:
        with numerical_context(legacy.FULL_FP32_POLICY):
            model = own_model(c, 'cpu')
            smoke = model.forward_identity_only(torch.zeros(2, 2, 256))['tx_logits']
            if smoke.shape != (2, 6) or not bool(torch.isfinite(smoke).all()):
                raise ValueError('Real checkpoint/default-head no-query smoke failed')
            d.write(d.BASE / 'source_smoke.json', dict(status='PASS', forward_packets=2, query_read=False,
                                                     checkpoint=d.PARENT + '/final_ssdg.pth', used_as_initializer=False))
            print('SMOKE PASS real E20 RDrop default head; no query; source probe continues', flush=True)
            preparation = d.read(Path(d.PREPARATION) / 'preparation_complete.json')
            if (preparation.get('target_access') is not False or preparation.get('identity_updates') != 0
                    or preparation.get('status') != 'SOURCE_GEOMETRY_AUDIT_COMPLETE'):
                raise ValueError('Existing read-only raw source preparation differs')
            packed = torch.load(Path(d.PREPARATION) / 'source_cache.pt', map_location='cpu', weights_only=False)
            contract = d.read(d.SOURCE)
            verify_contract(packed['contract'], contract)
            verify_contract(preparation['source_contract'], contract)
            args = d.read(Path(d.PARENT) / 'actual_native_args.json')
            args = args.get('args', args)
            if str(args.get('wisig_equalized')) != '1' or args.get('wisig_out_len') != 256:
                raise ValueError('Source waveform input contract differs from the frozen parent')
            V = validate_source({k:packed['validation'][k] for k in ('x', 'y', 'rx', 'day', 'ids')}, contract)
            del packed
            model.eval().requires_grad_(False)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            d.write(d.BASE / 'resolved_config.json', dict(config=d.config(), parent_config=c,
                parent_training_code=d.read(Path(d.PARENT) / 'resolved_config.json')['commit'],
                checkpoint_provenance='VERIFIED own full-L6300 scratch E20 final, no inherited weights',
                identity_input_policy=model._identity_input_policy, pid=os.getpid(), cwd=os.getcwd(),
                python=sys.executable, torch_version=torch.__version__, CUDA_VISIBLE_DEVICES='',
                source_cache=d.PREPARATION + '/source_cache.pt', source_role='V', U_opened=False,
                real_default_head=True, extra_parameters=0,
                module_files={k:str(sys.modules[k].__file__) for k in
                              ('model', 'DataAugmentation', 'leo_practical.channel', 'leo_practical.residual') if k in sys.modules},
                commit=(d.ROOT / 'release_commit.txt').read_text().strip()))
            outputs = {}
            wave = {}
            forwards = 2
            sessions = {}
            for condition in d.CONDITIONS:
                predictions, margins = [], []
                rms_error = mse = interior_error = 0.
                epoch = 1 if '_E1_' in condition else 20
                boundary = 'edge' if condition.endswith('_EDGE') else 'reflect'
                for start in range(0, 27000, 256):
                    end = min(start + 256, 27000)
                    x = V['x'][start:end]
                    y = V['y'][start:end]
                    if condition == 'RAW':
                        value = x
                    else:
                        coefficients = []
                        for rx, day in zip(V['rx'][start:end].tolist(), V['day'][start:end].tolist()):
                            a = session_response(20261015031, stream='source_probe', epoch=epoch,
                                                 rx=int(rx), day=int(day), eq=1)
                            key = f'epoch{epoch}/rx{rx}/day{day}/eq1'
                            v = [a.real, a.imag]
                            if key in sessions and sessions[key] != v:
                                raise ValueError('Virtual RX session changed within view/TX coverage')
                            sessions[key] = v
                            coefficients.append(a)
                        value, detail = apply_response(x, coefficients, boundary=boundary)
                        other, opposite = apply_response(x, coefficients, boundary='reflect' if boundary == 'edge' else 'edge')
                        a = value / detail['scalar_power_restoration'][:, None, None]
                        b = other / opposite['scalar_power_restoration'][:, None, None]
                        interior_error = max(interior_error, float((a[:, :, 1:255] - b[:, :, 1:255]).abs().max()))
                        if interior_error > 1e-6:
                            raise ValueError('Boundary dependence leaked into unscaled valid region')
                    p = x.square().sum(1).mean(1)
                    q = value.square().sum(1).mean(1)
                    rms_error = max(rms_error, float(((p-q).abs()/p.clamp_min(1e-12)).max()))
                    mse += float((value-x).square().sum())
                    logits = model.forward_identity_only(value)['tx_logits']
                    if logits.shape != (len(y), 6) or not bool(torch.isfinite(logits).all()):
                        raise ValueError('Finite default head logits required')
                    other = logits.masked_fill(F.one_hot(y,6).bool(), -torch.inf).max(1).values
                    margins.append((logits.gather(1,y[:,None]).squeeze(1)-other)/30)
                    predictions.append(logits.argmax(1))
                    forwards += len(x)
                    if start % 2048 == 0:
                        d.write(d.BASE / 'progress.json', dict(condition=condition, packets=end, forwards=forwards,
                                                              seconds=time.perf_counter()-started))
                outputs[condition] = (torch.cat(predictions), torch.cat(margins))
                wave[condition] = dict(max_relative_power_error=rms_error, waveform_MSE=mse/(27000*2*256),
                                       unscaled_boundary_interior_max_abs_error=interior_error)
                print('SOURCE '+condition+' complete packets=27000 forwards='+str(forwards), flush=True)
            if forwards != d.FORWARDS or not torch.equal(rng, torch.get_rng_state()):
                raise ValueError('Source forward budget or frozen random state differs')
            if any(not torch.equal(v, before[k]) for k,v in model.state_dict().items()):
                raise ValueError('Frozen parameters or model buffers changed')
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('Unexpected source gradient state')
            reference = outputs['RAW'][0]
            results = {}
            with gzip.open(d.BASE/'sample_decisions.jsonl.gz', 'xt', encoding='utf-8') as stream:
                for condition, (pred, margin) in outputs.items():
                    strata=[]
                    for field, tensor in (('RX',V['rx']),('day',V['day']),('TX',V['y'])):
                        for group in tensor.unique().tolist():
                            mask=tensor == group
                            strata.append(dict(dimension=field, value=int(group),
                                **summary(pred[mask], V['y'][mask], margin[mask], reference[mask])))
                    results[condition]=dict(overall=summary(pred,V['y'],margin,reference),strata=strata,waveform=wave[condition])
                    for i,pid in enumerate(V['ids']):
                        stream.write(json.dumps(dict(condition=condition, physical_id=pid, RX=int(V['rx'][i]),
                            day=int(V['day'][i]), TX=int(V['y'][i]), prediction=int(pred[i]),
                            correct=bool(pred[i]==V['y'][i]), margin=float(margin[i])),allow_nan=False)+'\n')
            decision=gate(results)
            result=dict(status='ANALYZED', source_only=True, query_read=False, config=d.config(),
                        conditions=results, source_gate=decision, virtual_sessions=sessions,
                        forward_packets=forwards, seconds=time.perf_counter()-started,
                        optimizer_updates=0, EMA_updates=0, head_fit_updates=0,
                        state_unchanged=True, U_opened=False, target_predictions=0,
                        parent_cost=dict(training_epochs=20, updates=4440,
                                         original_training_seconds=d.read(Path(d.PARENT)/'completion.json').get('elapsed_seconds')),
                        commit=(d.ROOT/'release_commit.txt').read_text().strip())
            d.write(d.BASE/'results.json',result)
            d.write(d.BASE/'complete.json',dict(status='ANALYZED',source_only=True,query_read=False,
                                               source_gate=decision,forward_packets=forwards,seconds=result['seconds']))
            print('SOURCE_PROBE_COMPLETE '+json.dumps(decision),flush=True)
    except Exception as exc:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=str(exc),no_automatic_retry=True,
                                          source_only=True,query_read=False))
        raise
