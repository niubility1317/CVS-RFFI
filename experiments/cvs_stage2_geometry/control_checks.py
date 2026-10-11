"""Synthetic control checks and read-only, source-only release preflight."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch


def check():
    assert importlib.util.find_spec('experiments.cvs_stage2_geometry.evaluate') is not None, 'Geometry evaluator missing'
    assert importlib.util.find_spec('experiments.cvs_stage2_geometry.dispatch') is not None, 'Geometry dispatcher missing'
    import numpy as np
    from . import design as d, evaluate as e, dispatch as q
    from experiments.cvs_stage2_decouple import common
    assert q.choose_gpu({0:dict(pids={1,2,3,4},free_mb=24000)}) is None
    assert q.choose_gpu({0:dict(pids={1,2,3},free_mb=13000),1:dict(pids={8},free_mb=16000)}) == 1
    assert q.choose_gpu({0:dict(pids=set(),free_mb=11000)}) is None
    with tempfile.TemporaryDirectory(prefix='geometry_control_') as temporary, ExitStack() as stack:
        root = Path(temporary)
        stack.enter_context(patch.object(d,'BASE',root/'run'))
        stack.enter_context(patch.object(d,'AUDIT_BASE',root/'audit'))
        c=d.config(d.rows()[0]);source=Path(c['output_root'])
        done=dict(status='SOURCE_TRAINED',config=c,epoch=40,optimizer_steps=8880,logged_steps=8880,
                  checkpoint_selection='final_only',target_access=False)
        frozen=dict(status='SOURCE_FROZEN',config=c,epoch=40,optimizer_steps=8880,
                    checkpoint=str(source/'final_ssdg.pth'),target_access=False)
        d.write(source/'completion.json',done)
        d.write(d.BASE/c['row_id']/'source_frozen.json',frozen)
        (source/'final_ssdg.pth').write_bytes(b'synthetic checkpoint; never loaded')
        assert e.snapshot(c['row_id']) == [c]
        with patch.object(common,'verify_parent',return_value={'status':'VERIFIED'}) as parent_check:
            assert e.source_provenance(c) == source
            parent_check.assert_called_once_with(c['parent_contract'])
        checkpoint=dict(epoch=40,optimizer_steps=8880,run_id=d.RUN,candidate_id=c['row_id'],config=c,
                        parent_checkpoint=c['parent_checkpoint'],checkpoint_selection='final_only',model={},baseline_args={})
        e.checkpoint_provenance(c,checkpoint)
        for key,value in [('optimizer_steps',8879),('candidate_id','C0-wrong'),('checkpoint_selection','best')]:
            try:e.checkpoint_provenance(c,dict(checkpoint,**{key:value}))
            except ValueError:pass
            else:raise AssertionError('Invalid checkpoint accepted: '+key)
        d.write(d.BASE/c['row_id']/'source_frozen.json',dict(frozen,epoch=39))
        try:e.snapshot(c['row_id'])
        except ValueError:pass
        else:raise AssertionError('Partial source freeze accepted')
        d.write(d.BASE/c['row_id']/'source_frozen.json',frozen)

        # Missing fixed prediction must fail before any truth read.
        real_read=d.read;reads=[]
        def traced(path):
            reads.append(str(path))
            if str(path)==str(d.TRUTH):raise AssertionError('Truth opened before frozen predictions')
            return real_read(path)
        with patch.object(e,'physical_ids',return_value=np.asarray(['synthetic'])),patch.object(d,'read',side_effect=traced):
            try:e.score(c['row_id'])
            except FileNotFoundError:pass
            else:raise AssertionError('Missing prediction did not stop scoring')
        assert str(d.TRUTH) not in reads

        for seed in d.SEEDS:
            out=d.AUDIT_BASE/f's{seed}'
            d.write(out/'preparation_complete.json',dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE',seed=seed,
                count=6300,target_access=False,model_state_restored=True,identity_updates=0,
                parent_contract=d.parent_config(seed),source_contract={'synthetic':True},audit=str(out/'audit.json')))
            d.write(out/'audit.json',dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE',seed=seed,count=6300,
                target_access=False,identity_updates=0,model_state_restored=True))
            for name in ('source_cache.pt','actions.pt','receiver.pt','source_responses.pt'):
                (out/name).write_bytes(b'only test metadata, never deserialized')
            assert q.preparation_ready(seed)
        assert len(q.require_preparations()) == 3
        broken=d.AUDIT_BASE/f's{d.SEEDS[0]}'/'preparation_complete.json'
        value=d.read(broken);d.write(broken,dict(value,target_access=True))
        assert not q.preparation_ready(d.SEEDS[0])
        try:q.require_preparations()
        except ValueError:pass
        else:raise AssertionError('Contaminated source preparation accepted')

    # Actual renamed score helpers operate on synthetic predictions, including stage.
    ids=[];labels=[];rx=[];days=[]
    for receiver in e.HELDOUT_RX:
        for day in e.TARGET_DAYS:
            for label in range(6):
                ids.append(str(len(ids)));labels.append(label);rx.append(receiver);days.append(day)
    arrays=[np.asarray(x) for x in (ids,labels,rx,days)]
    records,day_records=e.metric_rows(d.config(d.rows()[-1]),arrays[1],*arrays)
    assert len(records)==14 and len(day_records)==4 and records[0]['arm']=='B3'
    assert all(r['accuracy']==1 for r in records)
    results=[]
    for seed in d.SEEDS:
        for arm in ('P0',*d.ARMS):
            for record in records:
                value=deepcopy(record);value.update(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}')
                value['accuracy']=.8+.001*('P0',*d.ARMS).index(arm);results.append(value)
    summary,pairs=e.summarize(results),e.paired_results(results)
    assert {x['arm'] for x in summary}=={'P0','B0','B1','B2','B3'}
    assert all(x['complete_three_seeds'] for x in summary)
    assert {(x['treatment'],x['control']) for x in pairs}=={
        ('B0','P0'),('B1','B0'),('B2','B1'),('B3','B2'),('B3','B0'),('B3','P0')}
    count=dict(status='SCORED_COMPLETE',models=12,views=1,metric_records=168,day_metric_records=48,
        query_count_per_view=168000,prediction_count=2016000,truth_last=True,independent_recount='VERIFIED')
    q.require_scored(count,12)
    try:q.require_scored(dict(count,models=11),12)
    except ValueError:pass
    else:raise AssertionError('Incomplete full matrix accepted')
    values=np.arange(2*2*256,dtype=np.float32).reshape(2,2,256)
    import torch
    with patch.object(torch,'from_numpy',side_effect=TypeError('ABI bridge unavailable')):
        torch.testing.assert_close(e.iq_tensor(values,'cpu'),torch.tensor(values.tolist()))
    return dict(status='PASS',synthetic_only=True,real_target_access=False,remote_process_calls=False,
                frozen_E40_budget=True,parent_contract_routed=True,truth_last_failure_boundary=True,
                preparation_reuse_only=True,arms='B0-B3',capacity_includes_all=True,numpy_buffer_bridge=True)


def remote_preflight():
    """Reuse all source preparations; exercise fresh parents without an update.

    This runs on CPU even on the training host, so it neither occupies a GPU
    reservation nor competes with registered training slots. Only existing
    source artifacts and legally verified parent checkpoints are read.
    """
    import torch
    from . import design as d, dispatch as q
    from .actions import load_actions
    from .receiver import load_receiver
    from .source import forward_objective
    from experiments.cvs_stage2_decouple.common import build_model, verify_parent, parameter_groups
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context

    torch.set_num_threads(2)
    prepared = q.require_preparations()
    source_contract = d.read(d.SOURCE)
    records = []
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]), numerical_context(d.FULL_FP32_POLICY):
        for seed in d.SEEDS:
            folder = d.AUDIT_BASE / f's{seed}'
            complete = d.read(folder/'preparation_complete.json')
            provenance = verify_parent(d.parent_config(seed))
            packed = torch.load(folder/'source_cache.pt', map_location='cpu', weights_only=False)
            verify_contract(packed['contract'], source_contract)
            verify_contract(complete['source_contract'], packed['contract'])
            cache = packed['labeled']
            if (len(cache['ids']) != 6300 or len(set(cache['ids'])) != 6300 or
                    sorted(cache['ids']) != sorted(packed['contract']['role_ids']['L_s'])):
                raise ValueError('Preflight cached L_s physical IDs differ')
            if (len(packed['validation']['ids']) != 27000 or
                    sorted(packed['validation']['ids']) != sorted(packed['contract']['role_ids']['V'])):
                raise ValueError('Preflight cached V physical IDs differ')
            bank = load_receiver(torch.load(folder/'receiver.pt', map_location='cpu', weights_only=False))
            if (bank.payload['seed'] != seed or bank.payload['source_ids'] != cache['ids'] or
                    not torch.equal(bank.parent_feature, cache['parent_feature'])):
                raise ValueError('Preflight receiver/source parent cache mismatch')
            action_payload = torch.load(folder/'actions.pt', map_location='cpu', weights_only=False)
            if action_payload['seed'] != seed or action_payload['source_ids'] != cache['ids']:
                raise ValueError('Preflight action/source cache mismatch')
            relation = bank.sample_relations(torch.Generator().manual_seed(seed+723))
            indices = torch.cat((torch.arange(96), relation['indices']))
            x, y, margin = (cache[k][indices].to('cpu') for k in ('x','y','parent_margin'))
            for arm in d.ARMS:
                c = d.config(next(r for r in d.rows() if r['arm'] == arm and r['model_seed'] == seed))
                # Each arm starts again from the exact same formal parent. No
                # optimizer is constructed and no parameter update is allowed.
                model = build_model(c['parent_contract'], 'cpu')
                groups, _ = parameter_groups(model)
                model.train()
                actions = load_actions(action_payload, 'cpu') if c['enabled']['actions'] else None
                parameters_before = {n:p.detach().clone() for n,p in model.named_parameters()}
                domain_calls = []
                hook = model.dom_backbone.register_forward_hook(lambda *a: domain_calls.append(1))
                arm_started = time.perf_counter()
                try:
                    torch.manual_seed(seed+723)
                    total, weighted, parts, observed = forward_objective(c, model, x, y, margin, 888,
                        actions=actions, relation=relation, bank=bank)
                    if not torch.isfinite(total):
                        raise ValueError('Nonfinite real-parent preflight loss')
                    r_norm = None
                    if c['enabled']['receiver']:
                        gradients = torch.autograd.grad(weighted['receiver'], groups[0]['params'],
                            retain_graph=True, allow_unused=True)
                        r_norm = float(sum((g.detach().square().sum() for g in gradients if g is not None),
                                           torch.zeros(())).sqrt())
                        if not parts['receiver_details']['variance_gradient_retained']:
                            raise ValueError('Receiver variance gradient missing')
                    total.backward()
                    missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                    finite = all(torch.isfinite(p.grad).all() for g in groups for p in g['params'] if p.grad is not None)
                    if missing or not finite or domain_calls:
                        raise ValueError('Real-parent gradient/domain path check failed')
                    if any(not torch.equal(p, parameters_before[n]) for n,p in model.named_parameters()):
                        raise ValueError('Preflight changed parent parameters without an optimizer')
                    if actions is not None and any(p.grad is not None for p in actions.networks.parameters()):
                        raise ValueError('Frozen action network accumulated gradients')
                    records.append(dict(seed=seed,arm=arm,parent=provenance,device='cpu',batch=128,
                        mechanism_batch=32 if c['enabled']['actions'] else 0,loss=float(total.detach()),
                        receiver_encoder_gradient_norm=r_norm,missing_trainable_gradients=missing,
                        domain_forward_count=0,parameters_unchanged=True,optimizer_constructed=False,
                        target_access=False,seconds=time.perf_counter()-arm_started))
                finally:
                    hook.remove()
                del total, weighted, parts, observed, model, parameters_before, actions
            del packed, cache, bank, action_payload
    return dict(status='PASS',mode='SOURCE_ONLY_REMOTE_PREFLIGHT',device='cpu',target_access=False,
        preparations=prepared,source_audit_recomputed=False,formal_parent_models=12,
        formal_parent_forward_backward=records,optimizer_constructed=False,identity_updates=0,
        training_outputs_written=False,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-preflight',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Preflight output already exists; preserve the prior evidence')
    result=check()
    if args.remote_preflight:
        result=dict(status='PASS',synthetic=result,remote=remote_preflight())
    if args.output:
        from . import design as d
        d.write(args.output,result)
    print(json.dumps(result,indent=2),flush=True)
