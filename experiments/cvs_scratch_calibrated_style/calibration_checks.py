"""CPU-only synthetic calibration state, timing, and source-event checks."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from . import calibration as cal
from . import design as d
from experiments.cvs_e20_direction_audit.checks import Model, _fixture
from experiments.cvs_e20_boundary_style.multilevel import calibrate as original_calibrate


def snapshot(model):
    return dict(state={n:v.clone() for n,v in model.state_dict().items()},
        modes=[m.training for m in model.modules()],
        grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()],
        torch=torch.get_rng_state().clone(),python=random.getstate(),numpy=np.random.get_state(),
        hooks=sum(len(m._forward_hooks) for m in model.modules()))


def unchanged(model,before):
    assert all(torch.equal(v,before['state'][n]) for n,v in model.state_dict().items())
    assert before['modes']==[m.training for m in model.modules()]
    assert all((p.grad is None and g is None) or (g is not None and torch.equal(p.grad,g))
        for p,g in zip(model.parameters(),before['grads']))
    assert torch.equal(before['torch'],torch.get_rng_state()) and before['python']==random.getstate()
    after=np.random.get_state();old=before['numpy']
    assert old[0]==after[0] and np.array_equal(old[1],after[1]) and old[2:]==after[2:]
    assert before['hooks']==sum(len(m._forward_hooks) for m in model.modules())


def model_for(seed):
    torch.manual_seed(seed)
    model=Model(seed);model.multilevel=cal.BoundarySensitivity(model.base,seed)
    model.train();model.base.id_backbone.t2.bn.eval()
    model.base.id_backbone.cls_head.bias.grad=torch.ones_like(model.base.id_backbone.cls_head.bias)
    return model


def reject(call,types=(ValueError,FileExistsError)):
    try:call()
    except types:return
    raise AssertionError('Invalid calibration lifecycle was accepted')


def _run():
    from .source import publish_calibration_ready
    torch.set_num_threads(2)
    configs={r['calibration_policy']:r for r in (d.config(x) for x in d.rows())
        if r['base_arm']=='M_BOTH' and r['model_seed']==d.SEEDS[0]}
    L=_fixture();results={}
    for policy,c in configs.items():
        model=model_for(c['model_seed']);after=c['multilevel']['calibration_after_updates']
        with TemporaryDirectory() as tmp:
            out=Path(tmp);before=snapshot(model)
            current=cal.begin_calibration(c,model,L,'cpu',out)
            unchanged(model,before)
            initial=deepcopy(current);events=[]
            if after:
                assert current['status']=='PENDING_CALIBRATION' and not (out/'calibration.json').exists()
                reject(lambda:cal.calibrate(c,model,L,'cpu',out,after_identity_updates=0))
                reject(lambda:cal.require_calibration(c,model,current,448))
                reject(lambda:cal.advance_calibration(c,model,L,'cpu',out,current,445))
                # A bounded synthetic update clock exercises the same public
                # controller. These are toy bias updates, never experiment IQ.
                optimizer=torch.optim.SGD([model.base.id_backbone.cls_head.bias],lr=1e-5)
                for completed in range(1,445):
                    optimizer.zero_grad(set_to_none=True)
                    model.base.id_backbone.cls_head.bias.sum().backward();optimizer.step()
                    before=snapshot(model)
                    current,ready=cal.advance_calibration(c,model,L,'cpu',out,current,completed)
                    unchanged(model,before)
                    if ready:events.append(completed)
                    if completed<444:assert not (out/'calibration.json').exists()
                assert events==[444]
            else:
                assert current['status']=='FIXED_CALIBRATION_COMPLETE';events.append(0)
                # Initial calibration preserves the original exact numerical
                # recipe and the twelve private pair identities.
                peer=model_for(c['model_seed'])
                with TemporaryDirectory() as original:
                    previous=original_calibrate(c,peer,L,'cpu',original)
                assert previous['fixed_coefficients']==current['fixed_coefficients']
                for a,b in zip(previous['records'],current['records']):
                    assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
            assert current['pairs']==12 and current['model_forwards']==24 and current['extra_source_readouts']==1440
            assert current['after_identity_updates']==after and current['identity_updates']==0
            for step in (445,446,447,448):cal.require_calibration(c,model,current,step)
            same,ready=cal.advance_calibration(c,model,L,'cpu',out,current,448)
            assert same is current and not ready
            saved=(out/'calibration.json').read_bytes()
            reject(lambda:cal.calibrate(c,model,L,'cpu',out,after_identity_updates=after))
            assert (out/'calibration.json').read_bytes()==saved
            resolved=dict(config='synthetic',fishr_calibration=initial)
            (out/'initial_resolved_config.json').write_text(json.dumps(resolved),encoding='utf-8')
            with redirect_stdout(io.StringIO()) as stdout:
                final,event=publish_calibration_ready(out,resolved,current,after)
            assert stdout.getvalue().startswith('CALIBRATION_READY ')
            assert json.loads((out/'initial_resolved_config.json').read_text())==resolved
            assert json.loads((out/'resolved_config_final.json').read_text())==final
            assert json.loads((out/'resolved_config.json').read_text())==final
            assert json.loads((out/'calibration_ready.json').read_text())==event
            reject(lambda:publish_calibration_ready(out,resolved,current,after))
            results[policy]=dict(after_identity_updates=after,ready_events=events,pairs=12,
                fixed_coefficients=current['fixed_coefficients'],state_RNG_modes_gradients_preserved=True)
    # Role/exception checks restore changed buffers, flags and random streams,
    # and publish no completed file or coefficients after a failure.
    c=configs['CAP'];model=model_for(c['model_seed'])
    with TemporaryDirectory() as tmp:
        for role in ('U_s','V','target'):
            reject(lambda role=role:cal.calibrate(c,model,dict(L,role=role),'cpu',tmp))
        before=snapshot(model)
        def fail(*args,**kwargs):
            torch.rand(3);random.random();np.random.random()
            model.base.id_backbone.t2.bn.running_mean.add_(1)
            raise RuntimeError('injected calibration failure')
        with patch.object(model.multilevel,'_pair',side_effect=fail):
            reject(lambda:cal.calibrate(c,model,L,'cpu',tmp),types=(RuntimeError,))
        unchanged(model,before)
        assert not (Path(tmp)/'calibration.json').exists() and not hasattr(model.multilevel,'fixed_coefficients')
    # Pin the integration order in the actual loop without running a training
    # job: optimizer step -> counted/logged update -> calibration -> next batch.
    source=Path(__file__).with_name('source.py').read_text(encoding='utf-8');ast.parse(source)
    order=[source.index(text) for text in (
        "step=(epoch-1)*222+local",'optimizer.step()',"counts['identity_updates']+=1",
        'log.write(', 'calibration,became_ready=advance_calibration(')]
    assert order==sorted(order)
    assert source.index('require_calibration(c,model,calibration,step)')<source.index('loss,parts,weighted,detail=objective(')
    assert 'range(1,21)' in source and 'range(1,223)' in source
    return dict(status='PASS',device='cpu',fixture_only=True,target_access=False,real_training=False,
        policies=results,first_auxiliary_step=448,initial_recipe_exact=True,
        unique_completion_file_and_event=True,forbidden_roles_rejected=True,
        exception_state_and_RNG_restored=True,source_loop_order_AST=True,
        full_training_budget_unchanged=dict(epochs=20,steps_per_epoch=222,updates=4440))


def run():
    # The shared pair helper queries CUDA availability; CPU checks must not
    # initialize another process on any experiment GPU.
    with patch('torch.cuda.is_available',return_value=False):return _run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();value=run()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(value,allow_nan=False))
