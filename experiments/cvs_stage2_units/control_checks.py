"""Transient source/native graph checks; never publish the modified models."""
import argparse
import copy
import io
import time
import torch
from . import design as d
from .source import forward
from experiments.cvs_stage2_ssl.frequency_optim import FrequencyAdamW
from experiments.cvs_stage2_ssl.frequency_optim_checks import check as numerical_check
from experiments.cvs_stage2_hybrid.semisupervised import EMAIdentity,UnlabeledStream,VisitPassBank,make_plan,validate_unlabeled


def graphs(model,cache,bank,actions,device,seed,uc):
    from experiments.cvs_stage2_decouple.common import parameter_groups
    from experiments.cvs_stage2_decouple.source_counterfactual_audit import RNGState
    from experiments.cvs_stage2_geometry.features import identity
    from experiments.cvs_stage2_hybrid.optim import TemporarySAM
    initial=copy.deepcopy(model.state_dict());records=[];calls=[]
    hook=model.dom_backbone.register_forward_hook(lambda *args:calls.append(1))
    try:
        for arm in d.ARMS:
            model.load_state_dict(initial,strict=True);model.train()
            c=d.config(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}'))
            teacher=EMAIdentity(model);groups,names=parameter_groups(model)
            for g in groups:
                g['peak']=c['optimizer']['encoder_peak_lr' if g['name']=='encoder' else 'fusion_head_peak_lr']
                g['lr']=d.learning_rate(c,888,g['peak'])
            opt=FrequencyAdamW(model,groups,c['optimizer']['frequency']['peak_lr'],c['optimizer']['weight_decay'])
            opt.set_frequency_lr(d.frequency_learning_rate(c,888))
            assert all(torch.equal(initial[n],p) for n,p in model.state_dict().items())
            relation=bank.sample_relations(torch.Generator().manual_seed(seed+17))
            indices=torch.cat((torch.arange(96),relation['indices']))
            uindices,upasses=UnlabeledStream(len(uc['ids']),seed).batch(256)
            history=VisitPassBank();plan=make_plan(teacher,uc,uindices,upasses,888,seed,c['semi']['mode'],history)
            plan['rx']=uc['rx'][uindices].to(device);plan['ids']=(uindices+len(cache['ids'])).to(device)
            # Force code coverage with the student's prediction, not U truth.
            # This transient preflight intervention is never used by source.py.
            with torch.no_grad():plan['pseudo']=identity(model,plan['x_strong'])[0].argmax(1)
            plan['base_mask']=torch.ones(256,dtype=torch.bool,device=device)
            before={n:p.detach().clone() for n,p in model.named_parameters()};rng=RNGState()
            opt.zero_grad();loss,*_=forward(c,model,cache,indices,relation,888,device,actions,plan)
            loss.backward();entries=copy.deepcopy(history.entries)
            if c['enabled']['sam']:
                after=RNGState();sam=TemporarySAM(groups,c['sam']['rho']);sam.perturb();opt.zero_grad();rng.restore()
                try:
                    extra,*_=forward(c,model,cache,indices,relation,888,device,actions,plan);extra.backward()
                finally:sam.restore();after.restore()
            assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for g in groups for p in g['params'])
            assert teacher.calls==1 and teacher.updates==0 and history.entries==entries
            native_sinc=next(m for n,m in model.named_modules() if n.startswith('id_backbone.') and hasattr(m,'low_hz_'))
            with torch.no_grad():
                native_sinc.eval();filters_before=native_sinc._filters(torch.device(device),torch.float32).clone()
            version=native_sinc.low_hz_._version
            metrics=opt.step();teacher.update(model)
            assert native_sinc.low_hz_._version>version and teacher.calls==1 and teacher.updates==1
            with torch.no_grad():filters_after=native_sinc._filters(torch.device(device),torch.float32).clone()
            assert torch.isfinite(filters_after).all()
            # Model and optimizer coordinate state must restore the same next step.
            buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),buf);buf.seek(0)
            saved=torch.load(buf,map_location=device,weights_only=False)
            restored=copy.deepcopy(model);restored.load_state_dict(saved['model'],strict=True)
            restored_groups,_=parameter_groups(restored)
            ropt=FrequencyAdamW(restored,restored_groups,c['optimizer']['frequency']['peak_lr'],c['optimizer']['weight_decay'])
            ropt.load_state_dict(saved['optimizer'])
            for a,b in zip(model.parameters(),restored.parameters()):
                b.grad=None if a.grad is None else a.grad.clone()
            opt.step();ropt.step()
            assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
            records.append(dict(arm=arm,loss=float(loss.detach()),transient_steps=2,
                frequency=metrics,filter_max_abs_change=float((filters_after-filters_before).abs().max()),
                checkpoint_roundtrip=True,next_update_exact=True,teacher_calls=1,teacher_updates=1,
                forced_acceptance_for_graph_coverage_only=True))
        assert not calls
    finally:
        hook.remove();model.load_state_dict(initial,strict=True)
    return records


def check():
    from experiments.cvs_stage2_decouple.design import make_args
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_stage2_geometry.receiver_checks import synthetic_cache
    from experiments.cvs_stage2_geometry.receiver import fit_receiver,load_receiver
    from experiments.cvs_stage2_hybrid import actions as a
    seed=d.SEEDS[0];pc=d.parent_config(seed);args=make_args(pc,'cpu');args.num_domains,args.input_len=5,256
    with installed(pc) as native:model=native.build_baseline_model(args,torch.device('cpu'))
    cache=synthetic_cache();cache['x']=torch.randn(360,2,256)
    bank=load_receiver(fit_receiver(cache,seed))
    payload=dict(version=a.VERSION,contract=a.contract(),seed=seed,networks={k:a.new_net(k,a.a._generator(seed+i),'cpu').state_dict() for i,k in enumerate(a.a.KINDS)})
    uc=dict(x=cache['x'].clone(),ids=['U-'+str(i) for i in range(360)],role='U_s',preprocessing={},rx=cache['rx'],day=cache['day'],domain=cache['rx'],eq=torch.ones(360,dtype=torch.long),sig=torch.arange(360),base_index=torch.arange(360))
    validate_unlabeled(uc);numerical=numerical_check()
    result=graphs(model,cache,bank,a.FrozenBalanced(payload,'cpu'),'cpu',seed,uc)
    for row in d.rows():d.validate(d.config(row))
    from unittest.mock import patch
    from . import evaluate,dispatch
    with patch.object(evaluate,'snapshot',return_value=[d.config(d.rows()[0])]),patch.object(evaluate,'prediction_preflight',side_effect=ValueError('Incomplete prediction')),patch.object(d,'read',side_effect=AssertionError('Truth must stay closed')):
        try:evaluate.score()
        except ValueError:pass
        else:raise AssertionError('Incomplete prediction scored')
    assert dispatch.choose_gpu({0:dict(pids={1,2,3},free_mb=20000)}) is None
    assert dispatch.choose_gpu({0:dict(pids={1,2},free_mb=20000),1:dict(pids=set(),free_mb=11000)})==0
    return dict(status='PASS',numerical=numerical,graphs=result,source_input='synthetic native architecture; no real checkpoint loaded',target_access=False,production_training_updates=0,transient_copy_optimizer_updates=14,domain_forwards=0)


def remote_preflight():
    from experiments.cvs_stage2_decouple.common import verify_parent,build_model
    from experiments.cvs_stage2_geometry.receiver import load_receiver
    from experiments.cvs_stage2_hybrid.actions import FrozenBalanced
    from experiments.cvs_stage2_hybrid import design as h
    started=time.perf_counter();records=[]
    for seed in d.SEEDS:
        pc=d.parent_config(seed);provenance=verify_parent(pc);model=build_model(pc,'cpu');folder=d.AUDIT_BASE/f's{seed}'
        packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False)
        payload=torch.load(h.RELEASE_BASE/'actions'/f's{seed}.pt',map_location='cpu',weights_only=False)
        if payload['seed']!=seed or payload['source_ids']!=packed['labeled']['ids'] or payload['parent_provenance']!=provenance:raise ValueError('Action provenance differs')
        uc=torch.load(d.UNLABELED_CACHE,map_location='cpu',weights_only=False);validate_unlabeled(uc,packed['contract']['role_ids']['U_s'])
        bank=load_receiver(torch.load(folder/'receiver.pt',map_location='cpu',weights_only=False))
        records.append(dict(seed=seed,parent_provenance=provenance,graphs=graphs(model,packed['labeled'],bank,FrozenBalanced(payload,'cpu'),'cpu',seed,uc)))
    return dict(status='PASS',rows=records,target_access=False,production_training_updates=0,transient_copy_optimizer_updates=42,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);result=remote_preflight() if a.remote_preflight else check();d.write(a.output,result);print('PASS unit-aware optimizer and native SSL graph')
