"""Actual identity graphs and shortened native-loop integration, source-free fixtures."""
import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import torch
from unittest import mock
from torch.utils.data import Dataset, DataLoader
from . import design as d
from .runtime import OnlineTrainer, Observations, gradient_vector, feature_parameter_groups, installed
from experiments.cvs_equivariant_identity.precision import numerical_context


class Synthetic(Dataset):
    def __init__(self, hidden=False, poison=False):
        self.hide=hidden;self.poison=poison
        self.index=[dict(tx_i=y,rx_i=r,day_i=day,eq_i=0,sig_i=j)
                    for day in range(3) for r in (1,3,4,6,8) for y in range(6) for j in range(8)]
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        if self.poison:raise AssertionError('Pure CE fetched hidden U')
        m=dict(self.index[i],base_index=i)
        x=torch.randn(2,256,generator=torch.Generator().manual_seed(19000+i))
        if self.hide:return x,-1,-1,{}
        return x,m['tx_i'],m['rx_i'],m


def make_config(arm):
    row=next(r for r in d.rows() if r['arm']==arm)
    return d.config(row,.05 if arm in d.SELECTED_ARMS else None)


def prime_statistics(trainer,model):
    """Synthetic-only EMA maturation without wasting 20 full training epochs."""
    if trainer.fishr is None:return
    from experiments.cvs_multi_disentangle.model import _encoder
    weight=_encoder(model.id_backbone).core.id_backbone.cls_head.weight
    generator=torch.Generator().manual_seed(1337)
    for day in range(3):
        for first,second in ((1,3),(3,4),(4,6),(6,8),(8,1)):
            y=torch.arange(6,device=weight.device).repeat_interleave(4).repeat(2)
            rx=torch.tensor([first]*24+[second]*24,device=weight.device)
            days=torch.full_like(y,day)
            for step in range(8):
                z=torch.randn(48,160,generator=generator).to(weight.device).requires_grad_()
                trainer.fishr(z,weight,y,rx,days,step=step+1)


def current_native(trainer,model,batch,epoch):
    x,y,rx,meta=batch;x=x.to(trainer.device);y=y.to(trainer.device);rx=rx.to(trainer.device)
    day=meta['day_i'].to(trainer.device);pid=[tuple(int(meta[k][i]) for k in ('rx_i','day_i','eq_i','sig_i','base_index')) for i in range(len(x))]
    logits=trainer.native_forward(model,x,y=y,rx=rx,day=day,pid=pid,epoch=epoch,kwargs={})
    return x,y,rx,day,pid,torch.nn.functional.cross_entropy(logits,y)


def check_hard_iq_retention(trainer,model,x):
    from . import runtime
    from experiments.cvs_multi_action_risk.physics import factorial_batch
    x=x[:32];batch=factorial_batch(x,trainer.generator,step=0)
    h=x.new_zeros(32,349);y=torch.zeros(32,dtype=torch.long,device=x.device);calls=[0]
    def predict(*args,**kwargs):
        calls[0]+=1;delta=h.clone();delta[:,0]=calls[0];return delta
    def logits(identity,value):
        out=value.new_zeros(len(value),6);out[:,1]=10*value[:,0];return out
    trainer.calibration[('linear',0)]=dict(weight=1.,error=0.,zero=1.,ids=set(range(16)))
    with mock.patch.object(trainer,'_predict',predict),mock.patch.object(trainer,'_quality_bins',lambda x:torch.zeros(len(x),dtype=torch.long,device=x.device)),mock.patch.object(runtime,'intermediate',lambda model,x:h),mock.patch.object(runtime,'classify_intermediate',logits):
        chosen=trainer._proposal('linear',x,h,y,batch,{'x10':h},model.id_backbone)
    assert trainer.last['linear_mispredicted_legal_IQ_retained']==24
    assert int(((chosen-batch['x10']).flatten(1).abs().sum(1)>0).sum())==24
    return dict(status='PASS',legal_mispredicted_hard_IQ_retained=24,random_IQ=8)


def focused_checks(output,device='cpu'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    c=make_config('LTR_S_AF5');a=d.make_args(c,device);a.num_domains=5;a.input_len=256
    with installed(c,training=False) as native:model=native.build_baseline_model(a,torch.device(device))
    trainer=OnlineTrainer(c,model);trainer.output_root=output;trainer.set_source(DataLoader(Synthetic(),batch_size=128))
    x,y,rx,meta=trainer.relation.next_batch(8,device=torch.device(device))
    retention=check_hard_iq_retention(trainer,model,x)
    from experiments.cvs_multi_disentangle.model import _encoder
    z=_encoder(model.id_backbone).features(x);w=_encoder(model.id_backbone).core.id_backbone.cls_head.weight
    before=deepcopy(trainer.fishr.state_dict())
    trainer._quality_fishr_audit(x,z,w,y,meta['rx_i'].to(device),888)
    assert trainer.fishr.state_dict()['_extra_state']==before['_extra_state']
    quality=trainer.audit_snapshots[-1]['fishr_quality'];assert len(quality)==8
    assert any(v['variance_gap'] is not None for v in quality)
    assert any(v['variance_gap'] is None for v in quality)
    for epoch,step in ((20,4399),(20,4439)):
        if not trainer.fishr.buckets:prime_statistics(trainer,model)
        batch=next(iter(DataLoader(Synthetic(),batch_size=128)))
        nx,ny,nr,nd,ids,ce=current_native(trainer,model,batch,epoch)
        trainer.loss(model,None,nx,ny,nr,epoch,step,days=nd,physical_ids=ids,native_loss=ce)
    assert len(trainer.fishr_calibrator.observations)==2
    assert trainer.fishr_calibrator.lambda_max>0
    trainer.observations.add(dict(sparse_regression=.75));trainer.epoch_count=3
    rows=[dict(epoch=20)];trainer.annotate(rows);first=deepcopy(rows)
    trainer.annotate(rows)
    assert first==rows and rows[0]['risk_action_sparse_regression']==.75
    assert rows[0]['risk_action_observation_counts']['sparse_regression']==1
    assert rows[0]['risk_action_auxiliary_active_steps_epoch']==3
    trainer.style.detach()
    result=dict(status='PASS',device=device,retention=retention,quality=quality,
                calibration=trainer.fishr_calibrator.state_dict(),duplicate_epoch_annotation='PASS')
    (output/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS targeted hard IQ retention, read-only quality-stratified Fishr, multi-batch fixed E20 calibration')


def check_runtime(output,device='cpu'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    obs=Observations();obs.add(dict(sparse=.5))
    for _ in range(33):obs.add(dict(dense=1.))
    assert obs.finish()['sparse']==.5
    loader=DataLoader(Synthetic(),batch_size=128,shuffle=False)
    batch=next(iter(loader));findings=[]
    for arm in d.ARMS:
        c=make_config(arm);a=d.make_args(c,device);a.num_domains=5;a.input_len=256
        with installed(c,training=False) as native:model=native.build_baseline_model(a,torch.device(device))
        trainer=OnlineTrainer(c,model);trainer.output_root=output/arm;trainer.set_source(loader)
        x,y,rx,days,ids,ce=current_native(trainer,model,batch,1)
        before=torch.random.get_rng_state().clone()
        warm=trainer.loss(model,None,x,y,rx,1,7,days=days,physical_ids=ids,native_loss=ce)
        assert torch.equal(before,torch.random.get_rng_state())
        if arm=='native':assert warm is None and trainer.relation is None and trainer.reference is None
        else:assert warm is not None and trainer.feature_counts['relation_calls']==1
        prime_statistics(trainer,model)
        if trainer.fishr is not None:
            x,y,rx,days,ids,ce=current_native(trainer,model,batch,20)
            trainer.loss(model,None,x,y,rx,20,4439,days=days,physical_ids=ids,native_loss=ce)
            assert trainer.fishr_calibrator.lambda_max>0
        x,y,rx,days,ids,ce=current_native(trainer,model,batch,21)
        action=trainer.loss(model,None,x,y,rx,21,4443,days=days,physical_ids=ids,native_loss=ce)
        if c['arm_plan']['paths']:
            assert action is not None and torch.isfinite(action)
            assert trainer.fit_steps['receiver']>=0
            if trainer.style is not None:
                ref_points=[m for n,m in trainer.reference.named_modules() if n.split('.')[-1]=='time_down']
                assert not ref_points[0]._forward_hooks,'Style hook leaked to D reference'
        x,y,rx,days,ids,ce=current_native(trainer,model,batch,41)
        relation=trainer.loss(model,None,x,y,rx,41,4447,days=days,physical_ids=ids,native_loss=ce)
        if relation is not None:
            norms={k:float(gradient_vector(relation,ps).norm()) for k,ps in feature_parameter_groups(model.id_backbone).items()}
            assert all(norms[k]>0 for k in ('E','G','C','time','frequency','PA')),norms
            if trainer.fishr is not None:assert trainer.feature_counts['fishr_positive_weight_calls']==1
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            (ce+relation).backward();optimizer.step()
        if trainer.style is not None:
            for _ in range(20):current_native(trainer,model,batch,41)
            assert trainer.style.snapshot()['roles']['native_L_clean']['calls']>=20
        proposal_check=check_hard_iq_retention(trainer,model,x) if arm=='LTR_S' else None
        trainer.annotate([dict(epoch=200)])
        saved=torch.load(output/arm/'auxiliary_final.pth',map_location='cpu',weights_only=False)
        assert 'feature_state' in saved and not saved['target_access']
        assert saved['feature_state']['counts']==dict(trainer.feature_counts)
        if trainer.fishr is not None:assert saved['feature_state']['fishr']['_extra_state']['buckets']
        findings.append(dict(arm=arm,status='PASS',proposal_retention=proposal_check,execution=trainer.execution()))
        if trainer.style is not None:trainer.style.detach()
        del trainer,model
    (output/'runtime_completion.json').write_text(json.dumps(dict(status='PASS',device=device,rows=findings),indent=2),encoding='utf-8')
    print('PASS all16 feature arms: relation/style/Fishr graphs, fixed E20 calibration, no U/EMA/satellite, saved state',flush=True)


class Finished(Exception):pass


def native_smoke(output,device='cpu',arms=('native','S_raw5','LTR_S_AF5','joint_noR','joint_direct','joint_shared')):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);findings=[]
    for arm in arms:
        c=make_config(arm);a=d.make_args(c,device);a.output_dir=str(output/arm);a.batch_size=128;a.eval_batch_size=128
        forbidden=output/'must_remain_absent'/arm;c=deepcopy(c);c['output_root']=str(forbidden)
        a.source_val_heavy_eval_start_epoch=9999;a.rc4_calibration_update_epochs='1'
        a.daot_diagnostic_epochs='1';a.rc4_gradient_telemetry_epochs='1';seen=[]
        with numerical_context(d.FULL_FP32_POLICY),installed(c) as n:
            names=['_build_ssdg_wisig_data','_write_ssdg_epoch_telemetry','_should_run_source_val_heavy_eval','_muse_epoch_pairs','_risk_action_step']
            originals={k:getattr(n,k) for k in names}
            loader=DataLoader(Synthetic(),batch_size=128)
            poison=DataLoader(Synthetic(hidden=True,poison=True),batch_size=128)
            def ctx(*args):
                return dict(train_loader=loader,probe_train_loader=loader,unlabeled_loader=poison,val_loader=loader,source_calibration_loader=loader,
                    named_test_loaders={},domain_label_map={r:i for i,r in enumerate((1,3,4,6,8))},num_domains=5,input_len=256,class_id_to_tx=d.CLASSES,
                    balanced_train_sampler=None,split_info=dict(labeled_size=720,unlabeled_size=720,source_val_size=720,
                    source_calibration_size=720,source_selection_size=720,mode='synthetic',named_test_meta={},source_split_receipt={}))
            build=n.build_baseline_model
            def model_build(args,dev):
                model=build(args,dev);trainer=n._feature_training_state();trainer.output_root=output/arm;trainer.set_source(loader)
                prime_statistics(trainer,model)
                return model
            n.build_baseline_model=model_build
            def callback(model,ema,x,y,rx,epoch,step,**kwargs):
                base={1:0,20:4432,21:4440,200:44392}[epoch]
                return originals['_risk_action_step'](model,ema,x,y,rx,epoch,base+step%8,**kwargs)
            def telemetry(cp,jp,records):
                originals['_write_ssdg_epoch_telemetry'](cp,jp,records);seen.append(records[-1]['epoch'])
                if seen[-1]==200:raise Finished()
            n._build_ssdg_wisig_data=ctx;n._write_ssdg_epoch_telemetry=telemetry;n._risk_action_step=callback
            n._should_run_source_val_heavy_eval=lambda *args:False
            assert sum(1 for _ in originals['_muse_epoch_pairs'](loader,poison,use_muse=False))==222
            n._muse_epoch_pairs=lambda l,u,**kw:((next(iter(l)),None) for _ in range(8))
            n.range=lambda *args:iter([1,20,21,200]) if args==(1,201) else range(*args)
            try:
                with (output/(arm+'.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                    try:n.train(a)
                    except Finished:pass
                assert seen==[1,20,21,200],seen
                assert not forbidden.exists()
                artifact=torch.load(output/arm/'auxiliary_final.pth',map_location='cpu',weights_only=False)
                assert artifact['epoch']==200 and not artifact['target_access']
                counts=artifact['feature_state']['counts']
                if arm!='native':assert counts['relation_calls']==4,counts
                if c['feature_plan']['fishr']!='none':
                    assert counts['fishr_positive_weight_calls']==2
                    assert artifact['feature_state']['calibration']['lambda_max']>0
                findings.append(dict(arm=arm,epochs=seen,status='PASS',counts=counts))
            finally:
                del n.range;n.build_baseline_model=build
                for key,value in originals.items():setattr(n,key,value)
    (output/'completion.json').write_text(json.dumps(dict(status='PASS',device=device,rows=findings,synthetic_only=True),indent=2),encoding='utf-8')
    print('PASS actual native CE and feature schedule: '+str(findings),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--native',action='store_true');p.add_argument('--focused',action='store_true')
    a=p.parse_args();(focused_checks if a.focused else native_smoke if a.native else check_runtime)(a.output,a.device)
