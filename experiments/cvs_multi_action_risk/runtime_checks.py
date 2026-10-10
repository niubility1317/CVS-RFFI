"""Focused runtime checks including the actual patched native training loop."""
import argparse
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset
from . import design as d
from .runtime import OnlineTrainer, Observations, parameter_groups, gradient_vector, installed
from experiments.cvs_multi_state_action.model import intermediate, classify_intermediate
from experiments.cvs_equivariant_identity.precision import numerical_context


class Synthetic(Dataset):
    def __init__(self, hidden=False, poison=False): self.hidden=hidden; self.poison=poison
    def __len__(self): return 32
    def __getitem__(self, i):
        if self.poison: raise AssertionError('Pure CE baseline fetched source U')
        x=torch.randn(2,256,generator=torch.Generator().manual_seed(800+i))
        if self.hidden:return x,-1,-1,{}
        y=i%6; rx=(i//6)%2
        return x,y,rx,dict(rx_i=rx,day_i=1,eq_i=1,sig_i=i,base_index=i,tx_i=y)


def check_runtime(output,device='cpu'):
    torch.set_num_threads(2); output=Path(output);output.mkdir(parents=True,exist_ok=False)
    observations=Observations();observations.add(dict(gradient=.56,dense=1.))
    for _ in range(55):observations.add(dict(dense=1.))
    result=observations.finish();assert result['gradient']==.56 and result['metric_observation_counts']['gradient']==1
    assert observations.finish()['gradient'] is None
    findings=[]
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));a=d.make_args(c,device);a.num_domains=2;a.input_len=256
        with installed(c,training=False) as native:
            model=native.build_baseline_model(a,torch.device(device))
        trainer=OnlineTrainer(c,model);trainer.output_root=output/arm
        trainer.u_loader=DataLoader(Synthetic(hidden=True),batch_size=32) if c['arm_plan'].get('source_u_fit') else None
        x,y,rx,meta=next(iter(DataLoader(Synthetic(),batch_size=32)))
        x=x.to(device);y=y.to(device);rx=rx.to(device);days=meta['day_i'].to(device)
        ids=['runtime-'+str(i) for i in range(len(x))]
        model.train();native_loss=torch.nn.functional.cross_entropy(model(x),y)
        assert trainer.loss(model,None,x,y,rx,20,0,days=days,physical_ids=ids,native_loss=native_loss) is None
        loss=trainer.loss(model,None,x,y,rx,21,0,days=days,physical_ids=ids,native_loss=native_loss)
        if arm=='native':assert loss is None and trainer.reference is None and trainer.u_iterator is None
        else:
            assert torch.isfinite(loss) and (trainer.reference is not None)==bool(c['arm_plan']['paths'])
            groups=parameter_groups(model.id_backbone)
            eg=gradient_vector(loss,groups['E']).norm();gg=gradient_vector(loss,groups['G_C']).norm()
            if arm=='exact_g':assert eg==0 and gg>0
            elif c['arm_plan']['paths'] or c['arm_plan']['replacements']:assert eg>0 and gg>0
            if c['arm_plan'].get('source_u_fit'):assert trainer.exposure_totals['source_U_action_only_packets']==32
            else:assert trainer.u_iterator is None
            for kind in c['arm_plan']['paths']:
                if kind!='receiver':assert trainer.fit_steps[kind]==(2 if c['arm_plan'].get('source_u_fit') else 1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            (native_loss+loss).backward();trainer.optimizer_snapshot(model,True,0);optimizer.step();trainer.optimizer_snapshot(model,False,0)
            assert trainer.last['E_optimizer_actual_update_norm']>0
            if arm=='LT':
                later_native=torch.nn.functional.cross_entropy(model(x),y)
                later=trainer.loss(model,None,x,y,rx,21,4,days=days,physical_ids=ids,native_loss=later_native)
                assert torch.isfinite(later) and trainer.last['chain_order']=='TL'
                assert trainer.exposure_totals['linear_conditional_edge_packets']>0
                assert trainer.exposure_totals['temporal_conditional_edge_packets']>0
            rows=[dict(epoch=21)];trainer.annotate(rows)
            for key,value in rows[0].items():
                if key.endswith('_weighted_ratio') and value is not None:assert value>=0
        findings.append(dict(arm=arm,status='PASS',execution=trainer.execution()))
        del trainer,model
    (output/'runtime_completion.json').write_text(json.dumps(dict(status='PASS',device=device,rows=findings),indent=2),encoding='utf-8')
    print('PASS runtime: all16 arms, pure CE no EMA, fixed weighting, real E vs exact G, U isolation, per-metric counts, actual optimizer update')
    return findings


class Finished(Exception):pass


def native_smoke(output,device='cpu',arms=('native','LTR','LT_label_free_U')):
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    findings=[]
    for arm in arms:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));a=d.make_args(c,device)
        a.output_dir=str(output/arm);a.batch_size=32;a.eval_batch_size=32
        # Synthetic-only trap: any fallback to registered output would create
        # this distinct path. No formal source path is writable by the smoke.
        forbidden=output/'must_remain_absent'/arm
        c=deepcopy(c);c['output_root']=str(forbidden)
        a.source_val_heavy_eval_start_epoch=9999;a.rc4_calibration_update_epochs='1'
        a.daot_diagnostic_epochs='1';a.rc4_gradient_telemetry_epochs='1'
        seen=[]
        with numerical_context(d.FULL_FP32_POLICY),installed(c) as n:
            names=['_build_ssdg_wisig_data','_write_ssdg_epoch_telemetry','_should_run_source_val_heavy_eval','_muse_epoch_pairs']
            originals={k:getattr(n,k) for k in names}
            def ctx(*args):
                l=DataLoader(Synthetic(),batch_size=32)
                u=DataLoader(Synthetic(hidden=True,poison=not c['arm_plan'].get('source_u_fit')),batch_size=32)
                # Preserve runtime data hook's U-only assignment in synthetic context.
                # The native builder is bypassed only to avoid real data access.
                return dict(train_loader=l,probe_train_loader=l,unlabeled_loader=u,val_loader=l,source_calibration_loader=l,
                    named_test_loaders={},domain_label_map={0:0,1:1},num_domains=2,input_len=256,class_id_to_tx=d.CLASSES,
                    balanced_train_sampler=None,split_info=dict(labeled_size=32,unlabeled_size=32,source_val_size=32,
                    source_calibration_size=32,source_selection_size=32,mode='synthetic',named_test_meta={},source_split_receipt={}))
            # Runtime callback closes over trainer. Expose U only to explicit arm.
            build=n.build_baseline_model
            def model_build(args,dev):
                model=build(args,dev)
                callback=n._risk_action_step
                state=next(cell.cell_contents for cell in callback.__closure__ if isinstance(cell.cell_contents,dict) and 'trainer' in cell.cell_contents)
                state['trainer'].output_root=output/arm
                if c['arm_plan'].get('source_u_fit'):
                    state['trainer'].u_loader=DataLoader(Synthetic(hidden=True),batch_size=32)
                return model
            n.build_baseline_model=model_build
            def telemetry(cp,jp,records):
                originals['_write_ssdg_epoch_telemetry'](cp,jp,records)
                seen.append(records[-1]['epoch'])
                if seen[-1]==200:raise Finished()
            n._build_ssdg_wisig_data=ctx;n._write_ssdg_epoch_telemetry=telemetry
            n._should_run_source_val_heavy_eval=lambda *args:False
            # Check fixed222 independently, then shorten synthetic native loop.
            l=DataLoader(Synthetic(),batch_size=32);poison=DataLoader(Synthetic(poison=True),batch_size=32)
            assert sum(1 for _ in originals['_muse_epoch_pairs'](l,poison,use_muse=False))==222
            n._muse_epoch_pairs=lambda l,u,**kwargs:((next(iter(l)),None) for _ in range(4))
            n.range=lambda *args:iter([1,21,131,200]) if args==(1,201) else range(*args)
            try:
                with (output/(arm+'.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                    try:n.train(a)
                    except Finished:pass
                assert seen==[1,21,131,200],seen
                assert not forbidden.exists(), 'Synthetic trainer wrote outside smoke arm output'
                assert (output/arm/'auxiliary_final.pth').is_file()
                assert (output/arm/'auxiliary_cost.json').is_file()
                artifact=torch.load(output/arm/'auxiliary_final.pth',map_location='cpu',weights_only=False)
                assert artifact['epoch']==200 and artifact['target_access'] is False
                findings.append(dict(arm=arm,epochs=seen,status='PASS'))
            finally:
                del n.range;n.build_baseline_model=build
                for key,value in originals.items():setattr(n,key,value)
    (output/'completion.json').write_text(json.dumps(dict(status='PASS',device=device,rows=findings,target_access=False,formal_weights=False),indent=2),encoding='utf-8')
    print('PASS actual native pureCE warmup and late epochs: '+str(findings))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--native',action='store_true')
    a=p.parse_args();(native_smoke if a.native else check_runtime)(a.output,a.device)
