"""Thin native E20+10 entry and one-call ordinary-L input capture."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import csv
import gzip
import json
import math
import os
import sys
import time
import torch
from . import design as d


def clean(value):
    if torch.is_tensor(value):
        return clean(value.detach().cpu().item() if value.numel()==1 else value.detach().cpu().tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def draw_and_use(augmentor,x,labels):
    if (x.shape!=(128,2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all())
            or labels.shape!=(128,) or labels.dtype!=torch.long
            or bool(((labels<0)|(labels>5)).any())):
        raise ValueError('Only finite visible L128 IQ256 can enter the ordinary wrapper')
    raw=x.detach().clone()
    ordinary=augmentor(x,labels=labels,no_pa=False)
    if ordinary.shape!=raw.shape or ordinary.dtype!=raw.dtype or not bool(torch.isfinite(ordinary).all()) or not torch.equal(raw,x):
        raise ValueError('Original ordinary augmentation must preserve raw input')
    return raw,ordinary,dict(count=128,full_native_draws=1,original_input_unchanged=True,
        ordinary_raw_MSE=float((ordinary.detach()-raw).square().mean()),
        ordinary_changed_packets=int((ordinary.detach()!=raw).flatten(1).any(1).sum()),
        full_native_aug_computation=True)


class Wrapper:
    def __init__(self,native,records,before_return):
        object.__setattr__(self,'_native',native)
        object.__setattr__(self,'_records',records)
        object.__setattr__(self,'_before_return',before_return)
    def __getattr__(self,key):return getattr(self._native,key)
    def __setattr__(self,key,value):setattr(self._native,key,value)
    def __call__(self,x,*args,**kwargs):
        if args or set(kwargs)!={'labels','no_pa'} or kwargs['labels'] is None or kwargs['no_pa'] is not False:
            raise ValueError('Only native visible-L labels/no_paFalse augmentation is wrapped')
        raw,ordinary,record=draw_and_use(self._native,x,kwargs['labels'])
        step=len(self._records)+1
        first=self._before_return(raw,ordinary,kwargs['labels'],step)
        if first.shape!=raw.shape or first.dtype!=raw.dtype or not bool(torch.isfinite(first).all()):
            raise ValueError('Model capture must return the actual finite first-L tensor')
        self._records.append(dict(record,step=step,
            returned_raw_MSE=float((first.detach()-raw).square().mean()),
            returned_ordinary_MSE=float((first.detach()-ordinary.detach()).square().mean())))
        return first


def ordinary_factory(original,records,before_return):
    def create(*args,**kwargs):
        create.calls+=1
        if create.calls!=1:raise ValueError('Exactly one native ordinary L augmentor is supported')
        return Wrapper(original(*args,**kwargs),records,before_return)
    create.calls=0
    return create


def make_args(c,device='cuda:0'):
    d.validate(c)
    from experiments.cvs_F_nontrue20 import design as old
    a=old.make_args(d.parent_config(),device)
    a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    a.epochs=c['epochs'];a.label_epochs=c['label_epochs'];a.pseudo_epochs=c['pseudo_epochs']
    a.baseline_ckpt=str(Path(c['parent_source'])/'final_ssdg.pth')
    a.from_scratch=False;a.freeze_backbone=False;a.a1_scratch_only=False
    a.lr=c['optimizer']['lr'];a.weight_decay=c['optimizer']['weight_decay']
    a.phase1_lr_schedule='cosine';a.phase1_lr_min=c['optimizer']['lr_min']
    a.a1_ema_startup_average=False
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.seed,a.label_smoothing)!=(10,0,10,128,d.SEED,.01):
        raise ValueError('Fixed ten-epoch all-U continuation differs')
    if (a.teacher_ckpt or not a.use_unlabeled or not a.use_ema_teacher or not a.use_aug
            or a.use_concat_sat_channel_aug or a.pseudo_temporal_mode!='batch_neighbor'
            or getattr(a,'source_fold_held_rx',None) is not None or a.checkpoint_selection!='final_only'):
        raise ValueError('Original hidden-TX U/EMA/default-head/final-only native recipe changed')
    return a


def parent_payload(c):
    from .model import parent_payload as load
    return load(c)


@contextmanager
def native_default_head_only(native):
    original=native._evaluate_zid_leakage_probes
    evidence=dict(disabled_calls=0,head_fit_updates=0)
    def disabled(*args,**kwargs):
        evidence['disabled_calls']+=1
        return dict(status='DISABLED_DEFAULT_HEAD_ONLY',head_fit_updates=0,default_classifier_only=True)
    native._evaluate_zid_leakage_probes=disabled
    try:yield evidence
    finally:native._evaluate_zid_leakage_probes=original


def source_validation(model,loader,device):
    model.eval();groups={};hits=total=0
    with torch.no_grad():
        for x,y,domain,meta in loader:
            correct=model(x.to(device)).argmax(1).cpu()==y
            for rx in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==rx;g=groups.setdefault(str(rx),[0,0])
                g[0]+=int(correct[mask].sum());g[1]+=int(mask.sum())
            hits+=int(correct.sum());total+=len(y)
    rates={k:h/n for k,(h,n) in groups.items()}
    if total!=d.config(d.rows()[0])['data']['V'] or set(rates)!=set(map(str,d.FIT)):
        raise ValueError('Actual full source-V physical shape differs')
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()))


def segment_train(c,design=None,model_api=None,observer=None):
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure,batch_clean,incremental_native_writer,POLICY
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    if model_api is None:from . import model as model_api
    d.validate(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError(out)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    a=configure(make_args(c,str(device)));parent=model_api.parent_payload(c)
    if observer is None:raise ValueError('Complete native continuation observer required')
    observer.parent=parent
    expected=d.read(d.SOURCE);state={};started=time.time();steps=c['steps_per_epoch']
    # Runtime needs the original identity variant; the new fixed config is
    # consumed by model.install and never masquerades as the old scratch recipe.
    with numerical_context(d.FULL_FP32_POLICY),installed(d.parent_config()) as n,incremental_native_writer(n),native_default_head_only(n) as guard:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from scripts.train_daot_rc4_baseline import resolved_config
        build_data=n._build_ssdg_wisig_data;build_model=n.build_baseline_model
        writer=n._write_ssdg_epoch_telemetry;detach=n._detach_log_mapping;load=n.load_checkpoint
        def checked_data(*args,**kwargs):
            ctx=build_data(*args,**kwargs)
            if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids'] or len(ctx['unlabeled_loader'])!=steps or ctx['num_domains']!=15:
                raise ValueError('Original full-source roles or U-driven native update budget differs')
            observer.data(n,ctx,out);state['ctx']=ctx
            d.write(out/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
            d.write(out/'resolved_native_args.json',clean(resolved_config(a)))
            d.write(out/'actual_native_args.json',clean(resolved_config(a)))
            return ctx
        def checked_model(args,dev):
            m=model_api.install(build_model(args,dev),c,parent).to(dev)
            if actual_flags()!=d.FULL_FP32_POLICY:raise ValueError('Native numerical policy differs')
            p=out/'initial_smoke.pt'
            torch.save(dict(model=m.state_dict(),scratch_only=False,phase1_lineage=model_api.phase1_lineage(c)),p)
            m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)['model'],strict=True);m.eval()
            with torch.no_grad():z=m(torch.zeros(2,2,256,device=dev))
            if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real parent-checkpoint no-query smoke failed')
            state['model']=m;observer.initialize(m,c,out)
            d.write(out/'resolved_config.json',dict(c,config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
                commit=(d.ROOT/'release_commit.txt').read_text().strip(),torch_version=torch.__version__,backend_flags=actual_flags(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(dev),
                total_parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                execution_policy=POLICY,head_fit_updates=0,actual_feature_extractor=m._feature_learning_report))
            d.write(out/'initialization.json',dict(scratch_only=False,checkpoint_sources=c['checkpoint_sources'],
                ancestors=[c['parent_source']],source_roles='EXACT_MATCH',target_contact=False,
                parent_initialization=d.read(Path(c['parent_source'])/'initialization.json'),
                phase1_lineage=model_api.phase1_lineage(c),optimizer='reset empty AdamW/new cosine10',seed=c['model_seed']))
            print('SMOKE PASS strict own B0 FINAL20 + feature adapters; query_read=false; continuing native train',flush=True)
            return m
        def step_log(values):
            result=detach(values)
            if 'train/loss' in result:
                step=state['steps']=state.get('steps',0)+1
                if 'stream' not in state:state['stream']=gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf8',compresslevel=3)
                metrics=batch_clean(result)
                if float(metrics.get('train/pseudo_truth_available',0.) or 0.)!=0.:
                    raise ValueError('U TX labels became visible')
                state['stream'].write(json.dumps(dict(step=step,epoch=(step-1)//steps+1,
                    cumulative_step=c['parent_updates']+step,metrics=metrics),allow_nan=False)+'\n')
                if step>(c['epochs']-1)*steps:state['final_L_CE_sum']=state.get('final_L_CE_sum',0.)+float(metrics['train/loss_tx_labeled'])
                if result.get('train/skipped_nonfinite_loss',0) or result.get('train/skipped_nonfinite_grad',0):
                    state['stream'].flush();raise FloatingPointError('Native nonfinite update; preserve artifacts without retry')
            return result
        def telemetry(cp,jp,rows):
            writer(cp,jp,rows);observer.epoch(state['model'],state['ctx'],rows,out,device)
            compact=[{k:clean(v) for k,v in row.items() if not isinstance(v,(list,dict,tuple))} for row in rows]
            with (out/'epoch_metrics.jsonl').open('w',encoding='utf8') as f:
                for row in compact:f.write(json.dumps(row,allow_nan=False)+'\n')
            with (out/'epoch_metrics.csv').open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=sorted({k for r in compact for k in r}));w.writeheader();w.writerows(compact)
        def own_load(path,dev):
            return parent if Path(path)==Path(a.baseline_ckpt) else load(path,dev)
        n._build_ssdg_wisig_data=checked_data;n.build_baseline_model=checked_model
        n._write_ssdg_epoch_telemetry=telemetry;n._detach_log_mapping=step_log;n.load_checkpoint=own_load
        try:
            with observer.native_scope(n):code=n.train(a)
            if code:raise RuntimeError('Native trainer returned '+str(code))
        finally:
            n._build_ssdg_wisig_data=build_data;n.build_baseline_model=build_model
            n._write_ssdg_epoch_telemetry=writer;n._detach_log_mapping=detach;n.load_checkpoint=load
            if 'stream' in state:state['stream'].close()
        payload=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (guard['disabled_calls']!=1 or payload['epoch']!=c['epochs'] or payload['args']['baseline_ckpt']!=a.baseline_ckpt
                or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only'
                or state.get('steps',0)!=c['updates'] or payload['a1_ema_successful_updates']!=c['updates']):
            raise ValueError('Final native fixed-budget continuation contract differs')
        payload['phase1_lineage']=model_api.phase1_lineage(c)
        torch.save(payload,out/'final_ssdg.pth')
        m=state['model'];m.load_state_dict(payload['model'],strict=True)
        metrics=source_validation(m,state['ctx']['val_loader'],device)
        actual=clean(resolved_config(SimpleNamespace(**payload['args'])))
        d.write(out/'resolved_native_args.json',actual);d.write(out/'actual_native_args.json',actual)
        d.write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=c['cumulative_epochs'],segment_epoch=c['epochs'],
            parent_epoch=c['parent_epochs'],checkpoint=str(out/'final_ssdg.pth'),elapsed_seconds=time.time()-started,
            final_source_metrics=metrics,target_access=False,target_evaluated=False,head_fit_updates=0,
            checkpoint_sources=c['checkpoint_sources'],config=c,logged_steps=state['steps'],optimizer_steps=c['updates'],
            cumulative_optimizer_steps=c['cumulative_updates'],parent_optimizer_steps=c['parent_updates'],
            native_fitted_probes_disabled=True,native_disabled_probe_calls=guard['disabled_calls'],
            label_epochs=c['label_epochs'],pseudo_epochs=c['pseudo_epochs'],final_training_L_CE=state['final_L_CE_sum']/steps,
            ancestors=[c['parent_source']],initialization='own B0 FINAL20+10/resetoptimizer',phase1_lineage=payload['phase1_lineage']))
        d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'final_ssdg.pth'),
            epoch=c['cumulative_epochs'],segment_epoch=c['epochs'],selection='fixed_FINAL30',target_access=False,
            phase1_lineage=payload['phase1_lineage']))


def train(c):
    from .observer import Observer
    observer=Observer(c)
    segment_train(c,observer=observer)
    return observer.finish()
