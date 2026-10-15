import argparse
import csv
import json
import gzip
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import torch
from experiments.cvs_urban_sigtime50.design import *
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_phase1_stack.fast_execution import configure,batch_clean,incremental_native_writer,POLICY
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags


def clean(v):
    if torch.is_tensor(v):return clean(v.detach().cpu().item() if v.numel()==1 else v.detach().cpu().tolist())
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v


def require_budget(logged,successful):
    if logged!=11100 or successful!=11100:
        raise ValueError(f'Incomplete update budget: logged={logged}, successful={successful}, expected=11100')


def source_validation(model,loader,device):
    model.eval();groups={};hits=total=0
    with torch.no_grad():
        for batch in loader:
            x,y,domain,meta=batch
            logits=model(x.to(device));ok=(logits.argmax(1).cpu()==y)
            for rx in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==rx;g=groups.setdefault(str(rx),[0,0]);g[0]+=int(ok[mask].sum());g[1]+=int(mask.sum())
            hits+=int(ok.sum());total+=len(y)
    rates={k:h/n for k,(h,n) in groups.items()}
    if total!=27000 or set(rates)!=set(map(str,[1,3,4,6,8])):raise ValueError('Source V physical shape mismatch')
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()))


from contextlib import contextmanager

@contextmanager
def native_default_head_only(native):
    """Disable native post-training fitted RX/day/channel heads for this run."""
    original=native._evaluate_zid_leakage_probes
    evidence=dict(active=True,disabled_calls=0,head_fit_updates=0)
    def disabled(*args,**kwargs):
        evidence['disabled_calls']+=1
        return dict(status='DISABLED_DEFAULT_HEAD_ONLY',reason='User prohibits separate fitted classifiers',head_fit_updates=0,default_classifier_only=True)
    native._evaluate_zid_leakage_probes=disabled
    try:yield evidence
    finally:
        native._evaluate_zid_leakage_probes=original
        evidence['active']=False


def train(c,design=None,model_api=None,observer=None,data_adapter=None):
    if design is not None:
        validate,make_args,read,write,SOURCE,FULL_FP32_POLICY,ROOT=(getattr(design,k) for k in ("validate","make_args","read","write","SOURCE","FULL_FP32_POLICY","ROOT"))
    else:
        from . import design as design
        validate,make_args,read,write,SOURCE,FULL_FP32_POLICY,ROOT=(getattr(design,k) for k in ("validate","make_args","read","write","SOURCE","FULL_FP32_POLICY","ROOT"))
    if model_api is None:
        from . import model as model_api
    validate(c);out=Path(c['output_root'])
    if c['baseline_reuse']:raise ValueError('Reference control cannot be retrained')
    if out.exists():raise FileExistsError(out)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    a=configure(make_args(c));expected=read(SOURCE);state={};started=time.time()
    with numerical_context(FULL_FP32_POLICY),installed(c) as n,incremental_native_writer(n),native_default_head_only(n) as diagnostic_guard:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from scripts.train_daot_rc4_baseline import resolved_config
        build_data=n._build_ssdg_wisig_data;build_model=n.build_baseline_model;writer=n._write_ssdg_epoch_telemetry;detach=n._detach_log_mapping
        def checked_data(*args,**kw):
            ctx=build_data(*args,**kw)
            if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Physical source roles/target construction mismatch')
            if len(ctx['unlabeled_loader'])!=222:raise ValueError('Unexpected U-based step count')
            source_domain_count=model_api.source_domain_count
            if ctx['num_domains']!=source_domain_count(a):raise ValueError('Historical RX/day source model dimensions differ')
            if data_adapter is not None:
                inner=data_adapter(ctx,a,n,device)
                write(out/'inner_source_contract.json',inner)
            state['ctx']=ctx
            write(out/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
            write(out/'resolved_native_args.json',clean(resolved_config(a)));write(out/'actual_native_args.json',clean(resolved_config(a)))
            return ctx
        def checked_model(args,dev):
            m=build_model(args,dev)
            from experiments.cvs_source_identity_readout.input_policy import apply
            install=model_api.install
            m._identity_input_policy=apply(m);m=install(m,c).to(dev)
            if actual_flags()!=FULL_FP32_POLICY:raise ValueError('Numerical policy changed inside native trainer')
            p=out/'initial_smoke.pt';torch.save(dict(model=m.state_dict(),scratch_only=True),p)
            m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)['model'],strict=True);m.eval()
            with torch.no_grad():z=m(torch.zeros(2,2,256,device=dev))
            if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Initial checkpoint smoke failed')
            state['model']=m
            if observer is not None:observer.initialize(m,c,out)
            write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
                commit=(ROOT/'release_commit.txt').read_text().strip(),torch_version=torch.__version__,native_fitted_probes_disabled=diagnostic_guard['active'],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),total_parameters=sum(p.numel() for p in m.parameters()),
                hardware=torch.cuda.get_device_name(dev),backend_flags=actual_flags(),native_args_ref=str(out/'resolved_native_args.json'),execution_policy=POLICY,config=c,actual_feature_extractor=m._feature_learning_report,L_context_steps=m._feature_l_steps,mask_counts=m._feature_mask_counts,identity_input_policy=m._identity_input_policy,baseline_recipe='historicaloldgate+EMA+cos',label_epochs=33,pseudo_epochs=17))
            write(out/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']))
            print('SMOKE PASS own scratch checkpoint; query_read=false',flush=True)
            return m
        def step_log(values):
            result=detach(values)
            if 'train/loss' in result:
                state['steps']=state.get('steps',0)+1
                if 'step_file' not in state:state['step_file']=gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
                metrics=batch_clean(result)
                if state['steps']>7326 and float(metrics.get('train/pseudo_truth_available',0.) or 0.)!=0:raise ValueError('Hidden U TX became available during pseudo training')
                state['step_file'].write(json.dumps(dict(step=state['steps'],epoch=(state['steps']-1)//222+1,metrics=metrics),allow_nan=False)+'\n')
                if state['steps']>10878:state['final_L_CE_sum']=state.get('final_L_CE_sum',0.)+float(metrics['train/loss_tx_labeled'])
                if result.get('train/skipped_nonfinite_loss',0) or result.get('train/skipped_nonfinite_grad',0):
                    state['step_file'].flush()
                    raise FloatingPointError('Native nonfinite loss/gradient: preserve artifacts; no automatic retry')
            return result
        def telemetry(cp,jp,rows):
            writer(cp,jp,rows)
            if observer is not None:observer.epoch(state["model"],state["ctx"],rows,out,device)
            # Native complete telemetry is retained; scalar-only companion is
            # rewritten once per epoch, with nonfinite/missing values as null.
            compact=[{k:clean(v) for k,v in row.items() if not isinstance(v,(list,dict,tuple))} for row in rows]
            with (out/'epoch_metrics.jsonl').open('w',encoding='utf-8') as f:
                for row in compact:f.write(json.dumps(row,allow_nan=False)+'\n')
            fields=sorted({k for r in compact for k in r})
            with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(compact)
        n._build_ssdg_wisig_data=checked_data;n.build_baseline_model=checked_model;n._write_ssdg_epoch_telemetry=telemetry;n._detach_log_mapping=step_log
        try:
            if observer is None:code=n.train(a)
            else:
                with observer.native_scope(n):code=n.train(a)
            if code:raise RuntimeError('Native train returned '+str(code))
        finally:
            n._build_ssdg_wisig_data=build_data;n.build_baseline_model=build_model;n._write_ssdg_epoch_telemetry=writer;n._detach_log_mapping=detach
            if 'step_file' in state:state['step_file'].close()
        if diagnostic_guard['disabled_calls']!=1 or diagnostic_guard['head_fit_updates']!=0:raise ValueError('Native fitted classifier closure was not disabled')
        payload=torch.load(out/'final_ssdg.pth',map_location=device,weights_only=False)
        if payload['epoch']!=50 or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Final checkpoint contract mismatch')
        require_budget(state.get('steps',0),payload['a1_ema_successful_updates'])
        m=state['model'];m.load_state_dict(payload['model'],strict=True)
        write(out/'resolved_native_args.json',clean(resolved_config(SimpleNamespace(**payload['args']))));write(out/'actual_native_args.json',clean(resolved_config(SimpleNamespace(**payload['args']))))
        if m._feature_l_steps!=11100:raise ValueError('Real L context did not match optimizer budget')
        metrics=source_validation(m,state['ctx']['val_loader'],device)
        write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=50,checkpoint=str(out/'final_ssdg.pth'),
            elapsed_seconds=time.time()-started,final_source_metrics=metrics,target_access=False,target_evaluated=False,native_fitted_probes_disabled=True,native_disabled_probe_calls=diagnostic_guard['disabled_calls'],head_fit_updates=0,
            checkpoint_sources=[],config=c,logged_steps=state.get('steps',0),optimizer_steps=payload['a1_ema_successful_updates'],feature_extractor=m._feature_learning_report,L_context_steps=m._feature_l_steps,mask_counts=m._feature_mask_counts,identity_input_policy=m._identity_input_policy,baseline_recipe='historicaloldgate+EMA+cos',label_epochs=33,pseudo_epochs=17,final_training_L_CE=state['final_L_CE_sum']/222,ancestors=[],initialization='scratch'))
        write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'final_ssdg.pth'),epoch=50,selection='final_only',target_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();train(read(a.config))
