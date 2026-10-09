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
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT)]
import design as newdesign
sys.path[:0]=[str(newdesign.WORKER),str(newdesign.WORKER/'code')]
import torch
from experiments.cvs_phase1_stack.design import *
ROOT=Path(__file__).resolve().parent
from experiments.cvs_phase1_stack.runtime import installed as original_installed
from contextlib import contextmanager
from torch.utils.data import DataLoader

@contextmanager
def installed(c):
    with original_installed(c) as n:
        n._muse_epoch_pairs=lambda l,u,**kw: ((b,None) for b in l)
        yield n

validate=newdesign.validate
make_args=newdesign.make_args
from experiments.cvs_phase1_stack.fast_execution import configure,batch_clean,incremental_native_writer,POLICY
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags


def clean(v):
    if torch.is_tensor(v):return clean(v.detach().cpu().item() if v.numel()==1 else v.detach().cpu().tolist())
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v


def require_budget(logged,successful):
    if logged!=EXPECTED_UPDATES or successful!=EXPECTED_UPDATES:
        raise ValueError(f'Incomplete update budget: logged={logged}, successful={successful}, expected={EXPECTED_UPDATES}')


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


def train(c):
    global EXPECTED_UPDATES
    EXPECTED_UPDATES=200*c['steps_per_epoch']
    validate(c);out=Path(c['output_root'])
    if c['stage']=='r2':raise ValueError('Recovery reuses completed R2 artifacts; never retrain them')
    if out.exists():raise FileExistsError(out)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    a=configure(make_args(c));expected=read(SOURCE);state={};started=time.time()
    with numerical_context(FULL_FP32_POLICY),installed(c) as n,incremental_native_writer(n):
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from scripts.train_daot_rc4_baseline import resolved_config
        build_data=n._build_ssdg_wisig_data;build_model=n.build_baseline_model;writer=n._write_ssdg_epoch_telemetry;detach=n._detach_log_mapping
        def checked_data(*args,**kw):
            ctx=build_data(*args,**kw)
            if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Physical source roles/target construction mismatch')
            if len(ctx['unlabeled_loader'])!=222:raise ValueError('Unexpected U-based step count')
            old=ctx['train_loader']
            if old.batch_sampler.__class__.__name__!='BatchSampler':raise ValueError('Unexpected balanced sampler')
            ctx['train_loader']=DataLoader(old.dataset,batch_size=c['batch_size'],shuffle=True,drop_last=False,num_workers=0,pin_memory=True)
            if len(old.dataset)!=6300 or len(ctx['train_loader'])!=c['steps_per_epoch']:raise ValueError('Single-pass count differs')
            state['ctx']=ctx
            write(out/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
            write(out/'resolved_native_args.json',clean(resolved_config(a)))
            return ctx
        def checked_model(args,dev):
            m=build_model(args,dev)
            if actual_flags()!=FULL_FP32_POLICY:raise ValueError('Numerical policy changed inside native trainer')
            p=out/'initial_smoke.pt';torch.save(dict(model=m.state_dict(),scratch_only=True),p)
            m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)['model'],strict=True);m.eval()
            with torch.no_grad():z=m(torch.zeros(2,2,256,device=dev))
            if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Initial checkpoint smoke failed')
            state['model']=m
            torch.cuda.reset_peak_memory_stats(dev)
            write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
                commit=(ROOT/'release_commit.txt').read_text().strip(),torch_version=torch.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),total_parameters=sum(p.numel() for p in m.parameters()),
                hardware=torch.cuda.get_device_name(dev),backend_flags=actual_flags(),native_args_ref=str(out/'resolved_native_args.json'),execution_policy=POLICY))
            write(out/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']))
            print('SMOKE PASS own scratch checkpoint; query_read=false',flush=True)
            return m
        def step_log(values):
            result=detach(values)
            if 'train/loss' in result:
                state['steps']=state.get('steps',0)+1
                if 'step_file' not in state:state['step_file']=gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
                state['step_file'].write(json.dumps(dict(step=state['steps'],epoch=(state['steps']-1)//c['steps_per_epoch']+1,metrics=batch_clean(result)),allow_nan=False)+'\n')
                if result.get('train/skipped_nonfinite_loss',0) or result.get('train/skipped_nonfinite_grad',0):
                    state['step_file'].flush()
                    raise FloatingPointError('Native nonfinite loss/gradient: preserve artifacts; no automatic retry')
            return result
        def telemetry(cp,jp,rows):
            writer(cp,jp,rows)
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
            code=n.train(a)
            if code:raise RuntimeError('Native train returned '+str(code))
        finally:
            n._build_ssdg_wisig_data=build_data;n.build_baseline_model=build_model;n._write_ssdg_epoch_telemetry=writer;n._detach_log_mapping=detach
            if 'step_file' in state:state['step_file'].close()
        payload=torch.load(out/'final_ssdg.pth',map_location=device,weights_only=False)
        if payload['epoch']!=200 or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Final checkpoint contract mismatch')
        require_budget(state.get('steps',0),payload['a1_ema_successful_updates'])
        m=state['model'];m.load_state_dict(payload['model'],strict=True)
        write(out/'resolved_native_args.json',clean(resolved_config(SimpleNamespace(**payload['args']))))
        metrics=source_validation(m,state['ctx']['val_loader'],device)
        write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,checkpoint=str(out/'final_ssdg.pth'),
            elapsed_seconds=time.time()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),final_source_metrics=metrics,target_access=False,target_evaluated=False,
            checkpoint_sources=[],config=c,logged_steps=state.get('steps',0),optimizer_steps=payload['a1_ema_successful_updates']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();train(read(a.config))
