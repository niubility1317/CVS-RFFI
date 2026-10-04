"""Synthetic native-loop throughput and exact state comparison, no IQ/truth."""
import argparse
import contextlib
import gzip
import json
import random
import time
from pathlib import Path
import sys
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from experiments.cvs_phase1_stack.design import *
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_phase1_stack.source import clean
from experiments.cvs_phase1_stack.fast_execution import batch_clean,incremental_native_writer
from experiments.cvs_equivariant_identity.precision import numerical_context


class Synthetic(Dataset):
    def __init__(self,size,hidden=False,muse=False):self.size=size;self.hidden=hidden;self.muse=muse
    def __len__(self):return self.size
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(700+i);x=torch.randn(2,256,generator=g);y=i%6;d=(i//6)%15
        meta=dict(rx_i=d//3,day_i=d%3+1,eq_i=1,sig_i=i,base_index=i)
        if self.muse:return x,d,dict(meta,tx_label_visible=False)
        return x,-1 if self.hidden else y,d,dict(meta,**({} if self.hidden else {'tx_i':y}))


class Finished(Exception):pass


def run_case(out,stage,arm,fast,steps,profile,val_size=256,deterministic=False,device='cuda:0',batch_size=128):
    row=next(r for r in rows() if r['stage']==stage and r['arm']==arm)
    parent=['leo','twostage','ema','pseudo']+ (DG if stage in ['r4','r5','r6'] else [])+(OPEN if stage in ['r5','r6'] else [])
    c=config(row,parent);a=make_args(c,device);a.output_dir=str(out);a.batch_size=batch_size;a.eval_batch_size=256
    a.a1_gradient_snapshot=fast;a.source_validation_reuse=fast
    a.source_val_heavy_eval_start_epoch=1;a.source_val_heavy_eval_interval=1;a.source_val_heavy_eval_final_window=0
    a.rc4_calibration_update_epochs='91';a.daot_diagnostic_epochs='91';a.rc4_gradient_telemetry_epochs='91'
    if profile:a.execution_profile_epoch=131;a.execution_profile_start=3;a.execution_profile_steps=2
    epochs=[91,131,181];seen=[];state={};times=[];losses=[];logfile=None
    with numerical_context(FULL_FP32_POLICY),installed(c) as n, (incremental_native_writer(n) if fast else contextlib.nullcontext()):
        if deterministic:
            torch.use_deterministic_algorithms(True)
        old={k:getattr(n,k) for k in ['_build_ssdg_wisig_data','_write_ssdg_epoch_telemetry','build_baseline_model','_detach_log_mapping']}
        oldstep=torch.optim.AdamW.step
        def optimizer_step(self,*args,**kw):state['optimizer']=self;return oldstep(self,*args,**kw)
        torch.optim.AdamW.step=optimizer_step
        def ctx(*args):
            ubatch=256 if device.startswith('cuda') else batch_size
            l=DataLoader(Synthetic(batch_size*4),batch_size=batch_size);u=DataLoader(Synthetic(ubatch*steps,True,a.use_muse_ssdg),batch_size=ubatch)
            v=DataLoader(Synthetic(val_size),batch_size=256)
            return dict(train_loader=l,probe_train_loader=l,unlabeled_loader=u,val_loader=v,source_calibration_loader=v,named_test_loaders={},domain_label_map={i:i for i in range(15)},num_domains=15,input_len=256,class_id_to_tx=CLASSES,balanced_train_sampler=None,split_info=dict(labeled_size=512,unlabeled_size=256*steps,source_val_size=256,source_calibration_size=256,source_selection_size=256,mode='synthetic',named_test_meta={},source_split_receipt={}))
        def model(*args,**kw):m=old['build_baseline_model'](*args,**kw);state['model']=m;return m
        def detach(values):
            result=old['_detach_log_mapping'](values)
            if 'train/loss' in result:
                tic=time.perf_counter();v=(batch_clean if fast else clean)(result);logfile.write(json.dumps(v,allow_nan=False)+'\n');times.append(time.perf_counter()-tic);losses.append(v['train/loss'])
            return result
        def telemetry(cp,jp,records):
            old['_write_ssdg_epoch_telemetry'](cp,jp,records);seen.append(records[-1])
            if seen[-1]['epoch']==181:raise Finished()
        n._build_ssdg_wisig_data=ctx;n.build_baseline_model=model;n._detach_log_mapping=detach;n._write_ssdg_epoch_telemetry=telemetry
        n.range=lambda *x:iter(epochs) if x==(1,201) else range(*x)
        try:
            out.mkdir(parents=True,exist_ok=False)
            with (out/'stdout.log').open('x',encoding='utf-8') as textlog,gzip.open(out/'steps.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as logfile,contextlib.redirect_stdout(textlog):
                started=time.perf_counter()
                try:n.train(a)
                except Finished:pass
                if device.startswith('cuda'):torch.cuda.synchronize()
                wall=time.perf_counter()-started
            assert [r['epoch'] for r in seen]==epochs
            torch.save(dict(model=state['model'].state_dict(),optimizer=state['optimizer'].state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),losses=losses),out/'state.pt')
            result=dict(stage=stage,arm=arm,fast=fast,wall_seconds=wall,epochs=[dict(epoch=r['epoch'],seconds=r['epoch_time_s']) for r in seen],steps=len(times),logging_seconds=sum(times),hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'local CPU',torch=torch.__version__,profile=profile,synthetic=True,target_access=False,val_size=val_size,deterministic_diagnostic_only=deterministic,device=device,batch_size=batch_size)
            write(out/'result.json',result);return result
        finally:
            torch.optim.AdamW.step=oldstep;del n.range
            for k,v in old.items():setattr(n,k,v)


def compare(a,b):
    x=torch.load(a,map_location='cpu',weights_only=False);y=torch.load(b,map_location='cpu',weights_only=False);differences=[]
    def equal(x,y,path):
        if torch.is_tensor(x):
            if not torch.equal(x,y):differences.append(dict(path=path,max_abs=float((x.double()-y.double()).abs().max())))
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):differences.append(dict(path=path))
        elif isinstance(x,dict):
            if x.keys()!=y.keys():differences.append(dict(path=path,keys_differ=True));return
            for k in x:equal(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):differences.append(dict(path=path,length_differ=True));return
            for i,(u,v) in enumerate(zip(x,y)):equal(u,v,path+'.'+str(i))
        elif x!=y:differences.append(dict(path=path))
    equal(x,y,'state');return dict(exact_equal=not differences,differences=differences[:30],difference_count=len(differences))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--arm',required=True);p.add_argument('--fast',action='store_true');p.add_argument('--steps',type=int,default=8);p.add_argument('--profile',action='store_true');p.add_argument('--compare',type=Path);p.add_argument('--val-size',type=int,default=256);p.add_argument('--deterministic',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    torch.set_num_threads(2)
    if a.compare:print(json.dumps(compare(a.output/'state.pt',a.compare/'state.pt')))
    else:print(json.dumps(run_case(a.output,a.stage,a.arm,a.fast,a.steps,a.profile,a.val_size,a.deterministic,a.device,a.batch_size)))
