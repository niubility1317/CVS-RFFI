"""Frozen SIG_TIME/default-head source study; no target loader or head fit."""
import argparse,gzip,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .channel import audit,imports,numpy_iq,tensor_iq
from .controls import compare
from .panel import load,metadata

def frozen_model(seed=d.SEEDS[0],device='cpu'):
    from experiments.cvs_signature_stats50.predict import own_model
    from experiments.cvs_moment_area50 import model as api
    pd,c=d.diagnostic_config(seed)
    return own_model(c,device,design=pd,build_fn=api.build)

def run():
    out=d.BASE/'source_diagnostic';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);ch,r=imports();model=frozen_model(device='cpu');model.requires_grad_(True)
    with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real checkpoint no-query smoke failed')
    before={k:v.clone() for k,v in model.state_dict().items()};panel=load();configs=__import__('experiments.cvs_urban_sigtime.channel',fromlist=['configurations']).configurations()
    started=time.perf_counter();groups=defaultdict(list);all_records=[];gradient_records=[]
    parameters=[dict(model.named_parameters())[p] for p in __import__('experiments.cvs_urban_sigtime.observer',fromlist=['PARAMETERS']).PARAMETERS]
    with gzip.open(out/'sample_diagnostics.jsonl.gz','xt',encoding='utf-8') as f:
        for begin in range(0,720,90):
            x=panel['x'][begin:begin+90];y=panel['y'][begin:begin+90]
            clean=model.forward_identity_only(x);clean_correct=clean['tx_logits'].argmax(1)==y
            clean_ce=F.cross_entropy(clean['tx_logits'],y,label_smoothing=.01)
            cg=torch.autograd.grad(clean_ce,parameters,allow_unused=True,retain_graph=True)
            raw=numpy_iq(x)
            for scene,cfg in configs.items():
                batches=defaultdict(list);records=defaultdict(list)
                for i,pid in enumerate(panel['ids'][begin:begin+90]):
                    record,_=ch.normalize_input(raw[i,0]+1j*raw[i,1]);seed=ch.stable_seed(2026101403,'source_diagnostic',scene,pid)
                    session=f"rx{int(panel['rx'][begin+i])}:day{int(panel['day'][begin+i])}:eq1"
                    values,details,original=compare(record,cfg,seed,ch.receiver_for_session(cfg,2026092701,session))
                    for name,value in values.items():batches[name].append(np.stack((value.real,value.imag)));records[name].append(details[name])
                for name,values in batches.items():
                    output=model.forward_identity_only(tensor_iq(np.stack(values),'cpu'));logits=output['tx_logits']
                    ce=F.cross_entropy(logits,y,label_smoothing=.01)
                    margins=(logits.gather(1,y[:,None]).squeeze(1)-logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values)/30
                    sg=torch.autograd.grad(ce,parameters,allow_unused=True)
                    grads=[]
                    for p,a,b in zip(parameters,cg,sg):
                        if a is None:a=torch.zeros_like(p)
                        if b is None:b=torch.zeros_like(p)
                        aa=float(a.square().sum());bb=float(b.square().sum());dot=float((a*b).sum())
                        grads.append(dict(clean_norm=aa**.5,satellite_norm=bb**.5,cosine=dot/(aa*bb)**.5 if aa*bb else None))
                    gradient_records.append(dict(scene=scene,condition=name,batch=begin//90,gradients=grads))
                    for i,m in enumerate(records[name]):
                        baseline=records['full'][i]
                        item=dict(scene=scene,condition=name,physical_id=panel['ids'][begin+i],RX=int(panel['rx'][begin+i]),
                            baseline_state=baseline['state'],baseline_lock=baseline['receiver_lock_status'],
                            TX=int(y[i]),clean_correct=bool(clean_correct[i]),correct=bool(logits[i].argmax()==y[i]),
                            margin=float(margins[i]),feature_cosine=float(F.cosine_similarity(clean['z_id'][i:i+1].detach(),output['z_id'][i:i+1].detach())),**m)
                        f.write(json.dumps(item,allow_nan=False)+'\n');groups[(scene,name,baseline['state'],baseline['receiver_lock_status'])].append(item)
                print(json.dumps(dict(diagnostic_scene=scene,source_processed=begin+90,target_read=False)),flush=True)
            del clean,clean_ce,cg
    for key,items in sorted(groups.items()):
        correct=[v for v in items if v['clean_correct']]
        all_records.append(dict(scene=key[0],condition=key[1],state=key[2],lock=key[3],count=len(items),
            accuracy=sum(v['correct'] for v in items)/len(items),margin=sum(v['margin'] for v in items)/len(items),
            fixed_reference_clean_correct_count=len(correct),retention=sum(v['correct'] for v in correct)/len(correct) if correct else None,
            white_actual_difference_db_mean=sum(v['waveform_minus_white_power_db'] for v in items)/len(items),
            receiver_state_changes=sum(v['receiver_lock_status']!=v['baseline_lock'] for v in items)))
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen diagnostic mutated model state or accumulated gradients')
    result=dict(status='ANALYZED',source_only=True,source_count=720,views=6,conditions=11,optimizer_updates=0,
        head_fit_updates=0,target_read=False,seconds=time.perf_counter()-started,actual_execution=audit(),
        summary=all_records,gradients=gradient_records,source_checkpoint=d.diagnostic_config()[1]['output_root']+'/final_ssdg.pth',
        model_state_unchanged=True,total_SNR_known=False,formal_test_IQ_changed=False)
    d.write(out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('summary','gradients','actual_execution')}),flush=True)
    return result

if __name__=='__main__':run()
