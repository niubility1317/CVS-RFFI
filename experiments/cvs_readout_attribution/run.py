"""Evaluate four fixed readout interventions without changing any model state."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

VARIANTS=('readout_attention','readout_complex_attention')
CONDITIONS=('all_on','time_off','behavior_off','both_off')
SOURCE_COMMIT='6ca7a522bb4be34993f442129870e68cec61bbc7'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
SOURCE_RUN='20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01'
RUN='20261003-diagnostic-cvs-readout-attribution-source-manysig-m8-r01'
OWNER='codex/root/readout-source-attribution-20261003'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def validate_config(c):
    keys={'source_release','source_commit','source_output','variant','model_seed','output_root','conditions','role','launch_owner'}
    if set(c)!=keys or c['variant'] not in VARIANTS or c['role']!='V' or c['conditions']!=list(CONDITIONS):
        raise ValueError('Only registered frozen source V conditions allowed')
    if c['source_commit']!=SOURCE_COMMIT or c['model_seed'] not in range(2026092701,2026092705) or c['launch_owner']!=OWNER:
        raise ValueError('Unregistered checkpoint or owner')
    rid=c['variant']+'-s'+str(c['model_seed'])
    if (c['source_release']!=PROJECT+'/releases/cvs_neural_readout_identity_20261003_r01'
            or c['source_output']!=PROJECT+'/runs/'+SOURCE_RUN+'/'+rid+'/source'
            or c['output_root']!=PROJECT+'/runs/'+RUN+'/'+rid):
        raise ValueError('Source/output path differs')
    return c

@torch.no_grad()
def components(model,x):
    """Temporary output hooks preserve trained skip, downstream head and state."""
    scores={};features={}
    for condition in CONDITIONS:
        handles=[]
        def hook_for(path):
            def hook(module,args,output):
                skip=module.skip(args[0])
                if condition=='all_on':
                    features[path+'_skip']=skip.detach()
                    features[path+'_delta']=(output-skip).detach()
                if condition in (path+'_off','both_off'):return skip
                return output
            return hook
        try:
            for path in ('time','behavior'):
                handles.append(getattr(model.core,path+'_readout').register_forward_hook(hook_for(path)))
            scores[condition]=model(x)
        finally:
            for handle in handles:handle.remove()
    return scores,features

def execute(c):
    validate_config(c);release=Path(c['source_release'])
    if (release/'release_commit.txt').read_text().strip()!=c['source_commit']:raise ValueError('Frozen source release differs')
    sys.path[:0]=[str(release),str(release/'code')]
    from experiments.cvs_neural_readout_identity.dispatch import read_source_record
    from experiments.cvs_neural_readout_identity.model import build
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from experiments.cvs_identity_ce.source import source_args
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    source=Path(c['source_output']);resolved=read(source/'resolved_config.json');contract=read(source/'source_contract.json')
    record=read_source_record(c,read(resolved['source_contract']),'cvs_neural_readout_identity')
    initial=read(source/'initialization.json')
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    expected=dict(epoch=200,selection='fixed_last_epoch',method='cvs_neural_readout_identity',variant=c['variant'],config=resolved,source_contract=contract,initialization=initial,classes=contract['classes'],num_classes=6)
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint payload provenance mismatch')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    with numerical_context(resolved['numerical_policy']):
        model=build(c['variant']).to(device);model.load_state_dict(payload['model'],strict=True);model.eval();del payload
        for parameter in model.parameters():parameter.requires_grad_(False)
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            probe=torch.zeros(2,2,256,device=device);probe[1,0]=1.
            smoke,_=components(model,probe)
            if not torch.equal(smoke['all_on'],model(probe)):raise ValueError('Frozen no-query smoke changed all-on')
        split=build_contract_split(source_args(dict(resolved,output_root=str(out/'source_loader'))))
        if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Only complete source V allowed')
        loader=make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
        write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
            hardware=torch.cuda.get_device_name(device),backend_flags=actual_flags(),source_record=record,
            target_access=False,optimizer_updates=0,source_selection_changed=False))
        print('RESOLVED_CONFIG '+json.dumps(read(out/'resolved_config.json')),flush=True)
        ids=[];ys=[];rx=[];days=[];arrays={};ce={k:0. for k in CONDITIONS};tic=time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                x=batch['iq'].to(device);y=batch['label'].to(device);scores,features=components(model,x)
                for name,value in {**scores,**features}.items():
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite frozen output')
                    # N607 Torch was built against NumPy1: do not use .numpy().
                    arrays.setdefault(name,[]).append(np.asarray(value.cpu().tolist(),dtype=np.float32))
                for name,value in scores.items():ce[name]+=float(F.cross_entropy(value,y,reduction='sum'))
                ids.extend(m['sample_id'] for m in batch['meta']);ys.extend(batch['label'].tolist())
                rx.extend(batch['receiver'].tolist());days.extend(batch['day'].tolist())
                if len(ids)%4096==0:print('PROGRESS '+json.dumps(dict(source_v_packets=len(ids))),flush=True)
        if len(ids)!=27000 or len(set(ids))!=27000 or set(ids)!=set(contract['role_ids']['V']):raise ValueError('Physical source V mismatch')
        values={k:np.concatenate(v) for k,v in arrays.items()}
        ys=np.asarray(ys);rx=np.asarray(rx);days=np.asarray(days);rows=[]
        all_correct=values['all_on'].argmax(1)==ys
        for name in CONDITIONS:
            pred=values[name].argmax(1);correct=pred==ys
            rates={str(r):float(correct[rx==r].mean()) for r in sorted(set(rx))}
            rows.append(dict(condition=name,count=len(ids),accuracy=float(correct.mean()),ce=ce[name]/len(ids),
                rx_accuracy=rates,worst_rx=min(rates.values()),prediction_changes=int(np.count_nonzero(pred!=values['all_on'].argmax(1))),
                helped_by_all_on=int((all_correct&~correct).sum()),hurt_by_all_on=int((~all_correct&correct).sum())))
            print('CONDITION '+json.dumps(rows[-1]),flush=True)
        if rows[0]['accuracy']!=record['accuracy'] or rows[0]['worst_rx']!=record['worst_rx']:raise ValueError('All-on failed E200 reproduction')
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Frozen model state changed')
        if any(module._forward_hooks for _,module in model.named_modules()):raise ValueError('Diagnostic hooks leaked')
        np.savez(out/'source_readout_geometry.npz',ids=np.asarray(ids),truth=ys,receiver=rx,day=days,**values)
        done=dict(status='SOURCE_ATTRIBUTION_COMPLETE',variant=c['variant'],model_seed=c['model_seed'],rows=rows,
            source_v_count=len(ids),conditions=list(CONDITIONS),seconds=time.perf_counter()-tic,target_access=False,
            optimizer_updates=0,source_selection_changed=False,model_unchanged=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            claim='Full source V within frozen weights; interventions may leave training distribution. No retraining or causal channel disentanglement claim.')
        write(out/'completion.json',done);print(json.dumps(done),flush=True)
    return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);execute(read(p.parse_args().config))
