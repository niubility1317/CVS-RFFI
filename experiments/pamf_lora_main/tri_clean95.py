"""Clean-only LoRa PAMF-Tri optimization on the registered physical split."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss
from .comparison import seed_all
from .data import SOURCE, TARGET, WINDOW_START
from .full_packet_data import COUNTS, LIMIT
from .full_packet_views import native, target_contract
from .full_supervised import batches, handles
from .full_tri_repl_lora import labeled_loss, unlabeled_loss
from .packet_data import CHIRPS
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .tri_tuning import SEED, STEPS, model_for, shape_trace, trainable


RUN='20261125-phase1-lora-pamf-tri-clean95-m8-r01'
ROWS={
    'control': {'lr':6e-5},
    'lr3e4': {},
    'iq192': {'samples':192},
    'iq384': {'samples':384},
    'iq512': {'samples':512},
    'time_pool4': {'branch_pool':'time'},
    'signature_aligned': {'signature_half':True,'align_bins':12},
    'l_warmup': {'warmup':6},
}
LOCAL=('control','iq384')
QUICK_RUN='20261125-phase1-lora-pamf-tri-clean-branch-loss-m7-r01'
QUICK_ROWS={
    'control': {},
    'frequency_pool4': {'branch_pool':'frequency'},
    'pa_pool4': {'branch_pool':'pa'},
    'frequency_grid2x': {'freq_scale':2},
    'reference96': {'reference_width':96},
    'contrast_double': {'contrast_weight':.2},
    'kl_half': {'kl_weight':.5},
}


def study(rt):
    if rt.get('quick'):
        return QUICK_ROWS,8,16
    return ROWS,EPOCHS,U_BATCH
EPOCHS=24
U_BATCH=32
TARGET_SOURCE_ACCURACY=.95
VALIDATION_EPOCHS={0,2,4,8,12,16,20,24}
STARTS=np.asarray(CHIRPS)*1024+WINDOW_START-125
CROP=STARTS[:,None]+np.arange(1250)


def clean_iq(packets,samples):
    if samples not in (192,256,384,512):
        raise ValueError('Unknown registered clean aperture')
    z = resample_poly(packets[:,CROP],up=samples//8,down=125,axis=-1)
    edge=samples//8
    z=z[...,edge:edge+samples]
    norm=np.sqrt(np.mean(np.abs(z)**2,axis=(1,2),keepdims=True))
    if not np.isfinite(norm).all() or (norm<=1e-12).any():
        raise ValueError('Nonfinite or empty clean packet')
    z=z/norm
    return np.stack((z.real,z.imag),axis=2).astype(np.float32,copy=False)


def cache(rt):
    source,_,contract=handles(rt)
    root=Path(rt['run_root'])
    names=LOCAL if rt.get('local_parent') else ROWS
    for n in sorted({ROWS[k].get('samples',256) for k in names}):
        folder=root/'clean_cache'/str(n)
        folder.mkdir(parents=True,exist_ok=False)
        for key,count in COUNTS.items():
            x=native(source,key,count)
            out=np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',dtype=np.float32,shape=(count,6,2,n))
            for i in range(0,count,64):
                out[i:i+64]=clean_iq(x[i:i+64],n)
            out.flush()
            print('LORA_CLEAN_IQ',n,key,count,flush=True)
        write(folder/'complete.json',dict(status='CLEAN_SOURCE_READY',samples=n,source_protocol=contract['protocol']))


def labels(source,contract,key):
    with np.load(source/f'{key}_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=contract['role_ids'][key] or ('y' in m.files)!=(key!='U_s'):
            raise ValueError(f'Unexpected TX access or source physical order: {key}')
        return (m['y'],m['rx_i']) if key!='U_s' else m['rx_i']


def clean_validation(f,v,y,rx):
    f.eval()
    with torch.no_grad():
        q=torch.cat([f(batch)[0] for batch in v.split(64)])
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite clean source logits')
        hits=q.argmax(1)==y
        return dict(n=len(y),correct=int(hits.sum()),accuracy=float(hits.float().mean()),
                    nll=float(torch.nn.functional.cross_entropy(q,y)),
                    by_rx={name:float(hits[rx==i].float().mean()) for i,name in enumerate(SOURCE)},
                    by_tx={str(31+i):float(hits[y==i].float().mean()) for i in range(10)})


def terms(row):
    return {k:row[k] for k in ('contrast_weight','kl_weight') if k in row}


def _labeled_and_unlabeled(f,x,y,rx,u,ur,step,with_u,force=None,mechanism=None):
    f.train()
    lab,_=labeled_loss(f,x,x,y,rx,step,mechanism=mechanism)
    if not with_u:
        return lab,lab.detach(),0,0
    unlab,pseudo,keep,details=unlabeled_loss(f,u,ur,step,3.,forced_pseudo=force,mechanism=mechanism)
    safe,_,_=protected_loss(lab,unlab,trainable(f))
    if force is not None and (details['forwards']!=6 or not details['partners']):
        raise ValueError('Real smoke did not traverse the Tri training branches')
    return safe,lab.detach(),unlab.detach(),int(keep.sum())


def smoke(rt):
    from copy import deepcopy
    root=Path(rt['run_root'])
    source,_,c=handles(rt)
    parent=model_for(rt,root,c,{},'cpu')
    yn,rn=labels(source,c,'L_s')
    raw=native(source,'L_s',len(yn))
    i=np.sort(next(batches(yn,rn,np.random.default_rng(721)))[:8])
    y,rx=tensor(yn[i]),tensor(rn[i])
    records={}
    from .tri_tuning import configure
    for name,row in study(rt)[0].items():
        f=configure(deepcopy(parent),row)
        a=tensor(clean_iq(raw[i],row.get('samples',256)))
        t=shape_trace(f,a)
        loss,_,_,n=_labeled_and_unlabeled(f,a,y,rx,a,rx,1,True,force=y,mechanism=terms(row))
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True))
        if norm<=0 or n!=len(y):raise ValueError('Failed real clean Tri gradient check')
        records[name]=dict(t,gradient=norm)
    write(root/'smoke.json',dict(status='PASS',U_truth_reads=0,target_reads=0,rows=records))


def train(rt,name):
    root=Path(rt['run_root'])
    source,_,c=handles(rt)
    rows,epochs,u_count=study(rt)
    row=rows[name]
    n=row.get('samples',256)
    cache_root=Path(rt.get('fixed_clean_cache',root/'clean_cache'))
    clean=cache_root/str(n)
    if read(clean/'complete.json')!=dict(status='CLEAN_SOURCE_READY',samples=n,source_protocol=c['protocol']):
        raise ValueError('Clean source cache not ready')
    device='cuda:0'
    out=root/'F'/name
    out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    f=model_for(rt,root,c,row,device)
    seed_all(SEED)
    x,u,v=[tensor(np.load(clean/f'{key}.npy',mmap_mode='r',allow_pickle=False),device)
           for key in ('L_s','U_s','V')]
    yn,rn=labels(source,c,'L_s')
    y,rx=tensor(yn,device),tensor(rn,device)
    ur=tensor(labels(source,c,'U_s'),device)
    vy,vr=(tensor(a,device) for a in labels(source,c,'V'))
    opt=torch.optim.AdamW(f.parameters(),lr=row.get('lr',3e-4),weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs*STEPS,eta_min=1e-5)
    rng=np.random.default_rng(721)
    ugen=torch.Generator(device=device).manual_seed(SEED)
    path=out/'best.pt'
    initial=clean_validation(f,v,vy,vr)
    best=initial | {'epoch':0}
    torch.save(f.state_dict(),path)
    write(out/'initial.json',dict(V=initial,shapes=shape_trace(f,x[:2])))
    step=0
    mechanism=terms(row)
    with (out/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            use_u=epoch>row.get('warmup',0)
            order=torch.randperm(len(u),device=device,generator=ugen)[:STEPS*u_count].reshape(STEPS,u_count)
            ui=order if use_u else (None,)*STEPS
            sl,su,accepted=0.,0.,0
            for b,pos in zip(batches(yn,rn,rng),ui,strict=True):
                i=tensor(b,device)
                step+=1
                opt.zero_grad(set_to_none=True)
                ux,urb=(u[pos],ur[pos]) if use_u else (None,None)
                loss,lab,unlab,a=_labeled_and_unlabeled(f,x[i],y[i],rx[i],ux,urb,step,use_u,mechanism=mechanism)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                sch.step()
                sl+=float(lab);su+=float(unlab);accepted+=a
            rec=dict(epoch=epoch,step=step,L_loss=sl/STEPS,U_loss=su/STEPS,
                     U_seen=STEPS*u_count if use_u else 0,U_accepted=accepted,
                     lr=opt.param_groups[0]['lr'])
            if epoch in VALIDATION_EPOCHS:
                rec['V']=clean_validation(f,v,vy,vr)
                key=lambda p:(p['correct'],-p['nll'])
                if key(rec['V'])>key(best):
                    best=rec['V'] | {'epoch':epoch}
                    torch.save(f.state_dict(),path)
            log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
            print('LORA_CLEAN95',name,epoch,step,rec.get('V',{}).get('accuracy'),accepted,flush=True)
    if step!=epochs*STEPS:
        raise ValueError('Incomplete registered clean updates')
    f.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
    training=clean_validation(f,x,y,rx)
    write(out/'complete.json',dict(status='SOURCE_FROZEN',seed=SEED,method=name,frozen=best,L=training,
                                  steps=step,clean_only=True,target_reads=0,U_truth_reads=0))


@torch.no_grad()
def predict(rt,name):
    root=Path(rt['run_root'])
    if name not in read(root/'selection.json')['names']:
        raise ValueError('Clean test permission requires the source-frozen decision')
    _,_,c=handles(rt)
    target=Path(rt['native_target_root'])
    target_contract(target)
    f=model_for(rt,root,c,ROWS[name],'cuda:0').eval()
    path=root/'F'/name
    f.load_state_dict(torch.load(path/'best.pt',map_location='cuda:0',weights_only=True),strict=True)
    out=root/'predictions'/name
    out.mkdir(parents=True,exist_ok=False)
    for rx in TARGET:
        x=native(target,rx,LIMIT)
        codes=[]
        for i in range(0,LIMIT,64):
            y=f(tensor(clean_iq(x[i:i+64],ROWS[name].get('samples',256)),'cuda:0'))[0]
            if not torch.isfinite(y).all():raise ValueError('Nonfinite blind clean logits')
            codes.extend(y.argmax(1).tolist())
        np.save(out/f'{rx}__clean__direct.npy',np.asarray(codes,dtype=np.int16))
        print('LORA_CLEAN_BLIND',name,rx,flush=True)
    write(out/'complete.json',dict(method=name,seed=SEED,status='PREDICTIONS_FROZEN',
          target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=['clean']))


def dispatch(rt):
    if rt.get('quick') and rt.get('local_parent'):
        raise ValueError('Quick branch study is registered for N607 only')
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    fields=tuple(k for k in ('local_parent','quick','fixed_clean_cache') if k in rt)
    # Training config remains source-only. Target handles travel only in the predictor config.
    launch=partial(start,module=__spec__.name,extra_keys=fields)
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        stages=('smoke',) if rt.get('quick') else ('smoke','cache')
        for stage in stages:
            if launch(rt,stage,'all').wait():raise RuntimeError(f'Clean {stage} failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        names=LOCAL if rt.get('local_parent') else tuple(study(rt)[0])
        schedule(rt,[('train',name) for name in names],launch)
        evidence={name:read(root/'F'/name/'complete.json')['frozen'] for name in names}
        ranking=sorted(names,key=lambda name:(evidence[name]['correct'],-evidence[name]['nll']),reverse=True)
        source_only=bool(rt.get('local_parent') or rt.get('quick'))
        tested=[] if source_only else list(dict.fromkeys(['control',ranking[0]]))
        write(root/'selection.json',dict(names=tested,rank=ranking,source=evidence,
                                        target_reads=0,goal_met=evidence[ranking[0]]['correct']>=TARGET_SOURCE_ACCURACY*evidence[ranking[0]]['n']))
        if source_only:
            write(state,dict(status='SOURCE_DIAGNOSTIC_COMPLETE'),exclusive=False)
            return
        write(state,dict(status='PREDICTING',names=tested),exclusive=False)
        def launch_blind(runtime,kind,name,gpu=None):
            return start(runtime,kind,name,gpu,module=__spec__.name,extra_keys=('native_target_root',))
        schedule(rt,[('predict',name) for name in tested],launch_blind)
        write(state,dict(status='PREDICTIONS_FROZEN',names=tested),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('dispatch','smoke','cache','train','predict'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('name',nargs='?',default='')
    args=parser.parse_args()
    rt=read(args.runtime)
    if args.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if args.kind in ('cache','smoke'):
            {'cache':cache,'smoke':smoke}[args.kind](rt)
        else:
            {'train':train,'predict':predict}[args.kind](rt,args.name)
