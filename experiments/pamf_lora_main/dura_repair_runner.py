"""Train LoRa recovery without F, then score with one immutable classifier."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .adaptive_repair import load_f
from .data import VIEWS
from .dura_repair import MODES, PacketRestorer, complex_iq, losses, old_restorer, train
from .packet_models import SEED
from .packet_runner import schedule
from .refine_publish import ADAPTIVE_RUN, DURA_RUN as RUN
from .refine_runner import RUN as PREVIOUS_G, infer, predict_models, source_handles, start
from .runner import activate

EPOCHS=10
F='adaptive_tri'
READOUTS={F:('direct','old_chirp','consensus',*MODES)}


def smoke(rt):
    _,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    prior=old_restorer(root.parent/PREVIOUS_G,c,'cpu')
    model=PacketRestorer('coherent')
    model.net.load_state_dict(prior.net.state_dict(),strict=True)
    clean=tensor(np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:24])
    pair=np.load(cache/'G_pairs.npy',mmap_mode='r',allow_pickle=False)
    # Select real recoverable source IQ, never V or target.
    x=tensor(pair[1,:96]); y=tensor(pair[0,:96])
    active=model.track(x)[-1]>0
    if not active.any():
        raise ValueError('DURA smoke found no recoverable sample')
    x,y=x[active][:3],y[active][:3]
    output=model(x)
    _,rot,_,_,_=model.track(x)
    shift=(complex_iq(output.detach()-x)*rot.conj()).mean(-1).abs().max()
    m,_,_,_=losses(model,y,x,output)
    m.backward()
    restored,gate=model(clean,with_gate=True)
    if (not torch.isfinite(output).all() or float(shift)>2e-6 or not (gate==0).all()
            or not torch.equal(restored,clean) or model.dose[-1].bias.grad[0].abs()<=0):
        raise ValueError('DURA smoke failed tone retention, dose learning, or clean identity')
    write(root/'smoke.json',dict(status='PASS',target_reads=0,F_reads=0,TX_reads=0,clean=len(clean),
          mean_tone_shift=float(shift),dose_gradient=float(model.dose[-1].bias.grad[0])))


def frozen(root,c,device):
    fpath=root.parent/ADAPTIVE_RUN
    if read(fpath/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('The matched fixed adaptive Tri must be frozen')
    f=load_f(fpath,c,F,device)
    gs={'old_chirp':old_restorer(root.parent/PREVIOUS_G,c,device),
        'consensus':PacketRestorer('consensus').to(device).eval()}
    for mode in MODES:
        p=root/'G'/mode
        if read(p/'complete.json')!=dict(status='SOURCE_FROZEN',mode=mode,seed=SEED,epochs=EPOCHS,
                         source_contract=c,F_reads=False,TX_labels_read=False,target_read=False):
            raise ValueError('Restorer freeze or source lineage differs')
        engine=PacketRestorer(mode).to(device).eval()
        engine.load_state_dict(torch.load(p/'model.pt',map_location=device,weights_only=True),strict=True)
        gs[mode]=engine
    return f,gs


def audit(rt):
    src,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    f,gs=frozen(root,c,'cuda:0')
    a=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    with np.load(src/'V_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['V']:
            raise ValueError('Source V physical rows differ')
        y=m['y']
    scores={}
    for i,v in enumerate(VIEWS):
        xp=tensor(a[i],'cuda:0')
        p,gates=infer(f,gs,xp)
        base=p['direct']==y
        scores[v]={k:dict(n=len(y),correct=int((q==y).sum()),gained=int(((q==y)&(~base)).sum()),
                     lost=int(((q!=y)&base).sum()),active=sum(g>0 for g in gates.get(k,[])))
                   for k,q in p.items()}
    write(root/'source_audit.json',dict(status='SOURCE_AUDIT_COMPLETE',target_reads=0,records=scores))


def worker(rt,kind,name):
    activate(rt)
    source,parent,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    if kind=='train_g':
        return train(cache,root.parent/PREVIOUS_G,root/'G'/name,c,name,SEED,EPOCHS)
    if kind=='smoke':
        return smoke(rt)
    if kind=='audit':
        return audit(rt)
    f,gs=frozen(root,c,'cuda:0')
    return predict_models(root,parent,Path(rt['target_views']),F,f,gs,READOUTS[F])


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if launch(rt,'smoke','iq').wait():
            raise RuntimeError('DURA source smoke failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train_g',m) for m in MODES],launch)
        schedule(rt,[('audit','source')],launch)
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',F)],launch)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('dispatch','smoke','train_g','audit','predict'))
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?')
    a=p.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        worker(rt,a.kind,a.name)
