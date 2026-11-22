"""Read the same frozen LoRa IQ with the original Tri and R69 adaptations."""
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .full_dura_r69_lora import R69_LORA_RUN, R69_MODES, R69_SEEDS, load_model, load_utility, restored
from .full_packet_data import LIMIT
from .full_supervised import handles
from .full_tri_repl_lora import REPL_METHODS, REPL_RUN
from .packet_models import packet_tri


JOINT_RUN='20261122-phase1-lora-native-dura-tri-joint-r01'
JOINT_SEEDS=R69_SEEDS
JOINT_SCHEMES=('direct','r69','r69_half')
JOINT_MODES=(*R69_MODES,*(f'{m}_{s}' for m in REPL_METHODS for s in JOINT_SCHEMES))


def reader(root,c,seed,method,device):
    from .full_adaptive import case, complete_f
    f=packet_tri('packet_sup',device).eval()
    path=complete_f(root.parent/REPL_RUN,c,case(seed,method))
    f.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True),strict=True)
    return f.requires_grad_(False)


@torch.no_grad()
def smoke(rt):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    seed,=JOINT_SEEDS
    g=root.parent/R69_LORA_RUN
    model=load_model(g,c,seed,'student','cpu')
    gate=load_utility(g,c,seed,'cpu')
    practical=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)[1:,::47]
    x=tensor(practical).flatten(0,1)
    keep=gate.impairment(x)>0
    if not keep.any():
        raise ValueError('Frozen LoRa joint smoke has no practical recovery')
    pairs=restored(gate,model,x[keep][:4])
    for method in REPL_METHODS:
        f=reader(root,c,seed,method,'cpu')
        logits,_=f(torch.cat(pairs))
        if logits.shape!=(len(pairs[0])*2,10) or not torch.isfinite(logits).all():
            raise ValueError('Frozen LoRa joint smoke produced invalid logits')
    write(root/'smoke.json',dict(status='PASS',seed=seed,networks_frozen=True,U_truth_reads=0,target_reads=0))


@torch.no_grad()
def predict_joint(root,target,c,seed,device):
    from .full_adaptive import case
    groom=root.parent/R69_LORA_RUN
    froom=root.parent/REPL_RUN
    old=groom/'predictions'/str(seed)
    base=read(old/'complete.json')
    activation=base.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=seed,target_truth_read=False,
                  receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS))
    if base!=dict(expected,modes=list(R69_MODES)):
        raise ValueError('Original DURA joint operating controls are not frozen on the same matrix')
    individual=[]
    for m in REPL_METHODS:
        p=froom/'predictions'/case(seed,m)
        if read(p/'complete.json')!=dict(expected,method=m,modes=['direct']):
            raise ValueError('Original Tri controls are not frozen on the same matrix')
        individual.append(p)
    model=load_model(groom,c,seed,'student',device)
    gate=load_utility(groom,c,seed,device)
    learners=[reader(root,c,seed,m,device) for m in REPL_METHODS]
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    for rx in TARGET:
        iq=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if iq.shape!=(len(VIEWS),LIMIT,6,2,256) or iq.dtype!=np.float32:
            raise ValueError('Original DURA and Tri combined test IQ differ')
        for j,v in enumerate(VIEWS):
            previous=[np.load(old/f'{rx}__{v}__{m}.npy',allow_pickle=False) for m in R69_MODES]
            raw=[np.load(p/f'{rx}__{v}__direct.npy',allow_pickle=False) for p in individual]
            if any(a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=10
                   for a in (*previous,*raw)):
                raise ValueError('Invalid frozen joint LoRa controls')
            decisions=[(p,p.copy(),p.copy()) for p in raw]
            active=0
            for first in range(0,LIMIT,48):
                x=tensor(iq[j,first:first+48],device)
                k=gate.impairment(x)>0
                if k.any():
                    recovered=torch.cat(restored(gate,model,x[k]))
                    ii=np.asarray(torch.where(k)[0].tolist(),dtype=np.int64)+first
                    for f,(_,full,half) in zip(learners,decisions,strict=True):
                        logits,_=f(recovered)
                        if not torch.isfinite(logits).all():
                            raise ValueError('Combined R69/Tri prediction is nonfinite')
                        q,h=logits.argmax(1).chunk(2)
                        full[ii]=q.tolist()
                        half[ii]=h.tolist()
                    active+=len(ii)
            if active!=activation[f'{rx}/{v}']['r69_active']:
                raise ValueError('Combined R69/Tri run changed the frozen R69 gate')
            values=(*previous,*(v0 for r in decisions for v0 in r))
            for name,a in zip(JOINT_MODES,values,strict=True):
                np.save(out/f'{rx}__{v}__{name}.npy',a)
        print('LORA_NATIVE_JOINT_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=list(JOINT_MODES),activation=activation))
