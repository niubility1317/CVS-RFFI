"""Clean-only LoRa PAMF-Tri optimization on the registered physical split."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss
from .clean_aperture import ALL_CHIRPS, clean_iq
from .comparison import seed_all
from .data import SOURCE, TARGET
from .full_packet_data import COUNTS, LIMIT
from .full_packet_views import native, source_contract, target_contract
from .full_supervised import batches, handles
from .full_tri_repl_lora import clean_source_ce, labeled_loss, unlabeled_loss
from .packet_data import CHIRPS
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .repair_features import AlignedTrace, CoherentPacket
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


TRANSFER_RUN='20261125-phase1-lora-pamf-tri-clean-transfer-m5-r01'
TRANSFER_ROWS={
    'control': {'lr':1.5e-4},
    'raw_ce': {'lr':1.5e-4,'raw_ce':.25},
    'dual384': {'lr':1.5e-4,'samples':384,'two_rate':True},
    'dual384_raw_ce': {'lr':1.5e-4,'samples':384,'two_rate':True,'raw_ce':.25},
    'dual512': {'lr':1.5e-4,'samples':512,'two_rate':True},
}


class TwoRateCoherent(CoherentPacket):
    def forward(self, pack):
        raw, anchor = pack.split((pack.shape[-1]-256,256),dim=-1)
        logits,fused=self.raw(raw)
        traces=self.trace(anchor)
        aligned=self.encoder(traces.flatten(0,1)).reshape(len(pack),pack.shape[1],-1).mean(1)
        return logits+self.head(aligned),fused+.10*aligned


def preserve_alignment(f,row):
    if row.get('two_rate'):
        f.__class__=TwoRateCoherent
        f.trace=AlignedTrace().to(f.head.weight.device)
    return f


class SpectrumAligned(CoherentPacket):
    def forward(self,x):
        logits,fused=self.raw(x)
        traces=self.trace(x)
        aligned=self.encoder(traces.flatten(0,1)).reshape(len(x),x.shape[1],-1).mean(1)
        z=torch.complex(traces[:,:,0],traces[:,:,1])
        z=z-z.mean(-1,keepdim=True)
        energy=(z.real.square()+z.imag.square()).mean(-1,keepdim=True).clamp_min(1e-6)
        spectral=torch.fft.fftshift(torch.fft.fft(z,dim=-1),dim=-1)
        power=(spectral.abs().square()/(z.shape[-1]*energy)).log1p()
        bands=torch.nn.functional.adaptive_avg_pool1d(power.flatten(0,1).unsqueeze(1),64)
        bands=bands.reshape(len(x),x.shape[1],64)
        pooled=torch.cat((bands.mean(1),bands.std(1,unbiased=False)),dim=1)
        aligned=aligned+.10*self.spectral(pooled)
        return logits+self.head(aligned),fused+.10*aligned


class ChirpReadout(torch.nn.Module):
    def __init__(self,channels,classes,chirp_count):
        super().__init__()
        self.gate=torch.nn.Linear(channels,classes,bias=False)
        self.position=torch.nn.Parameter(torch.zeros(chirp_count,classes))
        torch.nn.init.zeros_(self.gate.weight)

    def forward(self,chirps,head_weight):
        scores=self.gate(chirps)+self.position
        delta=scores.softmax(dim=1)-1/chirps.shape[1]
        chirp_logits=torch.nn.functional.linear(chirps,head_weight)
        # Preserve half of the uniform full-packet weight for every chirp.
        return .5*(delta*chirp_logits).sum(dim=1)


class TimeFrequencyAligned(CoherentPacket):
    def forward(self,x):
        drift_layer=getattr(self,'relative_drift',None)
        phase_layer=getattr(self,'phase_curvature',None)
        physics_layer=getattr(self,'packet_physics',None)
        if physics_layer is not None:
            traces,phase,physics=self.trace(x,with_packet_physics=True)
        elif phase_layer is not None:
            traces,phase=self.trace(x,with_phase=True)
        elif drift_layer is None:
            traces=self.trace(x)
        else:
            traces,drift=self.trace(x,with_drift=True)
        core_input=x
        if getattr(self,'nominal_iq',False):
            baseband=torch.complex(traces[:,:,0],traces[:,:,1])
            nominal=baseband*self.trace.reference.chirp
            core_input=torch.stack((nominal.real,nominal.imag),dim=2)
        logits,fused=self.raw(core_input)
        chirps=self.encoder(traces.flatten(0,1)).reshape(len(x),x.shape[1],-1)
        aligned=chirps.mean(1)
        edge_layer=getattr(self,'edge_features',None)
        if edge_layer is not None:
            interior=chirps[:,1:-1].mean(1)
            tail=chirps[:,-1]-interior
            edges=torch.cat((chirps[:,0]-interior,torch.zeros_like(tail) if self.first_edge else tail),dim=1)
            aligned=aligned+.10*edge_layer(edges)
        z=torch.complex(traces[:,:,0],traces[:,:,1])
        z=z-z.mean(-1,keepdim=True)
        size=z.shape[-1]//8
        windows=z.unfold(-1,size,size//2)
        apodize=torch.hann_window(size,device=z.device,dtype=z.real.dtype)
        power=torch.fft.fftshift(torch.fft.fft(windows*apodize,dim=-1),dim=-1).abs().square()
        log=(power/size+1e-5).log()
        ratio=(log[...,1:,:]-log[...,:-1,:]).clamp(-8.,8.)
        local=log[...,1:,:]
        bands=[local.mean(1),ratio.mean(1)]
        if self.spectral_edges:
            interior=local[:,1:-1].mean(1)
            bands.extend((local[:,0]-interior,local[:,-1]-interior))
        features=torch.stack(bands,dim=1)
        aligned=aligned+.10*self.time_frequency(features)
        if drift_layer is not None:
            aligned=aligned+.10*drift_layer(torch.asinh(drift))
        if phase_layer is not None:
            aligned=aligned+.10*phase_layer(phase)
        if physics_layer is not None:
            aligned=aligned+.10*physics_layer(physics)
        logits=logits+self.head(aligned)
        chirp_readout=getattr(self,'chirp_readout',None)
        if chirp_readout is not None:
            logits=logits+chirp_readout(chirps,self.head.weight)
        return logits,fused+.10*aligned


def clean_model(rt,root,c,row,device):
    if rt.get('scratch'):
        from .holdout import construct
        from .tri_tuning import configure
        f=configure(construct(device),row)
        if row.get('spectrum'):
            f.__class__=SpectrumAligned
            f.spectral=torch.nn.Sequential(torch.nn.Linear(128,160),
                                           torch.nn.LayerNorm(160),torch.nn.SiLU()).to(device)
        if row.get('time_frequency'):
            from torch import nn
            f.__class__=TimeFrequencyAligned
            f.nominal_iq=row.get('nominal_iq',False)
            f.spectral_edges=row.get('spectral_edges',False)
            f.first_edge=row.get('first_edge',False)
            f.time_frequency=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
                nn.Conv2d(16,24,3,padding=1,stride=2),nn.GroupNorm(4,24),nn.SiLU(),
                nn.AdaptiveAvgPool2d((4,5)),nn.Flatten(),nn.Linear(480,160),nn.LayerNorm(160)).to(device)
            if row.get('relative_drift'):
                f.relative_drift=nn.Sequential(nn.Linear(len(row['chirps']),32),nn.LayerNorm(32),nn.SiLU(),
                                              nn.Linear(32,160),nn.LayerNorm(160)).to(device)
            if row.get('edge_features'):
                f.edge_features=nn.Sequential(nn.LayerNorm(320),nn.Linear(320,160),nn.LayerNorm(160)).to(device)
            if row.get('phase_curvature'):
                f.phase_curvature=nn.Sequential(nn.Linear(12,32),nn.LayerNorm(32),nn.SiLU(),
                                               nn.Linear(32,160),nn.LayerNorm(160)).to(device)
            if row.get('packet_physics'):
                f.packet_physics=nn.Sequential(nn.Linear(24,32),nn.LayerNorm(32),nn.SiLU(),
                                               nn.Linear(32,160),nn.LayerNorm(160)).to(device)
            if row.get('chirp_readout'):
                f.chirp_readout=ChirpReadout(f.head.in_features,f.head.out_features,len(row['chirps'])).to(device)
            if f.spectral_edges:
                original=f.time_frequency[0]
                extended=nn.Conv2d(4,16,3,padding=1,device=device)
                with torch.no_grad():
                    extended.weight[:,:2].copy_(original.weight)
                    extended.weight[:,2:].zero_()
                    extended.bias.copy_(original.bias)
                f.time_frequency[0]=extended
        return f
    return preserve_alignment(model_for(rt,root,c,row,device),row)


def study(rt):
    if rt.get('scratch'):
        if rt.get('pairmargin_study'):
            return PAIR_MARGIN_ROWS,112,16
        if rt.get('chirpread_study'):
            return CHIRP_READ_ROWS,112,16
        if rt.get('coverage_study'):
            return COVERAGE_ROWS,112,16
        if rt.get('physics_study'):
            return PHYSICS_ROWS,112,16
        if rt.get('rtl_study'):
            return RTL_ROWS,112,16
        if rt.get('phase_joint_study'):
            return PHASE_JOINT_ROWS,112,16
        if rt.get('timing_study'):
            return TIMING_ROWS,112,16
        if rt.get('anchor_study'):
            return ANCHOR_ROWS,112,16
        if rt.get('phase_study'):
            return PHASE_ROWS,80,16
        if rt.get('robust_study'):
            return ROBUST_ROWS,80,16
        if rt.get('edge_study'):
            return EDGE_ROWS,80,16
        if rt.get('budget_study'):
            return BUDGET_ROWS,88,16
        if rt.get('drift_study'):
            return DRIFT_ROWS,SCRATCH_EPOCHS,16
        if rt.get('noise_study'):
            return NOISE_ROWS,SCRATCH_EPOCHS,16
        return (TIMEFREQ_ROWS if rt.get('timefreq') else SCRATCH_ROWS),SCRATCH_EPOCHS,16
    if rt.get('transfer'):
        return TRANSFER_ROWS,8,16
    if rt.get('quick'):
        return QUICK_ROWS,8,16
    return ROWS,EPOCHS,U_BATCH
EPOCHS=24
U_BATCH=32
TARGET_SOURCE_ACCURACY=.95
SCRATCH_RUN='20261125-phase1-lora-pamf-tri-clean-scratch-m5-r01'
SCRATCH_ROWS={
    'control': {'warmup':32},
    'scratch384': {'warmup':32,'samples':384},
    'scratch512': {'warmup':32,'samples':512},
    'residual_spectrum': {'warmup':32,'spectrum':True},
    'raw_ce': {'warmup':32,'raw_ce':.25},
}
SCRATCH_LOCAL=('control','residual_spectrum')
SCRATCH_EPOCHS=48
TIMEFREQ_RUN='20261125-phase1-lora-pamf-tri-clean-timefreq-m2-r01'
TIMEFREQ_ROWS={
    'timefreq256': {'warmup':32,'time_frequency':True},
    'timefreq384': {'warmup':32,'samples':384,'time_frequency':True},
}
NOISE_RUN='20261125-phase1-lora-pamf-tri-clean-noise-m3-r01'
NOISE_ROWS={
    'standard_noise': {'warmup':32,'time_frequency':True,'ordinary_std':.015},
    'small_noise': {'warmup':32,'time_frequency':True,'ordinary_std':.003},
    'no_noise': {'warmup':32,'time_frequency':True,'ordinary_std':0.},
}
DRIFT_RUN='20261125-phase1-lora-pamf-tri-clean-relativetone-m2-r01'
DRIFT_ROWS={
    'all8_control384': {'warmup':32,'samples':384,'time_frequency':True,'chirps':ALL_CHIRPS,'ordinary_std':.003},
    'all8_relative384': {'warmup':32,'samples':384,'time_frequency':True,'chirps':ALL_CHIRPS,'ordinary_std':.003,
                         'relative_drift':True},
}
BUDGET_RUN='20261125-phase1-lora-pamf-tri-clean-budget-m2-r01'
BUDGET_ROWS={
    'budget_control384': {'warmup':72,'samples':384,'time_frequency':True},
    'budget_nominal384': {'warmup':72,'samples':384,'time_frequency':True,'nominal_iq':True},
}
EDGE_RUN='20261125-phase1-lora-pamf-tri-clean-full8-m2-r01'
EDGE_ROWS={
    'control': {'warmup':64,'samples':384,'time_frequency':True,'chirps':ALL_CHIRPS,'ordinary_std':.003},
    'edge_aware384': {'warmup':64,'samples':384,'time_frequency':True,'chirps':ALL_CHIRPS,'ordinary_std':.003,
                      'edge_features':True},
}
ROBUST_RUN='20261125-phase1-lora-pamf-tri-clean-robust98-m4-r01'
ROBUST_ROWS={
    'control': dict(EDGE_ROWS['edge_aware384']),
    'spectral_edges': dict(EDGE_ROWS['edge_aware384'],spectral_edges=True),
    'joint_rx': dict(EDGE_ROWS['edge_aware384'],joint_rx_weight=.12),
    'spectral_joint_rx': dict(EDGE_ROWS['edge_aware384'],spectral_edges=True,joint_rx_weight=.12),
}
ROBUST_SOURCE_CORRECT=11760
PHASE_RUN='20261125-phase1-lora-pamf-tri-clean-headphase98-m2-r01'
PHASE_ROWS={
    'first_edge': dict(ROBUST_ROWS['control'],first_edge=True),
    'first_phase': dict(ROBUST_ROWS['control'],first_edge=True,phase_curvature=True),
}
ANCHOR_RUN='20261125-phase1-lora-pamf-tri-clean-anchor98-m2-r01'
ANCHOR_ROWS={
    'long_phase': dict(PHASE_ROWS['first_phase'],warmup=96),
    'phase_anchor': dict(PHASE_ROWS['first_phase'],warmup=96,clean_anchor_weight=.5),
}
TIMING_RUN='20261125-phase1-lora-pamf-tri-clean-timing98-s2026092702-r01'
TIMING_ROWS={
    'phase_jitter': dict(ANCHOR_ROWS['phase_anchor'],timing_jitter=True),
}
PHASE_JOINT_RUN='20261125-phase1-lora-pamf-tri-clean-phasejoint98-s2026092702-r01'
PHASE_JOINT_ROWS={
    'phase_joint': dict(ANCHOR_ROWS['phase_anchor'],joint_rx_weight=.12),
}
PAIR_MARGIN_RUN='20261125-phase1-lora-pamf-tri-clean-pairmargin98-s2026092702-r01'
PAIR_MARGIN_ROWS={
    'phase_pairmargin': dict(ANCHOR_ROWS['phase_anchor'],pair_margin_weight=.25),
}
CHIRP_READ_RUN='20261125-phase1-lora-pamf-tri-clean-chirpread98-s2026092702-r01'
CHIRP_READ_ROWS={
    'phase_chirpread': dict(ANCHOR_ROWS['phase_anchor'],chirp_readout=True),
}
COVERAGE_RUN='20261125-phase1-lora-pamf-tri-clean-ucoverage98-s2026092702-r01'
COVERAGE_ROWS={
    'phase_coverage': dict(ANCHOR_ROWS['phase_anchor'],cover_u=True),
}
PHYSICS_RUN='20261125-phase1-lora-pamf-tri-clean-packetphysics98-s2026092702-r01'
PHYSICS_ROWS={
    'phase_physics': dict(ANCHOR_ROWS['phase_anchor'],packet_physics=True),
}
RTL_RUN='20261125-phase1-lora-pamf-tri-clean-rtl98-s2026092702-r01'
RTL_ROWS={
    'phase_rtl': dict(ANCHOR_ROWS['phase_anchor'],rtl_weight=2.),
}
VALIDATION_EPOCHS={0,2,4,8,12,16,20,24}


def source_handles(rt):
    if rt.get('scratch'):
        source=Path(rt['source_root'])
        return source,None,source_contract(source)
    return handles(rt)


def on_local(rt):
    return bool(rt.get('local_parent') or rt.get('local_only'))


def row_names(rt):
    if not on_local(rt):
        return tuple(study(rt)[0])
    if rt.get('noise_study'):
        return ('small_noise',)
    if rt.get('drift_study'):
        return ('all8_relative384',)
    if rt.get('budget_study'):
        return ('budget_nominal384',)
    if rt.get('edge_study'):
        return ('edge_aware384',)
    if rt.get('robust_study'):
        return ('spectral_joint_rx',)
    if rt.get('phase_study'):
        return ('first_phase',)
    if rt.get('anchor_study'):
        return ('phase_anchor',)
    if rt.get('timing_study'):
        return ('phase_jitter',)
    if rt.get('phase_joint_study'):
        return ('phase_joint',)
    if rt.get('pairmargin_study'):
        return ('phase_pairmargin',)
    if rt.get('chirpread_study'):
        return ('phase_chirpread',)
    if rt.get('coverage_study'):
        return ('phase_coverage',)
    if rt.get('physics_study'):
        return ('phase_physics',)
    if rt.get('rtl_study'):
        return ('phase_rtl',)
    return SCRATCH_LOCAL if rt.get('scratch') else LOCAL


def cache_layout(samples,chirps,protocol):
    name=str(samples)
    expected=dict(status='CLEAN_SOURCE_READY',samples=samples,source_protocol=protocol)
    if chirps==ALL_CHIRPS:
        name+='-all8'
        expected['chirps']=list(chirps)
    return name,expected


def cache(rt):
    source,_,contract=source_handles(rt)
    root=Path(rt['run_root'])
    rows=study(rt)[0]
    shapes={(rows[k].get('samples',256),rows[k].get('chirps',CHIRPS)) for k in row_names(rt)}
    for n,chirps in sorted(shapes):
        key,expected=cache_layout(n,chirps,contract['protocol'])
        folder=root/'clean_cache'/key
        folder.mkdir(parents=True,exist_ok=False)
        for key,count in COUNTS.items():
            x=native(source,key,count)
            out=np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',dtype=np.float32,shape=(count,len(chirps),2,n))
            for i in range(0,count,64):
                out[i:i+64]=clean_iq(x[i:i+64],n,chirps)
            out.flush()
            print('LORA_CLEAN_IQ',n,key,count,flush=True)
        write(folder/'complete.json',expected)


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


def _labeled_and_unlabeled(f,x,y,rx,u,ur,step,with_u,force=None,mechanism=None,raw_ce=0.,ordinary_std=.015,
                           joint_rx_weight=0.,clean_anchor_weight=0.,rtl_weight=1.,pair_margin_weight=0.):
    f.train()
    scores=[]
    hook=f.raw.register_forward_hook(lambda _,args,out:scores.append(out[0])) if raw_ce else None
    try:
        lab,_=labeled_loss(f,x,x,y,rx,step,mechanism=mechanism,ordinary_std=ordinary_std,
                           joint_rx_weight=joint_rx_weight,clean_anchor_weight=clean_anchor_weight,
                           rtl_weight=rtl_weight,pair_margin_weight=pair_margin_weight)
    finally:
        if hook is not None:
            hook.remove()
    if raw_ce:
        if len(scores)!=6:
            raise ValueError('Original raw Tri readout must occur in each supervised pass')
        lab=lab+raw_ce*torch.stack([torch.nn.functional.cross_entropy(q,y,label_smoothing=.01) for q in scores]).mean()
    if not with_u:
        return lab,lab.detach(),0,0
    unlab,pseudo,keep,details=unlabeled_loss(f,u,ur,step,3.,forced_pseudo=force,mechanism=mechanism,
                                          ordinary_std=ordinary_std)
    safe,_,_=protected_loss(lab,unlab,trainable(f))
    if force is not None and (details['forwards']!=6 or not details['partners']):
        raise ValueError('Real smoke did not traverse the Tri training branches')
    return safe,lab.detach(),unlab.detach(),int(keep.sum())


def smoke(rt):
    from copy import deepcopy
    root=Path(rt['run_root'])
    source,_,c=source_handles(rt)
    parent=None if rt.get('scratch') else model_for(rt,root,c,{},'cpu')
    yn,rn=labels(source,c,'L_s')
    raw=native(source,'L_s',len(yn))
    i=np.sort(next(batches(yn,rn,np.random.default_rng(721)))[:8])
    y,rx=tensor(yn[i]),tensor(rn[i])
    records={}
    from .tri_tuning import configure
    for name,row in study(rt)[0].items():
        f=(clean_model(rt,root,c,row,'cpu') if rt.get('scratch')
           else preserve_alignment(configure(deepcopy(parent),row),row))
        grid=clean_iq(raw[i],row.get('samples',256),row.get('chirps',CHIRPS))
        if row.get('timing_jitter'):
            shifts=[clean_iq(raw[i],row['samples'],row['chirps'],offset=d) for d in (-2,2)]
            if any(z.shape!=grid.shape or np.array_equal(z,grid) for z in shifts):
                raise ValueError('Physical source jitter is missing')
        if row.get('two_rate'):
            grid=np.concatenate((grid,clean_iq(raw[i],256)),axis=-1)
        a=tensor(grid)
        t=shape_trace(f,a)
        loss,_,_,n=_labeled_and_unlabeled(f,a,y,rx,a,rx,1,True,force=y,mechanism=terms(row),
                                         raw_ce=row.get('raw_ce',0.),ordinary_std=row.get('ordinary_std',.015),
                                         joint_rx_weight=row.get('joint_rx_weight',0.),
                                         clean_anchor_weight=row.get('clean_anchor_weight',0.),
                                         rtl_weight=row.get('rtl_weight',1.),
                                         pair_margin_weight=row.get('pair_margin_weight',0.))
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True))
        if norm<=0 or n!=len(y):raise ValueError('Failed real clean Tri gradient check')
        records[name]=dict(t,gradient=norm)
    write(root/'smoke.json',dict(status='PASS',U_truth_reads=0,target_reads=0,rows=records))


def distinct_u_batches(total,epochs,steps,batch_size,*,device,generator):
    draws=epochs*steps*batch_size
    if draws>total:
        raise ValueError('Unique source U budget exceeds the available source packets')
    return torch.randperm(total,device=device,generator=generator)[:draws].reshape(epochs,steps,batch_size)


def train(rt,name):
    root=Path(rt['run_root'])
    source,_,c=source_handles(rt)
    rows,epochs,u_count=study(rt)
    row=rows[name]
    n=row.get('samples',256)
    chirps=row.get('chirps',CHIRPS)
    cache_root=Path(rt.get('fixed_clean_cache',root/'clean_cache'))
    key,expected=cache_layout(n,chirps,c['protocol'])
    clean=cache_root/key
    if read(clean/'complete.json')!=expected:
        raise ValueError('Clean source cache not ready')
    device='cuda:0'
    out=root/'F'/name
    out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    f=clean_model(rt,root,c,row,device)
    seed_all(SEED)
    def load(key):
        raw=np.load(clean/f'{key}.npy',mmap_mode='r',allow_pickle=False)
        if row.get('two_rate'):
            anchor=np.load(cache_root/'256'/f'{key}.npy',mmap_mode='r',allow_pickle=False)
            raw=np.concatenate((raw,anchor),axis=-1)
        return tensor(raw,device)
    x,u,v=[load(key) for key in ('L_s','U_s','V')]
    yn,rn=labels(source,c,'L_s')
    y,rx=tensor(yn,device),tensor(rn,device)
    shifts=()
    if row.get('timing_jitter'):
        original=native(source,'L_s',len(yn))
        shifts=tuple(tensor(clean_iq(original,n,chirps,offset=d),device) for d in (-2,2))
        lgen=torch.Generator(device=device).manual_seed(SEED+11)
    ur=tensor(labels(source,c,'U_s'),device)
    vy,vr=(tensor(a,device) for a in labels(source,c,'V'))
    rate=row.get('lr',3e-4)
    opt=torch.optim.AdamW(f.parameters(),lr=rate,weight_decay=5e-4)
    if rt.get('scratch'):
        from math import cos, pi
        warm=row['warmup']*STEPS
        def lr_factor(step):
            if step<warm:
                return .25+.75*.5*(1+cos(pi*step/warm))
            phase=(step-warm)/(epochs*STEPS-warm)
            return 1/30.+(.25-1/30.)*.5*(1+cos(pi*phase))
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    else:
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
    ordinary_std=row.get('ordinary_std',.015)
    joint_rx_weight=row.get('joint_rx_weight',0.)
    clean_anchor_weight=row.get('clean_anchor_weight',0.)
    rtl_weight=row.get('rtl_weight',1.)
    pair_margin_weight=row.get('pair_margin_weight',0.)
    if joint_rx_weight:
        from .packet_models import cross_rx
    warmup=row.get('warmup',0)
    covered_u=(distinct_u_batches(len(u),epochs-warmup,STEPS,u_count,device=device,generator=ugen)
               if row.get('cover_u') else None)
    validation_epochs=set(VALIDATION_EPOCHS)
    if rt.get('scratch'):
        validation_epochs.update(range(warmup,epochs+1,4))
        if any(rt.get(k) for k in ('budget_study','edge_study','robust_study','phase_study')):
            validation_epochs.update((32,48,60))
        if any(rt.get(k) for k in ('anchor_study','timing_study','phase_joint_study','rtl_study','physics_study','coverage_study','chirpread_study','pairmargin_study')):
            validation_epochs.update((32,48,64,80,88))
    with (out/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            use_u=epoch>warmup
            if covered_u is None:
                order=torch.randperm(len(u),device=device,generator=ugen)[:STEPS*u_count].reshape(STEPS,u_count)
                ui=order if use_u else (None,)*STEPS
            else:
                ui=covered_u[epoch-warmup-1] if use_u else (None,)*STEPS
            sl,su,accepted=0.,0.,0
            for b,pos in zip(batches(yn,rn,rng),ui,strict=True):
                i=tensor(b,device)
                step+=1
                opt.zero_grad(set_to_none=True)
                ux,urb=(u[pos],ur[pos]) if use_u else (None,None)
                if rt.get('scratch') and not use_u:
                    f.train()
                    labeled=x[i]
                    if shifts:
                        draw=torch.rand(len(i),device=device,generator=lgen)[:,None,None,None]
                        labeled=torch.where(draw<.125,shifts[0][i],labeled)
                        labeled=torch.where(draw>.875,shifts[1][i],labeled)
                    logits,features=f(labeled)
                    loss=clean_source_ce(logits,y[i],rx[i],rtl_weight,pair_margin_weight)
                    if joint_rx_weight:
                        loss=loss+joint_rx_weight*cross_rx(features,y[i],rx[i],balance=True)[0]
                    lab,unlab,a=loss.detach(),0.,0
                else:
                    loss,lab,unlab,a=_labeled_and_unlabeled(f,x[i],y[i],rx[i],ux,urb,step,use_u,
                              mechanism=mechanism,raw_ce=row.get('raw_ce',0.),ordinary_std=ordinary_std,
                              joint_rx_weight=joint_rx_weight,clean_anchor_weight=clean_anchor_weight,
                              rtl_weight=rtl_weight,pair_margin_weight=pair_margin_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                sch.step()
                sl+=float(lab);su+=float(unlab);accepted+=a
            rec=dict(epoch=epoch,step=step,L_loss=sl/STEPS,U_loss=su/STEPS,
                     U_seen=STEPS*u_count if use_u else 0,U_accepted=accepted,
                     lr=opt.param_groups[0]['lr'])
            if epoch in validation_epochs:
                rec['V']=clean_validation(f,v,vy,vr)
                key=lambda p:(p['correct'],-p['nll'])
                # Keep the supervised trajectory for diagnosis. Only complete
                # PAMF-Tri REPL checkpoints are eligible for a scratch result.
                eligible=not rt.get('scratch') or use_u
                if eligible and (best['epoch']==0 and rt.get('scratch') or key(rec['V'])>key(best)):
                    best=rec['V'] | {'epoch':epoch}
                    torch.save(f.state_dict(),path)
            log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
            print('LORA_CLEAN95',name,epoch,step,rec.get('V',{}).get('accuracy'),accepted,flush=True)
    if step!=epochs*STEPS:
        raise ValueError('Incomplete registered clean updates')
    f.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
    training=clean_validation(f,x,y,rx)
    write(out/'complete.json',dict(status='SOURCE_FROZEN',seed=SEED,method=name,frozen=best,L=training,
                                  steps=step,clean_only=True,satellite_batches=0,
                                  ordinary_std=ordinary_std,
                                  initialization='random_seed' if rt.get('scratch') else 'same_contract_E140',
                                  target_reads=0,U_truth_reads=0))


@torch.no_grad()
def predict(rt,name):
    root=Path(rt['run_root'])
    if name not in read(root/'selection.json')['names']:
        raise ValueError('Clean test permission requires the source-frozen decision')
    _,_,c=source_handles(rt)
    target=Path(rt['native_target_root'])
    target_contract(target)
    row=study(rt)[0][name]
    f=clean_model(rt,root,c,row,'cuda:0').eval()
    path=root/'F'/name
    f.load_state_dict(torch.load(path/'best.pt',map_location='cuda:0',weights_only=True),strict=True)
    out=root/'predictions'/name
    out.mkdir(parents=True,exist_ok=False)
    for rx in TARGET:
        x=native(target,rx,LIMIT)
        codes=[]
        for i in range(0,LIMIT,64):
            y=f(tensor(clean_iq(x[i:i+64],row.get('samples',256),row.get('chirps',CHIRPS)),'cuda:0'))[0]
            if not torch.isfinite(y).all():raise ValueError('Nonfinite blind clean logits')
            codes.extend(y.argmax(1).tolist())
        np.save(out/f'{rx}__clean__direct.npy',np.asarray(codes,dtype=np.int16))
        print('LORA_CLEAN_BLIND',name,rx,flush=True)
    write(out/'complete.json',dict(method=name,seed=SEED,status='PREDICTIONS_FROZEN',
          target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=['clean']))


def dispatch(rt):
    if (rt.get('quick') or rt.get('transfer') or rt.get('timefreq')) and on_local(rt):
        raise ValueError('Quick branch study is registered for N607 only')
    root=Path(rt['run_root'])
    screens=('quick','transfer','timefreq','noise_study','drift_study','budget_study')
    gated_study=any(rt.get(k) for k in ('robust_study','phase_study','anchor_study','timing_study',
                                       'phase_joint_study','rtl_study','physics_study','coverage_study','chirpread_study','pairmargin_study'))
    if (gated_study or any(rt.get(k) for k in ('noise_study','drift_study','budget_study','edge_study'))) and not rt.get('scratch'):
        raise ValueError('The clean input comparisons require random initialization')
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    fields=tuple(k for k in ('local_parent','local_only','quick','transfer','scratch','timefreq',
                             'noise_study','drift_study','budget_study','edge_study','robust_study','phase_study','anchor_study',
                             'timing_study','phase_joint_study','rtl_study','physics_study','coverage_study','chirpread_study','pairmargin_study',
                             'fixed_clean_cache') if k in rt)
    # Training config remains source-only. Target handles travel only in the predictor config.
    launch=partial(start,module=__spec__.name,extra_keys=fields)
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        stages=('smoke',) if rt.get('fixed_clean_cache') else ('smoke','cache')
        for stage in stages:
            if launch(rt,stage,'all').wait():raise RuntimeError(f'Clean {stage} failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        names=row_names(rt)
        schedule(rt,[('train',name) for name in names],launch)
        evidence={name:read(root/'F'/name/'complete.json')['frozen'] for name in names}
        ranking=sorted(names,key=lambda name:(evidence[name]['correct'],-evidence[name]['nll']),reverse=True)
        source_only=bool(on_local(rt) or any(rt.get(k) for k in screens))
        tested=[] if source_only else list(dict.fromkeys(['control',ranking[0]]))
        best=evidence[ranking[0]]
        goal_met=best['correct']>=TARGET_SOURCE_ACCURACY*best['n']
        if gated_study:
            goal_met=best['n']==12000 and best['correct']>=ROBUST_SOURCE_CORRECT
            tested=[ranking[0]] if goal_met and not source_only else []
        write(root/'selection.json',dict(names=tested,rank=ranking,source=evidence,target_reads=0,goal_met=goal_met))
        if not tested:
            write(state,dict(status='SOURCE_DIAGNOSTIC_COMPLETE'),exclusive=False)
            return
        write(state,dict(status='PREDICTING',names=tested),exclusive=False)
        def launch_blind(runtime,kind,name,gpu=None):
            return start(runtime,kind,name,gpu,module=__spec__.name,
                         extra_keys=('native_target_root',*fields))
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
