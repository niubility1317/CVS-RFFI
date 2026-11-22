"""Learn LoRa noncarrier detail on repeated chirps from source IQ pairs."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .dura_repair import complex_iq, no_tone, power, real_iq
from .full_dura_core import iq_arrays, protected_shapes, read_gate
from .full_dura_stage import alternate_arrays
from .full_packet_data import LIMIT
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import SEEDS, handles


DETAIL_RUN='20261122-phase1-lora-dura-coherent-detail-m1-r01'
DETAIL_DOSE_RUN='20261122-phase1-lora-dura-coherent-dose-m2-r01'
DETAIL_CONFIRM_RUN='20261122-phase1-lora-dura-coherent-dose-confirm-m4-r01'
SHARED_DETAIL_RUN='20261123-phase1-lora-dura-packet-common-m1-r01'
PHASE_DETAIL_RUN='20261123-phase1-lora-dura-half-phase-residual-m1-r01'
PHASE_CONFIRM_RUN='20261123-phase1-lora-dura-half-phase-confirm-m4-r01'
SPECTRAL_PHASE_RUN='20261123-phase1-lora-dura-spectral-phase-m1-r01'
END_PHASE_RUN='20261123-phase1-lora-dura-chirp-ends-phase-m1-r01'
PHASE_RUNS=(PHASE_DETAIL_RUN,PHASE_CONFIRM_RUN,SPECTRAL_PHASE_RUN,END_PHASE_RUN)
DETAIL_CONFIRM_RUNS=(DETAIL_CONFIRM_RUN,PHASE_CONFIRM_RUN)
DETAIL_SEEDS=(SEEDS[1],)
CONFIRM_NEW_SEEDS=(SEEDS[0],SEEDS[2])
DETAIL_MODES=('direct','fixed','protect_shape','coherent_detail')
DETAIL_DOSES={'coherent_quarter':.25,'coherent_half':.50}
DETAIL_DOSE_MODES=(*DETAIL_MODES,*DETAIL_DOSES)
SHARED_DETAIL_DOSES={'packet_common_full':1.,'packet_common_half':.5}
SHARED_DETAIL_MODES=(*DETAIL_DOSE_MODES,*SHARED_DETAIL_DOSES)
PHASE_DETAIL_DOSES={'phase_residual_full':1.,'phase_residual_half':.5}
PHASE_DETAIL_MODES=(*DETAIL_DOSE_MODES,*PHASE_DETAIL_DOSES)
STEPS=1800
CAP=.15
QUALITY=.94
SHARED_QUALITY=.40
PHASE_BANDS=((1,4),(5,16),(17,48))


def conv_head(channels,outputs=2):
    return nn.Sequential(nn.Conv1d(channels,32,9,padding=4,padding_mode='reflect'),nn.SiLU(),
        nn.Conv1d(32,32,9,padding=8,dilation=2,padding_mode='reflect'),nn.SiLU(),
        nn.Conv1d(32,outputs,9,padding=4,padding_mode='reflect'))


def pair_channels(z):
    return torch.stack((z.real,z.imag),dim=-2)


def strength_ratio(delta,x):
    return power(delta).clamp_min(1e-12).sqrt()/power(x).clamp_min(1e-12).sqrt()


class PacketDetail(nn.Module):
    """Predict separate common and individual shape corrections in carrier coordinates."""

    phase=False

    def __init__(self,gate,*,shared=False):
        super().__init__()
        self.gate=gate.requires_grad_(False)
        self.shared=shared
        outputs=1 if self.phase else 2
        self.common=conv_head(3,outputs)
        self.individual=None if shared else conv_head(5,outputs)
        heads=(self.common,) if shared else (self.common,self.individual)
        for head in heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.utility=nn.Sequential(nn.Linear(13,24),nn.SiLU(),nn.Linear(24,1))
        self.register_buffer('center',torch.zeros(13))
        self.register_buffer('scale',torch.ones(13))
        self.register_buffer('threshold',torch.tensor(float('inf')))

    @torch.no_grad()
    def eligible(self,x):
        tracker=self.gate.tracker
        quality=SHARED_QUALITY if self.shared or self.phase else QUALITY
        return tracker.reference.coherence(x.flatten(0,1)).reshape(len(x),6).median(1).values>=quality

    @torch.no_grad()
    def coordinates(self,x):
        w,rot,_,_,_=self.gate.tracker.track(x)
        rms=w.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
        normalized=w/rms
        return (normalized if self.phase else no_tone(normalized)),rot,rms

    def head_inputs(self,shape):
        common=shape.mean(1)
        own=shape-common[:,None]
        spread=own.abs().square().mean(1).clamp_min(1e-8).sqrt()[:,None]
        shared=torch.cat((pair_channels(common),spread),dim=1)
        return shared,own

    def correction(self,shape,rot,rms):
        shared,own=self.head_inputs(shape)
        s=self.common(shared)
        a=no_tone(torch.complex(s[:,0],s[:,1]))
        step=a[:,None].expand_as(shape)
        if self.individual is not None:
            individual=torch.cat((pair_channels(own),shared[:,None].expand(-1,6,-1,-1)),dim=2)
            i=self.individual(individual.flatten(0,1)).reshape(len(shape),6,2,256)
            b=no_tone(torch.complex(i[:,:,0],i[:,:,1]))
            step=step+(b-b.mean(1,keepdim=True))
        delta=step*rms
        ratio=strength_ratio(delta,rms.expand_as(delta))
        shrink=(CAP/ratio).clamp_max(1.)[:,None,None]
        return step*shrink,real_iq(delta*shrink*rot)

    def proposal(self,x):
        return self.correction(*self.coordinates(x))[1]

    def features(self,x,delta):
        d=complex_iq(delta)
        r=strength_ratio(d,complex_iq(x))
        offset=d.abs().square().mean(-1).sqrt()
        variation=offset.std(1)/offset.mean(1).clamp_min(1e-6)
        return torch.cat((self.gate.features(x),r[:,None],variation[:,None]),dim=1)

    def score(self,features):
        return self.utility((features-self.center)/self.scale).flatten()

    def accepted(self,x,delta):
        return self.accepted_features(self.features(x,delta))

    def accepted_features(self,features):
        return self.score(features)>self.threshold


class PhaseDetail(PacketDetail):
    """Correct residual phase on a frozen half-dose waveform at fixed amplitude."""

    phase=True

    def proposed_angle(self,shape):
        shared,own=self.head_inputs(no_tone(shape))
        packet=self.common(shared)[:,0]
        individual=torch.cat((pair_channels(own),shared[:,None].expand(-1,6,-1,-1)),dim=2)
        local=self.individual(individual.flatten(0,1)).reshape_as(shape.real)
        return packet[:,None]+local-local.mean(1,keepdim=True)

    def bounded_angle(self,angle,shape,rms):
        magnitude=shape.abs()
        angle=angle-(angle*magnitude).sum(-1,keepdim=True)/magnitude.sum(-1,keepdim=True).clamp_min(1e-8)
        upper_bound=strength_ratio(angle*magnitude*rms,rms.expand_as(magnitude))
        return angle*(CAP/upper_bound).clamp_max(1.)[:,None,None]

    def correction(self,shape,rot,rms):
        angle=self.bounded_angle(self.proposed_angle(shape),shape,rms)
        step=shape*(torch.polar(torch.ones_like(angle),angle)-1.)
        return step,real_iq(step*rot*rms)


class SpectralPhaseDetail(PhaseDetail):
    """Limit the learned phase trajectory to the registered nonzero Fourier bins."""

    def __init__(self,gate):
        super().__init__(gate)
        bins=torch.arange(129)
        self.register_buffer('bandpass',(bins>=1)&(bins<=PHASE_BANDS[-1][1]),persistent=False)
        self.register_buffer('clean_rms_ceiling',torch.tensor(float('inf')))

    def proposed_angle(self,shape):
        angle=super().proposed_angle(shape)
        return torch.fft.irfft(torch.fft.rfft(angle)*self.bandpass,n=angle.shape[-1])

    def bounded_angle(self,angle,shape,rms):
        angle=super().bounded_angle(angle,shape,rms)
        # Keep the half-dose readout on the same branch of the complex argument.
        peak=angle.abs().amax((1,2),keepdim=True).clamp_min(1e-8)
        return angle*((torch.pi-1e-4)/peak).clamp_max(1.)

    def accepted_features(self,features):
        return super().accepted_features(features)&(features[:,-2]>self.clean_rms_ceiling)

    def band_error(self,angle):
        power=2*torch.fft.rfft(angle,norm='ortho').abs().square()/angle.shape[-1]
        return torch.stack([power[...,a:b+1].sum(-1).mean(1) for a,b in PHASE_BANDS],dim=1)

    def training_terms(self,shape,target,rot,rms):
        angle=self.bounded_angle(self.proposed_angle(shape),shape,rms)
        wanted=(target*shape.conj()).angle()
        raw=self.band_error(wanted)
        residual=self.band_error(wanted-angle)
        scale=raw+.0004
        step=shape*(torch.polar(torch.ones_like(angle),angle)-1.)
        return (raw/scale).mean(1),(residual/scale).mean(1),real_iq(step*rot*rms)


class EndPhaseDetail(SpectralPhaseDetail):
    """Learn the source-paired phase residual at the two ends of each chirp."""

    def __init__(self,gate):
        super().__init__(gate)
        n=torch.arange(256)
        distance=torch.minimum(n-64,191-n).clamp(0,15)
        window=.5+.5*torch.cos(torch.pi*distance/15)
        self.register_buffer('ends',window,persistent=False)

    def bounded_angle(self,angle,shape,rms):
        magnitude=shape.abs()
        selected=magnitude*self.ends
        mean=(angle*selected).sum(-1,keepdim=True)/selected.sum(-1,keepdim=True).clamp_min(1e-8)
        local=(angle-mean)*self.ends
        upper_bound=strength_ratio(local*magnitude*rms,rms.expand_as(magnitude))
        peak=local.abs().amax((1,2),keepdim=True).clamp_min(1e-8)
        return local*torch.minimum((CAP/upper_bound).clamp_max(1.)[:,None,None],(torch.pi-1e-4)/peak)

    def training_terms(self,shape,target,rot,rms):
        magnitude=shape.abs()
        wanted=(target*shape.conj()).angle()
        filtered=torch.fft.irfft(torch.fft.rfft(wanted)*self.bandpass,n=wanted.shape[-1])
        expected=self.bounded_angle(filtered,shape,rms)
        actual=self.bounded_angle(self.proposed_angle(shape),shape,rms)
        raw=power(expected*magnitude)
        error=power((expected-actual)*magnitude)
        step=shape*(torch.polar(torch.ones_like(actual),actual)-1.)
        return raw,error,real_iq(step*rot*rms)


def detail_model(root,gate):
    if root.name==END_PHASE_RUN:
        return EndPhaseDetail(gate)
    if root.name==SPECTRAL_PHASE_RUN:
        return SpectralPhaseDetail(gate)
    if root.name in PHASE_RUNS:
        return PhaseDetail(gate)
    return PacketDetail(gate,shared=root.name==SHARED_DETAIL_RUN)


def shape_error(error,*,shared=False):
    common=error.mean(1).abs().square().mean(-1)
    return common if shared else power(error)+.5*common


def training_terms(model,x,clean):
    shape,rot,rms=model.coordinates(x)
    target,_,_=model.coordinates(clean)
    if isinstance(model,SpectralPhaseDetail):
        return model.training_terms(shape,target,rot,rms)
    step,delta=model.correction(shape,rot,rms)
    if model.phase:
        target=torch.polar(shape.abs(),target.angle())
    raw=detail_error(model,shape-target)
    error=detail_error(model,shape+step-target)
    return raw,error,delta


def detail_error(model,error):
    return shape_error(no_tone(error) if model.phase else error,shared=model.shared)


def record(c,seed,*,shared=False,phase=False,banded=False,ends=False):
    r=dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,steps=STEPS,
                draw_run=DRAW_RUN,quality=QUALITY,cap=CAP,F_reads=0,TX_reads=0,target_reads=0)
    if shared:
        r.update(quality=SHARED_QUALITY,component='chirp_common',paired_draws=True)
    if phase:
        baseline=DETAIL_DOSE_RUN if seed in DETAIL_SEEDS else DETAIL_CONFIRM_RUN
        r.update(quality=SHARED_QUALITY,component='phase',baseline=baseline)
    if banded:
        r.update(component='spectral_phase',phase_bands=[list(b) for b in PHASE_BANDS],clean_rms_guard=True)
    if ends:
        r.update(component='chirp_ends_phase',phase_bands=[list(b) for b in PHASE_BANDS],
                 center=[64,192],taper=15,clean_rms_guard=True)
    return r


def paired_common_loss(model,first,second,clean):
    x=torch.cat((first,second))
    shape,rot,rms=model.coordinates(x)
    target=model.coordinates(clean)[0].mean(1)
    direct=shape.mean(1)
    recovered=direct+model.correction(shape,rot,rms)[0].mean(1)
    x0,x1=direct.chunk(2)
    y0,y1=recovered.chunk(2)
    scale=((x0-target).abs().square().mean(1)+(x1-target).abs().square().mean(1))*.5+.0004
    return ((y0-y1).abs().square().mean(1)/scale).mean()


def source_inputs(rt,device):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    return c,iq_arrays(cache,device),alternate_arrays(root,device)


def tracker_gate(root,c,device):
    from .full_dura import RUN
    return read_gate(root.parent/RUN/'G/gate',c,device)


@torch.no_grad()
def pairs_for_gate(model,clean,practical):
    features=[]
    improvement=[]
    for a,x in zip(clean.split(128),practical.split(128),strict=True):
        keep=model.eligible(x)
        if keep.any():
            x=x[keep]
            raw,error,d=training_terms(model,x,a[keep])
            features.append(model.features(x,d))
            improvement.append((raw-error)/raw.clamp_min(.0001))
    return torch.cat(features),torch.cat(improvement)


@torch.no_grad()
def clean_features(model,clean):
    features=[]
    for x in clean.split(128):
        keep=model.eligible(x)
        if keep.any():
            x=x[keep]
            features.append(model.features(x,model.proposal(x)))
    return torch.cat(features)


def fit_utility(model,l,u,sat,alt,held,*,bypass_reference=None):
    fit=held[::2]
    check=held[1::2]
    l_clean,u_clean=(l,u) if bypass_reference is None else bypass_reference
    observations=[pairs_for_gate(model,u[fit],a[fit]) for a in (sat,alt)]
    clean_fit=clean_features(model,u_clean[fit])
    x=torch.cat((clean_fit,*(f for f,_ in observations)))
    label=torch.cat((x.new_zeros(len(clean_fit)),*((e>.05).float() for _,e in observations)))
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(1e-5))
    opt=torch.optim.AdamW(model.utility.parameters(),lr=.004,weight_decay=.01)
    history=[]
    for _ in range(30):
        losses=[]
        for i in torch.randperm(len(x),device=u.device).split(512):
            z=label[i]
            loss=F.binary_cross_entropy_with_logits(model.score(x[i]),z,weight=torch.where(z==0,2.,1.))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    with torch.no_grad():
        controls=[clean_features(model,a) for a in (l_clean,u_clean)]
        top=max(model.score(features).max() for features in controls)
        model.threshold.copy_(top+1e-4)
        if isinstance(model,SpectralPhaseDetail):
            ceiling=max(features[:,-2].max() for features in controls)
            model.clean_rms_ceiling.copy_(ceiling+1e-4)
        audits=[]
        for a in (sat,alt):
            f,e=pairs_for_gate(model,u[check],a[check])
            picked=model.accepted_features(f)
            audits.append(dict(high_quality=len(f),selected=int(picked.sum()),beneficial=int((e[picked]>0).sum()),
                                relative_change_sum=float(e[picked].sum())))
    guard=dict(clean_rms_ceiling=float(model.clean_rms_ceiling)) if isinstance(model,SpectralPhaseDetail) else {}
    return dict(loss=history,fit_n=len(x),beneficial_fit_n=int(label.sum()),threshold=float(model.threshold),
                withheld_U_pairs=audits,**guard)


def train_detail(rt,seed):
    root=Path(rt['run_root'])
    seeds=CONFIRM_NEW_SEEDS if root.name in DETAIL_CONFIRM_RUNS else DETAIL_SEEDS
    if seed not in seeds:
        raise ValueError('Unregistered high-coherence DURA seed')
    dev='cuda:0'
    torch.manual_seed(seed)
    c,(l,u,sat),(alt_l,alt_u)=source_inputs(rt,dev)
    shared=root.name==SHARED_DETAIL_RUN
    phase=root.name in PHASE_RUNS
    gate=tracker_gate(root,c,dev)
    model=detail_model(root,gate).to(dev)
    held=torch.arange(0,len(u),5,device=dev)
    signals=(l[1:].flatten(0,1),sat,alt_l,alt_u)
    sources=(l[0],u,l[0],u)
    identity=None
    if phase:
        base,prior=phase_baseline_models(root,c,seed,dev)

        def previous(array):
            return torch.cat([half_baseline(base,prior,x)[0] for x in array.split(96)])

        signals=tuple(previous(array) for array in signals)
        identity=(previous(l[0]),previous(u))
        del base,prior
    masks=[torch.cat([model.eligible(x) for x in data.split(256)]) for data in signals]
    for i in (1,3):
        masks[i][held]=False
    pools=[torch.where(mask)[0] for mask in masks]
    if min(map(len,pools))<24:
        raise ValueError('Insufficient high-coherence source support')
    if shared:
        paired=torch.where(masks[1]&masks[3])[0]
        if len(paired)<24:
            raise ValueError('Insufficient matched practical draws for shared correction')
    out=root/'G'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    parameters=list(model.common.parameters())
    if model.individual is not None:
        parameters.extend(model.individual.parameters())
    opt=torch.optim.AdamW(parameters,lr=.0006,weight_decay=.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00006)
    counts=(6,8,4,6)
    allowed=torch.where(torch.arange(len(u),device=dev)%5!=0)[0]
    identity_pool=u if identity is None else identity[1]
    with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            ids=[pool[torch.randint(len(pool),(n,),device=dev)] for pool,n in zip(pools,counts,strict=True)]
            if shared:
                pairs=paired[torch.randint(len(paired),(counts[3],),device=dev)]
                ids[1][:len(pairs)]=pairs
                ids[3]=pairs
            observation=torch.cat([a[i] for a,i in zip(signals,ids,strict=True)])
            ground=torch.cat([a[i%len(a)] for a,i in zip(sources,ids,strict=True)])
            clean=identity_pool[allowed[torch.randint(len(allowed),(8,),device=dev)]]
            raw,error,_=training_terms(model,observation,ground)
            clean_error=detail_error(model,model.correction(*model.coordinates(clean))[0])
            objective=(error/(raw+.0004)).mean()+.15*(clean_error/.0004).mean()
            if shared:
                consistency=paired_common_loss(model,sat[pairs],alt_u[pairs],u[pairs])
                objective=objective+.15*consistency
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
            opt.step()
            sch.step()
            if step==1 or step%100==0:
                entry=dict(step=step,loss=float(objective.detach()),raw=float(raw.mean()),
                           restored=float(error.detach().mean()),identity=float(clean_error.detach().mean()))
                if shared:
                    entry.update(paired_draw_consistency=float(consistency.detach()))
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                log.flush()
                print('LORA_COHERENT_DETAIL',seed,entry,flush=True)
    utility=fit_utility(model,l[0],u,signals[1],signals[3],held,bypass_reference=identity)
    shared_log=dict(paired_U_pool=len(paired),pair_count=STEPS*counts[3]) if shared else {}
    write(out/'source_support.json',dict(training_pool=list(map(len,pools)),withheld_U=len(held),
          practical_exposures=STEPS*sum(counts),clean_exposures=STEPS*8,utility=utility,**shared_log))
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',record(c,seed,shared=shared,phase=phase,
                                    banded=root.name==SPECTRAL_PHASE_RUN,ends=root.name==END_PHASE_RUN))


def load_detail(root,c,seed,device):
    if root.name==DETAIL_DOSE_RUN:
        root=root.parent/DETAIL_RUN
    out=root/'G'/str(seed)
    shared=root.name==SHARED_DETAIL_RUN
    phase=root.name in PHASE_RUNS
    if read(out/'complete.json')!=record(c,seed,shared=shared,phase=phase,
                                        banded=root.name==SPECTRAL_PHASE_RUN,ends=root.name==END_PHASE_RUN):
        raise ValueError('LoRa detail checkpoint source ancestry differs')
    gate=tracker_gate(root,c,device)
    model=detail_model(root,gate).to(device)
    model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True),strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def extra_correction(detail,x,old):
    ids=torch.where(detail.eligible(x)&~old)[0]
    if not len(ids):
        return ids,x[:0]
    wave=x[ids]
    delta=detail.proposal(wave)
    extra=detail.accepted(wave,delta)
    if not torch.isfinite(delta).all() or (strength_ratio(complex_iq(delta),complex_iq(wave))>CAP+1e-5).any():
        raise ValueError('Invalid high-coherence LoRa detail correction')
    return ids[extra],wave[extra]+delta[extra]


def scale_detail(raw,corrected,amount):
    return raw+amount*(corrected-raw)


@torch.no_grad()
def restore(base,detail,x):
    fixed,strength=base(x,with_gate=True)
    old=strength>0
    restored=x.clone()
    if old.any():
        restored[old]=protected_shapes(base,x[old],fixed[old])[0]
    selected,adjusted=extra_correction(detail,x,old)
    restored[selected]=adjusted
    active=old.clone()
    active[selected]=True
    return fixed,restored,old,active


def phase_baseline_models(root,c,seed,device):
    from .full_dura import RUN, engines
    base,_=engines(root.parent/RUN,c,seed,device)
    prior=phase_baseline_detail(root,c,seed,device)
    return base.requires_grad_(False),prior


def phase_baseline_detail(root,c,seed,device):
    name=DETAIL_RUN if seed in DETAIL_SEEDS else DETAIL_CONFIRM_RUN
    return load_detail(root.parent/name,c,seed,device)


@torch.no_grad()
def half_baseline(base,prior,x):
    _,restored,protected,active=restore(base,prior,x)
    extra=active&~protected
    restored[extra]=scale_detail(x[extra],restored[extra],.5)
    return restored,active


def phase_dose(x,adjusted,amount):
    observed=complex_iq(x)
    angle=(complex_iq(adjusted)*observed.conj()).angle()
    return real_iq(observed*torch.polar(torch.ones_like(angle),amount*angle))


def detail_smoke(rt):
    from .full_dura import RUN, engines
    root=Path(rt['run_root'])
    _,cache,c=handles(rt)
    seed=(CONFIRM_NEW_SEEDS if root.name in DETAIL_CONFIRM_RUNS else DETAIL_SEEDS)[0]
    base,_=engines(root.parent/RUN,c,seed,'cpu')
    if root.name==DETAIL_DOSE_RUN:
        detail=load_detail(root,c,seed,'cpu')
        draw=load_draw(root.parent/DRAW_RUN,'L_s')
        x=tensor(draw[::16])
        extra,wave=extra_correction(detail,x,base.strength(x)>0)
        if not len(extra):
            raise ValueError('Frozen source detail smoke has no utility-gated proposals')
        for amount in DETAIL_DOSES.values():
            adjusted=scale_detail(x[extra],wave,amount)
            ratio=strength_ratio(complex_iq(adjusted-x[extra]),complex_iq(x[extra]))
            if (ratio>amount*CAP+1e-5).any() or torch.equal(adjusted,x[extra]):
                raise ValueError('Frozen detail dose did not satisfy its RMS bound')
        write(root/'smoke.json',dict(status='PASS',seed=seed,extra_active=len(extra),F_reads=0,
                                   TX_reads=0,target_reads=0,doses=DETAIL_DOSES))
        return
    shared=root.name==SHARED_DETAIL_RUN
    phase=root.name in PHASE_RUNS
    model=detail_model(root,base.gate)
    l=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)[0]
    old=np.load(cache/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)[::32]
    a=load_draw(root.parent/DRAW_RUN,'L_s')
    x=tensor(a[::16])
    if phase:
        prior=phase_baseline_detail(root,c,seed,'cpu')
        x=half_baseline(base,prior,x)[0]
    keep=torch.where(model.eligible(x))[0][:8]
    if len(keep)<8:
        raise ValueError('Real high-coherence source data are missing')
    x=x[keep]
    clean=tensor(l[::16][keep.tolist()])
    raw,error,_=training_terms(model,x,clean)
    error.mean().backward()
    params=list(model.common.parameters())
    if model.individual is not None:
        params.extend(model.individual.parameters())
    grad=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True))
    if not grad>0 or not torch.equal(raw,error):
        raise ValueError('Source detail gradient or initial identity failed')
    torch.optim.SGD(params,lr=.03).step()
    clean_sample=tensor(l[::61])
    fixed,updated,old_gate,new_gate=restore(base,model,clean_sample)
    delta=model.proposal(x)
    if (old_gate.any() or new_gate.any() or not torch.equal(fixed,clean_sample)
            or not torch.equal(updated,clean_sample) or not torch.isfinite(delta).all()
            or not (delta!=0).any() or not (base.strength(tensor(old))>0).any()):
        raise ValueError('Real source detail bypass, bound, or protected control failed')
    if phase:
        adjusted=phase_dose(x,x+delta,.5)
        if not torch.allclose(complex_iq(adjusted).abs(),complex_iq(x).abs(),atol=3e-6,rtol=1e-5):
            raise ValueError('Phase-only source correction changed amplitude')
    observations=(dict(source_observations=len(x),shared=shared,phase=phase) if shared or phase
                  else dict(high_coherence=len(x)))
    write(root/'smoke.json',dict(status='PASS',seed=seed,clean_exact=len(clean_sample),
              grad=grad,F_reads=0,TX_reads=0,target_reads=0,**observations))


@torch.no_grad()
def source_wave_detail(rt,seed):
    from .full_dura import RUN, engines
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    base,_=engines(root.parent/RUN,c,seed,dev)
    detail=load_detail(root,c,seed,dev)
    arr=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    clean=tensor(arr[0,::4],dev)
    if detail.phase:
        prior=phase_baseline_detail(root,c,seed,dev)
        rows={}
        for i,v in enumerate(VIEWS):
            old_count=count=beneficial=0
            before=after=0.
            for x,target in zip(tensor(arr[i,::4],dev).split(48),clean.split(48),strict=True):
                half,old=half_baseline(base,prior,x)
                selected,_=extra_correction(detail,half,torch.zeros_like(old))
                old_count+=int(old.sum())
                if len(selected):
                    raw,restored,_=training_terms(detail,half[selected],target[selected])
                    count+=len(selected)
                    beneficial+=int((restored<raw).sum())
                    before+=float(raw.sum())
                    after+=float(restored.sum())
            rows[v]=dict(n=len(clean),prior_active=old_count,phase_selected=count,
                         phase_improved=beneficial,phase_before=before,phase_after=after)
        write(root/'source_wave'/f'{seed}.json',dict(seed=seed,records=rows,target_reads=0,TX_reads=0,F_reads=0))
        return
    refs=[detail.coordinates(y)[0] for y in clean.split(48)]
    rows={}
    for i,v in enumerate(VIEWS):
        total=high=extra=better=old=clean_changed=0
        raw_error=restored_error=0.
        for x,ref in zip(tensor(arr[i,::4],dev).split(48),refs,strict=True):
            _,output,baseline,active=restore(base,detail,x)
            high+=int(detail.eligible(x).sum())
            extra_mask=active&~baseline
            a=shape_error(detail.coordinates(x)[0]-ref,shared=detail.shared)
            b=shape_error(detail.coordinates(output)[0]-ref,shared=detail.shared)
            extra+=int(extra_mask.sum())
            better+=int((b[extra_mask]<a[extra_mask]).sum())
            raw_error+=float(a[extra_mask].sum())
            restored_error+=float(b[extra_mask].sum())
            old+=int(baseline.sum())
            clean_changed+=int((output!=x).flatten(1).any(1).sum())
            total+=len(x)
        rows[v]=dict(n=total,high_coherence=high,existing_active=old,extra_active=extra,
                     extra_improved=better,extra_raw_error=raw_error,extra_restored_error=restored_error,
                     IQ_changed=clean_changed)
    write(root/'source_wave'/f'{seed}.json',dict(seed=seed,records=rows,target_reads=0,TX_reads=0,F_reads=0))


@torch.no_grad()
def predict_detail(root,target,c,seed,reader,device):
    from .full_dura import PROJECTED_MODES, RUN, STACK_RUN, engines
    dose=root.name==DETAIL_DOSE_RUN
    shared=root.name==SHARED_DETAIL_RUN
    phase=root.name in PHASE_RUNS
    extended=shared or phase
    stack=root.parent/(DETAIL_DOSE_RUN if extended else DETAIL_RUN if dose else STACK_RUN)
    if root.name==PHASE_CONFIRM_RUN:
        stack=root.parent/DETAIL_CONFIRM_RUN
    if read(stack/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('The frozen protected DURA controls are missing')
    old=stack/'predictions'/str(seed)
    old_receipt=read(old/'complete.json')
    controls=old_receipt.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=seed,target_truth_read=False,receivers=list(TARGET),
                  views=list(VIEWS),per_receiver=LIMIT)
    comparison_modes=DETAIL_DOSE_MODES if extended else DETAIL_MODES if dose else PROJECTED_MODES
    if old_receipt!=dict(expected,modes=list(comparison_modes)):
        raise ValueError('The frozen Tri-set and DURA controls do not match')
    base,_=engines(root.parent/RUN,c,seed,device)
    detail=load_detail(root,c,seed,device)
    prior=phase_baseline_detail(root,c,seed,device) if phase else None
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    records={}
    old_modes=DETAIL_DOSE_MODES if extended else DETAIL_MODES if dose else DETAIL_MODES[:-1]
    if phase:
        amounts=PHASE_DETAIL_DOSES
    elif shared:
        amounts=SHARED_DETAIL_DOSES
    else:
        amounts={} if dose else {'coherent_detail':1.}
        if dose or root.name==DETAIL_CONFIRM_RUN:
            amounts.update(DETAIL_DOSES)
    new_modes=tuple(amounts)
    for rx in TARGET:
        waves=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if waves.shape!=(7,LIMIT,6,2,256) or waves.dtype!=np.float32:
            raise ValueError('High-coherence target IQ differs')
        for i,v in enumerate(VIEWS):
            arrays=[np.load(old/f'{rx}__{v}__{m}.npy',allow_pickle=False) for m in old_modes]
            if any(a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=10 for a in arrays):
                raise ValueError('Invalid frozen protected DURA predictions')
            fallback='coherent_half' if phase else 'protect_shape'
            updated=[arrays[old_modes.index(fallback)].copy() for _ in new_modes]
            n_old=n_extra=n_overlap=0
            for step in range(0,LIMIT,48):
                x=tensor(waves[i,step:step+48],device)
                if phase:
                    x,baseline=half_baseline(base,prior,x)
                    extra,result=extra_correction(detail,x,torch.zeros_like(baseline))
                    n_overlap+=int(baseline[extra].sum())
                else:
                    baseline=base.strength(x)>0
                    extra,result=extra_correction(detail,x,baseline)
                if len(extra):
                    apply_dose=phase_dose if phase else scale_detail
                    waves_to_read=[result if amount==1. else apply_dose(x[extra],result,amount)
                                   for amount in amounts.values()]
                    logits=reader(torch.cat(waves_to_read))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('High-coherence DURA produced nonfinite logits')
                    ids=np.asarray(extra.tolist(),dtype=np.int64)+step
                    for a,labels in zip(updated,logits.argmax(1).chunk(len(updated)),strict=True):
                        a[ids]=labels.tolist()
                    n_extra+=len(ids)
                n_old+=int(baseline.sum())
            key=f'{rx}/{v}'
            active_key='detail_active' if phase else 'active'
            if controls[key][active_key]!=n_old:
                raise ValueError('The existing protected branch active count changed')
            if dose and controls[key]['detail_extra']!=n_extra:
                raise ValueError('Frozen dose comparison changed the detail gate')
            if phase:
                records[key]=dict(controls[key],phase_active=n_old+n_extra-n_overlap,
                                  phase_extra=n_extra,phase_overlap=n_overlap)
            elif shared:
                records[key]=dict(controls[key],shared_active=n_old+n_extra,shared_extra=n_extra)
            else:
                records[key]=dict(controls[key],detail_active=n_old+n_extra,detail_extra=n_extra)
            for m,a in zip((*old_modes,*new_modes),(*arrays,*updated),strict=True):
                np.save(out/f'{rx}__{v}__{m}.npy',a)
        print('LORA_DETAIL_DURA_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=[*old_modes,*new_modes],activation=records))
