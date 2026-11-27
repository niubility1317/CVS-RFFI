"""Source-only staged training and raw inference for 8192-point LoRa."""

import importlib
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.data import BANDWIDTH_HZ
from experiments.pamf_lora_main.w24_clean_geometry import clean_geometry_risk, clean_prototypes
from experiments.pamf_lora_main.w24_aperture_reliability import InvisibleIQ, LocalizedWhite, VisibleIQ, reliability
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.w24_native_dura import (
    NativeDura, NATIVE_LENGTH, STAGES, action_loss, capture_scale, coarse_loss, iq_error, source_pair,
)
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair


DURA_RUN = '20261127-phase1-lora-raw8192-r02-r07-r17-s2026113009-r01'
TASK_RUN = '20261127-phase1-lora-raw8192-e104-strong-s2026113011-r01'
TASK = 'raw_e104_strong'
REL_RUN = '20261127-phase1-lora-raw8192-local-reliability-s2026113015-r01'
REL_ARMS = ('chirp_reliability','local_reliability')
VISIBLE_RUN = '20261127-phase1-lora-raw8192-visible-wave-s2026113017-r01'
WAVE_ARMS = ('full_wave_native','visible_wave_native')
GEOMETRY_RUN = '20261127-phase1-lora-raw8192-clean-margin-s2026113021-r01'
GEOMETRY_ARMS = ('clean_anchor_native','rival_margin_native')
GEOMETRY_WEIGHT = .35
TASK_STAGES = (TASK,*REL_ARMS,*WAVE_ARMS,*GEOMETRY_ARMS)
RUNS = {DURA_RUN: STAGES, TASK_RUN: (TASK,), REL_RUN: REL_ARMS, VISIBLE_RUN: WAVE_ARMS, GEOMETRY_RUN: GEOMETRY_ARMS}
STAGE_RUN = {stage:run for run,stages in RUNS.items() for stage in stages}
SEEDS = {DURA_RUN: 2026113009, TASK_RUN: 2026113011, REL_RUN: 2026113015,VISIBLE_RUN: 2026113017,GEOMETRY_RUN: 2026113021}
STEPS = dict(zip((*STAGES,*TASK_STAGES),(10000,10000,12000,18000,15000,15000,15000,15000,15000,15000),strict=True))
BATCH_ROLE = 8
DURA_LR = 2e-4
TASK_LR = 4e-5
ISOLATION = ('U_TX_reads','V_reads','target_reads','practical_reads','trace_reads')


class NativeTaskRestorer(nn.Module):
    def __init__(self, tracker, parent=None, *, projector=None, localized=False):
        super().__init__()
        self.tracker = tracker
        self.projector, self.localized = projector, localized
        self.core = PacketUNet(across_chirps=True,center=1024,with_reliability=projector is not None)
        if parent is not None:
            state = parent.state_dict()
            pos = state['temporal_position'].T.unsqueeze(0)
            state['temporal_position'] = F.interpolate(pos,size=128,mode='linear',align_corners=False)[0].T.contiguous()
            self.core.load_state_dict(state,strict=True)

    def forward(self, raw):
        packets = raw.unflatten(-1,(8,1024))
        with torch.no_grad():
            framed, rotation, coherence, *_ = self.tracker.track(torch.stack((packets.real,packets.imag),dim=2))
            rms = framed.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
            observed = framed / rms
            edges = torch.cat((packets[..., :24],packets[..., -24:]),dim=-1)
            cue = (None if self.projector is None else
                   reliability(self.projector(raw),rms,localized=self.localized))
        corrected = self.core(observed,coherence.unsqueeze(-1),rms,edges,reliability=cue)[-1]
        return raw + .5 * ((corrected-observed) * rotation * rms).flatten(-2)


def task_model(parent, device, **kwargs):
    archive=importlib.import_module('lora_wide_dura_archive.dura_w24_model')
    tracker=archive.PacketRestorer('consensus',samples=1024).to(device).eval().requires_grad_(False)
    t=torch.arange(1024,dtype=torch.float64,device=device)/1e6
    tracker.reference.chirp = torch.exp(2j*torch.pi*BANDWIDTH_HZ*(t.square()/.002048-t/2)).to(torch.complex64)
    return NativeTaskRestorer(tracker,parent,**kwargs).to(device)


def build_projector(frontend,device):
    return InvisibleIQ(frontend.indices,frontend.taps,frontend.edges,width=NATIVE_LENGTH).to(device)


def _proof(stage,contract,model_proof):
    if (model_proof['stage'],model_proof['steps'],model_proof['raw_length'],model_proof['protocol']) != (
            stage,STEPS[stage],NATIVE_LENGTH,contract['protocol']):
        raise ValueError('Unexpected native frozen training record')
    if any(model_proof[k] for k in ISOLATION):
        raise ValueError('Unexpected data access in native recovery training')
    if model_proof['F_reads'] != (stage in TASK_STAGES):
        raise ValueError('Unexpected native recovery teacher')
    if stage == TASK and (model_proof['F_trainable'] or model_proof['initialization'] != 'same_source_w24_length_adapted'):
        raise ValueError('Unexpected E104 trainable state or parent')
    if stage in REL_ARMS and (model_proof['F_trainable']
            or model_proof['initialization']!=TASK_RUN
            or model_proof['localized_reliability']!=(stage==REL_ARMS[-1])
            or model_proof['source_envelope_fraction']!=.5):
        raise ValueError('Unexpected native reliability parent or branch')
    if stage in WAVE_ARMS and (model_proof['F_trainable']
            or model_proof['initialization']!=TASK_RUN
            or model_proof['visible_waveform']!=(stage==WAVE_ARMS[-1])):
        raise ValueError('Unexpected native waveform objective or parent')
    if stage in GEOMETRY_ARMS and (model_proof['F_trainable']
            or model_proof['initialization']!=TASK_RUN
            or model_proof['relative_clean_geometry']!=(stage==GEOMETRY_ARMS[-1])
            or model_proof['clean_prototype_source_count']!=contract['counts']['L_s']):
        raise ValueError('Unexpected native clean geometry objective or parent')


def load_stage(root, stage, contract, module, device):
    folder=root/stage
    ckpt=torch.load(folder/'model.pt',map_location=device,weights_only=True)
    proof=ckpt['proof']
    _proof(stage,contract,proof)
    summary=read(folder/'complete.json')
    if (ckpt['contract'] != contract or summary['status']!='SOURCE_FROZEN'
            or read(folder/'smoke.json')['status']!='PASS'
            or any(summary[k] != v for k,v in proof.items())):
        raise ValueError('Expected same-contract, complete source freeze')
    module.load_state_dict(ckpt['model'],strict=True)


def dura_model(root, contract, device, stage=STAGES[-1], *, parents_only=False):
    model=NativeDura().to(device)
    order=STAGES[:STAGES.index(stage)+(not parents_only)]
    for name,part in zip(order,(model.coarse,model.estimator,model.priors)):
        load_stage(root,name,contract,part,device)
    return model


def training_task_parts(joint, model, reader, frontend, mix, raw, label, rng, *, waveform_projector=None, geometry=None):
    n=len(label)
    raw=raw/capture_scale(raw)
    observed=mix(raw.unflatten(-1,(8,1024)),rng).flatten(-2)
    restored=model(observed)
    clean_out=model(raw[:n])
    clean_ap, _=joint.normalize_aperture(frontend.aperture(raw))
    repaired_ap, _=joint.normalize_aperture(frontend.aperture(restored))
    clean_end, _=joint.normalize_aperture(frontend.aperture(clean_out))
    logits, feature=reader(torch.cat((repaired_ap[:n],clean_ap[:n],clean_end)))
    predict, clean_logits, clean_prediction=logits.split(n)
    lx=F.cross_entropy(predict,label)
    lc=.5*(F.cross_entropy(clean_logits,label)+F.cross_entropy(clean_prediction,label))
    teacher=clean_logits.detach().softmax(-1)
    kl=.5*F.kl_div(clean_prediction.log_softmax(-1),teacher,reduction='batchmean')
    error=restored-raw
    wave_ref=raw
    if waveform_projector is not None:
        error=VisibleIQ.apply(error,waveform_projector)
        wave_ref=raw-waveform_projector(raw)
    error_spectral=torch.fft.fft(error,norm='ortho').abs().square()
    power=torch.fft.fft(wave_ref,norm='ortho').abs().square()
    spectral=(error_spectral/(.001+power+.03*power.mean(-1,keepdim=True))).log1p().mean()
    mse=error.abs().square().mean()
    fidelity=(clean_out-raw[:n]).abs().square().mean()
    pseudo, accepted=joint.clean_pseudo_loss(reader,reader,clean_ap[n:],repaired_ap[n:])
    parts=(lx,lc,kl,spectral,mse,fidelity,pseudo,accepted)
    if geometry is not None:
        prototypes, relative=geometry
        estimated_feature,clean_feature,_=feature.split(n)
        parts+=(clean_geometry_risk(estimated_feature,clean_feature,label,prototypes,relative=relative),)
    return torch.stack(parts)


def fit(joint, config, stage, *, smoke, device='cuda:0'):
    rt=read(config)
    if set(rt)!=set(joint.FIT_FIELDS):
        raise ValueError('Unexpected native train runtime role')
    folder,source=Path(rt['run_root']),Path(rt['source_root'])
    run=STAGE_RUN[stage]
    if folder.parent.name!=run or folder.name!=stage or stage not in RUNS[run]:
        raise ValueError('Unexpected isolated native stage root')
    is_task=stage in TASK_STAGES
    seed=SEEDS[run]+(STAGES.index(stage) if stage in STAGES else 0)
    torch.manual_seed(seed)
    count=2 if smoke and stage not in GEOMETRY_ARMS else BATCH_ROLE
    steps=1 if smoke else STEPS[stage]
    order=np.random.default_rng(seed)
    rng=torch.Generator(device=device).manual_seed(seed)
    probes=torch.Generator(device=device).manual_seed(seed+11)
    waveform_projector=None
    if is_task:
        contract,reader,g0,frontend=joint.boot(rt,device)
        if stage==TASK:
            parent=joint.continuation_parent(rt,contract,reader,g0,device,joint.RUN_STRONG)
            model=task_model(parent,device)
        else:
            base=task_model(None,device)
            load_stage(Path(rt['project'])/'runs'/TASK_RUN,TASK,contract,base.core,device)
            model=base
            if stage in REL_ARMS:
                projector=build_projector(frontend,device)
                model=task_model(None,device,projector=projector,localized=stage==REL_ARMS[-1])
                state=base.core.state_dict()
                state['reliability_stem.weight']=model.core.reliability_stem.weight.detach().clone()
                model.core.load_state_dict(state,strict=True)
                if not smoke:
                    del base
            if stage==WAVE_ARMS[-1]:
                waveform_projector=build_projector(frontend,device)
        trained=model.core
        reader.requires_grad_(False).eval()
        original=importlib.import_module('lora_wide_dura_archive.dura_w24_model').condition_waveforms
        shape = (LocalizedWhite(torch.Generator(device=device).manual_seed(seed+2))
                 if stage in REL_ARMS else None)
        mix=joint.StrongSourceMix(original,torch.Generator(device=device).manual_seed(seed+1),
                                  device,enabled=True,noise_shape=shape)
        labels=joint.training_labels(source,contract)
    else:
        contract=source_contract(source)
        model=dura_model(folder.parent,contract,device,stage,parents_only=True)
        model.requires_grad_(False).eval()
        trained=dict(zip(STAGES,(model.coarse,model.estimator,model.priors),strict=True))[stage]
        trained.requires_grad_(True).train()
    l_iq=source_role(source,contract,'L_s')[0]
    u_iq=source_role(source,contract,'U_s')[0]
    geometry=None
    if stage in GEOMETRY_ARMS:
        geometry=(clean_prototypes(joint,reader,frontend,l_iq,labels,device),stage==GEOMETRY_ARMS[-1])

    optimizer=torch.optim.AdamW(trained.parameters(),lr=TASK_LR if is_task else DURA_LR,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=2e-6 if is_task else 2e-5)
    accepted=torch.zeros((),device=device)
    weights=torch.tensor((1.,.65,.3,.30,.25,1.5,1.,0.) + ((GEOMETRY_WEIGHT,) if geometry is not None else ()),
                         device=device)
    started=monotonic()
    fit_gradient = supervised = pseudo_gradient = geometry_gradient = None
    for step in range(1,steps+1):
        l=order.choice(len(l_iq),count,replace=False)
        u=order.choice(len(u_iq),count,replace=False)
        raw=tensor(np.concatenate((l_iq[l],u_iq[u])).view(np.float32),device).view(torch.complex64)
        optimizer.zero_grad(set_to_none=True)
        if is_task:
            label=tensor(labels[l],device).long()
            if smoke and stage in REL_ARMS:
                with torch.no_grad():
                    torch.testing.assert_close(model(raw),base(raw),rtol=0.,atol=0.)
                    missing=projector(raw)
                    scale=raw.abs().square().mean(-1,keepdim=True).sqrt()
                    residue=frontend.aperture(missing/scale).abs().max()
                    if not residue < 3e-6:
                        raise ValueError('Unstable W24-invisible native component')
            parts=training_task_parts(joint,model,reader,frontend,mix,raw,label,rng,
                                      waveform_projector=waveform_projector,geometry=geometry)
            accepted+=parts[7].detach()
            loss=parts@weights
            if smoke:
                supervised=joint.norm_gradient(torch.autograd.grad(parts[0],trained.parameters(),retain_graph=True))
                pseudo_gradient=joint.norm_gradient(torch.autograd.grad(parts[6],trained.parameters(),retain_graph=True))
                if not (0<supervised<float('inf') and 0<pseudo_gradient<float('inf')):
                    raise ValueError('Actual E104 L_s and accepted clean U_s must train native IQ')
                if geometry is not None:
                    geometry_gradient=joint.norm_gradient(torch.autograd.grad(
                        parts[8],trained.parameters(),retain_graph=True))
                    if not 0<geometry_gradient<float('inf'):
                        raise ValueError('Clean geometry loss has no finite restoration gradient')
        else:
            clean,observed,exact,variance=source_pair(raw,rng)
            incoming=pair(observed)
            if stage==STAGES[0]:
                loss=coarse_loss(complex_iq(model.coarse(incoming)),clean)
            else:
                coarse=model.coarse(incoming)
                estimated,noise=model.estimator(incoming,coarse)
                if stage==STAGES[1]:
                    loss=action_loss(estimated,noise,exact,variance,observed,probes)
                else:
                    loss=iq_error(complex_iq(model.unroll(incoming,coarse,estimated,noise)),clean)
            parts=loss[None]
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native objective')
        loss.backward()
        fit_gradient=float(torch.nn.utils.clip_grad_norm_(trained.parameters(),1.,error_if_nonfinite=True))
        if step==1 and not (0<fit_gradient<float('inf')):
            raise ValueError('Native stage has no finite gradient')
        if smoke and is_task and any(t.grad is not None for t in reader.parameters()):
            raise ValueError('Frozen E104 received gradients')
        optimizer.step()
        schedule.step()
        if step==1 or step%50==0:
            print('LORA_NATIVE_FIT',stage,step,*parts.detach().tolist(),flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    proof=dict(stage=stage,seed=seed,steps=steps,raw_length=NATIVE_LENGTH,protocol=contract['protocol'],
               labeled_exposures=steps*count,unlabeled_exposures=steps*count,
               F_reads=is_task,F_trainable=False,**dict.fromkeys(ISOLATION,0))
    if is_task:
        proof.update(initialization='same_source_w24_length_adapted',U_PL_teacher=joint.AUG_RUN,
                     U_PL_threshold=joint.PL_THRESHOLD,U_PL_accepted=int(accepted),
                     strong_exposures=int(mix.selected))
    if stage in REL_ARMS:
        proof.update(initialization=TASK_RUN,localized_reliability=stage==REL_ARMS[-1],
                     source_envelope_fraction=.5)
    if stage in WAVE_ARMS:
        proof.update(initialization=TASK_RUN,visible_waveform=stage==WAVE_ARMS[-1])
    if stage in GEOMETRY_ARMS:
        proof.update(initialization=TASK_RUN,relative_clean_geometry=stage==GEOMETRY_ARMS[-1],
                     clean_prototype_source_count=len(l_iq))
    summary=dict(**proof,status='PASS' if smoke else 'SOURCE_FROZEN',
                 gradient=fit_gradient,CE_to_G=supervised,pseudo_CE_to_G=pseudo_gradient,geometry_to_G=geometry_gradient,
                 worker_seconds=monotonic()-started,parameters=sum(x.numel() for x in trained.parameters()),
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0)
    if not smoke:
        with (folder/'model.pt').open('xb') as out:
            torch.save(dict(contract=contract,proof=proof,model=trained.state_dict()),out)
    write(folder/('smoke.json' if smoke else 'complete.json'),summary)


@torch.no_grad()
def load_readers(joint, rt, device):
    run=rt['run_id']
    root=Path(rt['run_root'])
    contract,f,g0,tracker=joint.boot(rt,device)
    if run==DURA_RUN:
        model=dura_model(root,contract,device).eval().requires_grad_(False)
    elif run==TASK_RUN:
        model=task_model(None,device)
        load_stage(root,TASK,contract,model.core,device)
        model.eval().requires_grad_(False)
    else:
        projector=build_projector(tracker,device) if run==REL_RUN else None
        model={}
        for stage in RUNS[run]:
            restorer=task_model(None,device,projector=projector,localized=stage==REL_ARMS[-1])
            load_stage(root,stage,contract,restorer.core,device)
            model[stage]=restorer.eval().requires_grad_(False)
    return f,model,tracker,contract


def restored_raw(model, raw, mode, device):
    iq=tensor(raw.view(np.float32),device).view(torch.complex64)
    if mode in (*REL_ARMS,*WAVE_ARMS,*GEOMETRY_ARMS):
        out=model[mode](iq)
    else:
        out=model(iq) if mode==TASK else model(iq,stage=mode)
    if out.shape!=iq.shape or not out.is_complex():
        raise ValueError('Expected one complete physical native IQ packet per input')
    return np.asarray(out.detach().cpu().view(torch.float32).tolist(),dtype=np.float32).view(np.complex64)


def render(joint, f, model, mode, raw, device):
    return joint.classify_aperture(f,joint.received_aperture(restored_raw(model,raw,mode,device),device))
