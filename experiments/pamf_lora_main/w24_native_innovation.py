"""Matched access to native out-of-aperture residual IQ in a scratch LoRa restorer."""

import argparse
from pathlib import Path, PurePosixPath
from time import monotonic

import numpy as np
import torch
from torch.nn.functional import avg_pool1d, conv1d, pad

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.w24_edit_budget_audit import score
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_packet_reconstruction import (
    CENTER, CHIRPS, PARENT, SEED, PacketReconstructor, classify_aperture, predict_panel,
    real_iq, received_aperture, receiver_frame, risks, source_tools,
)


RUN = '20261126-phase1-lora-w24-native-innovation-s2026112991-r01'
STEPS, FIT_PACKETS, HOLD_PACKETS, BATCH = 12000, 16000, 160, 8
MILESTONES = (2000, 6000, STEPS)
MODES = ('direct', 'aperture_half', 'aperture_full', 'native_half', 'native_full', 'outer_half')
CONTRASTS = (('native_half','aperture_half'), ('native_full','aperture_full'),
             ('native_full','native_half'), ('native_half','outer_half'), ('native_half','direct'))


def packet_tensor(raw, device):
    iq = tensor(np.stack((raw.real,raw.imag),axis=-1),device)
    return torch.view_as_complex(iq)


def high_pass_taps(device, dtype):
    offsets = torch.arange(-16,17,device=device,dtype=dtype)
    low = .384 * torch.sinc(.384*offsets) * torch.hamming_window(
        offsets.numel(),periodic=False,device=device,dtype=dtype)
    return ((offsets==0).to(dtype) - low/low.sum())[None,None,:]


def native_residue(raw, taps):
    values = torch.stack((raw.real,raw.imag),dim=1)
    high = conv1d(pad(values,(16,16),mode='reflect'),taps.expand(2,-1,-1),groups=2)
    return torch.complex(high[:,0],high[:,1])


def local_mean(x):
    shape = x.shape
    return avg_pool1d(x.reshape(-1,1,shape[-1]),5,stride=1,padding=2,
                      count_include_pad=False).reshape(shape)


def side_fields(observed, frame, residue, index):
    local = observed - torch.complex(local_mean(observed.real),local_mean(observed.imag))
    native_window = residue[...,index] / frame[...,None]
    native_mean = native_window.mean(-1)
    local_energy = local_mean(local.abs().square())
    native_energy = local_mean(native_window.abs().square().mean(-1))

    def make(mean, energy):
        log_energy = energy.clamp_min(1e-8).log().clamp(-16.,4.) / 8.
        return torch.stack((mean.real.clamp(-3.,3.),mean.imag.clamp(-3.,3.),log_energy),dim=-2)

    return make(local,local_energy),make(native_mean,native_energy)


def observation(tracker, raw, taps):
    aperture = tracker.aperture(raw)
    center, guard = aperture[...,:CENTER],aperture[...,CENTER:]
    inputs, frame, rms = receiver_frame(tracker,center,guard)
    # WideTracker.native index support spans offsets -26...27. Select +/-1 and +2.
    index = tracker.indices[...,25:29]
    fields = side_fields(inputs[0],frame,native_residue(raw,taps),index)
    packet_rms = center.abs().square().mean((-1,-2),keepdim=True).clamp_min(1e-12).sqrt()
    return inputs, frame, rms, fields, packet_rms


def training_pair(tracker, raw, generic, rng, taps):
    scale = raw.abs().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    clean = raw/scale
    disturbed = generic(clean.unflatten(-1,(CHIRPS,1024)),rng).flatten(-2)
    received = torch.cat((disturbed[:-1],clean[-1:]))
    inputs, frame,rms,fields,_ = observation(tracker,received,taps)
    target = tracker.aperture(clean)
    return inputs,fields,target[...,:CENTER]/frame,target[...,CENTER:]/rms


def metrics(output, target, edges, inputs):
    base = risks(output,target,edges)
    center, _, _, guard = output
    identity = ((center[-1]-inputs[0][-1]).abs().square().mean()
                + (guard[-1]-inputs[1][-1]).abs().square().mean())
    # Keep training tied to both predeclared actual readout strengths.
    half = .5*(center+inputs[0])
    half_risk = risks((half,output[1],output[2],guard),target,edges)[1]
    return torch.cat((base,torch.stack((half_risk,identity))))


@torch.no_grad()
def heldout(models, tracker, generic, taps, native_iq, rows, device):
    rng = torch.Generator(device=device).manual_seed(SEED+1)
    values = []
    for first in range(0,len(rows),BATCH):
        raw = packet_tensor(native_iq[rows[first:first+BATCH]],device)
        inputs,field,clean,edges = training_pair(tracker,raw,generic,rng,taps)
        values.append(torch.stack([metrics(m(*inputs,x),clean,edges,inputs)
                                   for m,x in zip(models,field,strict=True)]))
    return torch.stack(values).mean(0).tolist()


def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    tracker,generic = source_tools(rt,device)
    contract = source_contract(source)
    raw = native(source,'U_s',contract['counts']['U_s'])
    selection = np.random.default_rng(SEED)
    rows = selection.choice(len(raw),FIT_PACKETS+HOLD_PACKETS,replace=False)
    taps = high_pass_taps(device,torch.float32)
    rng = torch.Generator(device=device).manual_seed(SEED)
    torch.manual_seed(SEED)
    models = [PacketReconstructor(True,receive_fields=3).to(device) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [torch.optim.AdamW(model.parameters(),lr=.0006) for model in models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=.00008)
                  for optimizer in optimizers]
    with torch.no_grad():
        inputs,fields,target,guard = training_pair(tracker,packet_tensor(raw[rows[:2]],device),generic,rng,taps)
        check = torch.stack([metrics(model(*inputs,x),target,guard,inputs)
                             for model,x in zip(models,fields,strict=True)])
        if fields[1].shape != (2,CHIRPS,3,CENTER) or not torch.isfinite(check).all():
            raise ValueError('Native receive IQ source smoke failed')
    write(root/'fit-smoke.json',dict(status='PASS',F_reads=0,target_reads=0,source_u_packets=2))
    print('W24_NATIVE_INNOVATION_FIT_SMOKE PASS',flush=True)
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    evaluations, block = {}, []
    for step in range(1,STEPS+1):
        batch = selection.choice(rows[:FIT_PACKETS],BATCH,replace=False)
        with torch.no_grad():
            inputs,fields,target,guard = training_pair(tracker,packet_tensor(raw[batch],device),generic,rng,taps)
        risks_by_row = []
        for model,optimizer,scheduler,feature in zip(models,optimizers,schedulers,fields,strict=True):
            value = metrics(model(*inputs,feature),target,guard,inputs)
            loss = value @ value.new_tensor((1., .35, .15, .15, .15, .12, .30, .55))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            scheduler.step()
            risks_by_row.append(value.detach())
        block.append(torch.stack(risks_by_row))
        if step % 200 == 0:
            average = torch.stack(block).mean(0)
            if not torch.isfinite(average).all():
                raise ValueError('Nonfinite source loss')
            print('W24_NATIVE_INNOVATION_STEP',step,average.tolist(),flush=True)
            block.clear()
        if step in MILESTONES:
            result = heldout(models,tracker,generic,taps,raw,rows[FIT_PACKETS:],device)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite source waveform holdout')
            evaluations[str(step)] = dict(seconds=monotonic()-start,metrics=result)
            write(root/f'holdout-{step}.json',evaluations[str(step)])
            print('W24_NATIVE_INNOVATION_HOLDOUT',step,result,flush=True)
    torch.save([model.cpu().state_dict() for model in models],root/'native-innovation.pt')
    write(root/'fit-complete.json',dict(status='SOURCE_FROZEN',initialization='scratch',steps=STEPS,
          arms=['aperture','native'],seed=SEED,F_reads=0,target_reads=0,source_truth_reads=0,source_V_reads=0,
          fit_pool=FIT_PACKETS,holdout_packets=HOLD_PACKETS,evaluations=evaluations,
          parameters=sum(p.numel() for p in models[0].parameters()),
          peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2))


@torch.no_grad()
def render(f, tracker, models, raw, device):
    original = received_aperture(raw,device)
    measured = packet_tensor(raw,device)
    inputs,frame,_,fields,rms = observation(tracker,measured,high_pass_taps(device,measured.real.dtype))
    decisions = [classify_aperture(f,original)]
    for model, field in zip(models,fields,strict=True):
        estimate = model(*inputs,field)[0]
        delta = (estimate-inputs[0])*frame/rms
        for share in (.5,1.):
            out = original.clone()
            out[...,:CENTER] += real_iq(share*delta)
            decisions.append(classify_aperture(f,out))
    return np.stack(decisions)


def predict(rt):
    root,device = Path(rt['run_root']),'cuda:0'
    frozen = read(root/'fit-complete.json')
    if (frozen['status'],frozen['steps'],frozen['arms'],frozen['initialization']) != (
            'SOURCE_FROZEN',STEPS,['aperture','native'],'scratch') or any(
            frozen[k] for k in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
        raise ValueError('Missing legally frozen source model pair')
    models = [PacketReconstructor(True,receive_fields=3).to(device).eval() for _ in range(2)]
    weights = torch.load(root/'native-innovation.pt',map_location=device,weights_only=True)
    for model,state in zip(models,weights,strict=True):
        model.load_state_dict(state)
    predict_panel(rt,models,MODES[:-1],render_batch=render)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    phases = ('fit-source','predict-source','score-source')
    for phase in ('dispatch',*phases):
        sub.add_parser(phase).add_argument('runtime',type=Path)
    pub = sub.add_parser('publish')
    for key in ('context','repo','output','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime,entry=__file__,phases=phases,gpu_phases=phases[:2])
    elif args.action == 'fit-source':
        fit(read(args.runtime))
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime,modes=MODES,contrasts=CONTRASTS)
    else:
        ctx=read(args.context)
        rt={k:ctx[k] for k in ('project','python','frozen_sources')}
        project=PurePosixPath(rt['project'])
        runs,releases=project/'runs',project/'releases'
        rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'),run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3,6,0],cpu_workers=8)
        rt['external_releases']=[rt[k] for k in ('source_root','classifier_root','w24_release_root',
                                                'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
