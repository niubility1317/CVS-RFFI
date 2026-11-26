"""Fit an observation-conditioned rectified flow on lawful LoRa source pairs."""

import argparse
from pathlib import Path, PurePosixPath
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.w24_edit_budget_audit import score
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_packet_reconstruction import (
    CENTER, CHIRPS, PARENT, SEED, TOKENS, WIDTH, Mixing, classify_aperture, complex_channels,
    complex_iq, predict_panel, real_iq, received_aperture, receiver_frame, source_tools, training_pair,
)


RUN = '20261126-phase1-lora-w24-packet-flow-s2026112991-r01'
UPDATES, FIT_COUNT, HOLD_COUNT, SAMPLES = 16000, 16000, 160, 8
SHOW_STEPS = (2000, 8000, UPDATES)
MODES = ('direct', 'one_half', 'one_full', 'four_half', 'four_full', 'outer_half')
CONTRASTS = (('four_half','one_half'), ('four_full','one_full'), ('four_full','four_half'),
             ('four_half','outer_half'), ('four_half','direct'))


def noise_level(observed):
    spectrum = torch.fft.fft(observed, norm='ortho')
    high = spectrum[..., CENTER//4:3*CENTER//4]
    return (high.abs().square().median(-1).values / .69314718056).clamp_min(1e-9).sqrt()


class PacketVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow = nn.Sequential(nn.Conv1d(7, 48, 9, padding=4), nn.SiLU(),
                                     nn.Conv1d(48, 48, 5, padding=2), nn.SiLU())
        self.down = nn.Conv1d(48, WIDTH, 16, stride=16)
        self.position = nn.Parameter(.02*torch.randn(1, CHIRPS, TOKENS, WIDTH))
        self.mixing = nn.ModuleList(Mixing() for _ in range(2))
        self.temporal = nn.Sequential(nn.Conv1d(WIDTH+48, 64, 7, padding=3), nn.SiLU(),
                                       nn.Conv1d(64, 32, 5, padding=2), nn.SiLU(),
                                       nn.Conv1d(32, 2, 3, padding=1))
        self.spectral = nn.Sequential(nn.Conv1d(WIDTH+4, 64, 7, padding=3), nn.SiLU(),
                                       nn.Conv1d(64, 32, 5, padding=2), nn.SiLU(),
                                       nn.Conv1d(32, 2, 3, padding=1))
        self.blend = nn.Linear(WIDTH, 1)
        for head in (self.temporal[-1], self.spectral[-1]):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    def forward(self, state, observed, quality, noise, time):
        batch = len(state)
        y, x = (value.flatten(0, 1) for value in (observed, state))
        condition = torch.stack((quality.flatten(), noise.flatten().log().clamp(-10., 2.)/5.,
                                 time.repeat_interleave(CHIRPS)), dim=-1)
        features = torch.cat((complex_channels(y), complex_channels(x),
                              condition[..., None].expand(-1, -1, CENTER)), dim=1)
        local = self.shallow(features)
        tokens = self.down(local).transpose(1, 2).reshape(batch, CHIRPS, TOKENS, WIDTH)
        tokens = (tokens+self.position).flatten(1, 2)
        for block in self.mixing:
            tokens = block(tokens, None)
        tokens = tokens.reshape(batch*CHIRPS, TOKENS, WIDTH)
        summary = tokens.mean(1)
        time_features = interpolate(tokens.transpose(1, 2), size=CENTER, mode='linear', align_corners=False)
        a, c = self.temporal(torch.cat((local, time_features), dim=1)).unbind(1)
        time_edit = torch.complex(a, c)
        joint_spectra = tuple(complex_channels(torch.fft.fftshift(torch.fft.fft(v, norm='ortho'), dim=-1))
                              for v in (x,y))
        a, c = self.spectral(torch.cat((*joint_spectra, summary[..., None].expand(-1,-1,CENTER)), dim=1)).unbind(1)
        freq_edit = torch.fft.ifft(torch.fft.ifftshift(torch.complex(a,c), dim=-1), norm='ortho')
        mix = self.blend(summary).sigmoid()
        combined = mix*time_edit + (1.-mix)*freq_edit
        return tuple(value.reshape_as(state) for value in (combined,time_edit,freq_edit))


def latent_example(inputs, ground, rng):
    observed, _, quality = inputs
    time = torch.rand(len(observed), device=observed.device, generator=rng)
    time = torch.where(time < .25, 0., (time-.25)/.75)
    velocity = ground-observed
    state = observed + time[:, None, None]*velocity
    return (state,observed,quality,noise_level(observed),time), velocity


def risks(outputs, target):
    final, time, frequency = outputs
    delta = final-target
    time_err = delta.abs().square().mean((-1,-2))
    bins = torch.arange(CENTER, device=delta.device)
    bins = torch.minimum(bins, CENTER-bins)
    weights = torch.where((bins >= 1) & (bins <= 32), 3., 1.)
    spec = torch.fft.fft(delta,norm='ortho')
    complex_err = (weights*spec.abs().square()).mean((-1,-2))
    difference = (delta[:,1:]-delta[:,:-1]).abs().square().mean((-1,-2))
    branches = ((time-target).abs().square()+(frequency-target).abs().square()).mean((-1,-2))
    return torch.stack((time_err, complex_err, difference, branches),dim=-1)


def integrate(model, observed, quality, steps):
    """One or four fixed Euler evaluations in the same observation frame."""
    current = observed
    noise = noise_level(observed)
    for step in range(steps):
        t = observed.real.new_full((len(observed),),step/steps)
        rate, *_ = model(current, observed, quality, noise, t)
        current = current + rate/steps
    return current


@torch.no_grad()
def heldout(model, tracker, generic, iq, rows, device):
    rng = torch.Generator(device=device).manual_seed(SEED+1)
    metrics = []
    for start in range(0,len(rows),SAMPLES):
        inputs, target, _ = training_pair(iq[rows[start:start+SAMPLES]],tracker,generic,rng,device)
        observed, _, quality = inputs
        recon = (observed, integrate(model,observed,quality,1), integrate(model,observed,quality,4))
        residual = [(x-target).abs().square().mean((-1,-2)) for x in recon]
        metrics.append(torch.stack([torch.stack((v[:-1].mean(),v[-1])) for v in residual]))
    return torch.stack(metrics).mean(0).tolist()


def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    tracker, generic = source_tools(rt,device)
    contract = source_contract(source)
    iq = native(source,'U_s',contract['counts']['U_s'])
    selection = np.random.default_rng(SEED)
    selected = selection.choice(len(iq),FIT_COUNT+HOLD_COUNT,replace=False)
    rng = torch.Generator(device=device).manual_seed(SEED)
    torch.manual_seed(SEED)
    model = PacketVelocity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0006)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,UPDATES,eta_min=.00008)
    with torch.no_grad():
        inputs, clean, _ = training_pair(iq[selected[:2]],tracker,generic,rng,device)
        point, target = latent_example(inputs,clean,rng)
        if not torch.isfinite(risks(model(*point),target)).all():
            raise ValueError('Invalid real source flow smoke')
    write(root/'fit-smoke.json',dict(status='PASS',F_reads=0,target_reads=0,source_u_packets=2))
    print('W24_PACKET_FLOW_FIT_SMOKE PASS',flush=True)
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    snapshots, block = {}, []
    for step in range(1,UPDATES+1):
        rows = selection.choice(selected[:FIT_COUNT], SAMPLES, replace=False)
        with torch.no_grad():
            inputs, ground, _ = training_pair(iq[rows],tracker,generic,rng,device)
            point, target = latent_example(inputs,ground,rng)
        values = risks(model(*point),target)
        per_packet = values @ values.new_tensor((1., .45, .12, .17))
        loss = (per_packet.sum()+per_packet[-1])/(len(per_packet)+1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        schedule.step()
        block.append(values.detach().mean(0))
        if step % 200 == 0:
            v = torch.stack(block).mean(0)
            if not torch.isfinite(v).all():
                raise ValueError('Nonfinite flow risk')
            print('W24_PACKET_FLOW_STEP',step,v.tolist(),flush=True)
            block.clear()
        if step in SHOW_STEPS:
            valid = heldout(model,tracker,generic,iq,selected[FIT_COUNT:],device)
            if not np.isfinite(valid).all():
                raise ValueError('Nonfinite fixed holdout')
            snapshots[str(step)] = dict(elapsed_s=monotonic()-start,errors=valid)
            write(root/f'holdout-{step}.json',snapshots[str(step)])
            print('W24_PACKET_FLOW_HOLDOUT',step,valid,flush=True)
    torch.save(model.cpu().state_dict(),root/'packet-flow.pt')
    write(root/'fit-complete.json',dict(status='SOURCE_FROZEN',initialization='scratch',steps=UPDATES,
          seed=SEED,F_reads=0,target_reads=0,source_truth_reads=0,source_V_reads=0,
          parameters=sum(p.numel() for p in model.parameters()),physical_fit_pool=FIT_COUNT,heldout_count=HOLD_COUNT,
          snapshots=snapshots,peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2))


@torch.no_grad()
def render(f, tracker, solvers, raw, device):
    original = received_aperture(raw,device)
    center, guard = (complex_iq(x) for x in (original[...,:CENTER],original[...,CENTER:]))
    (y,_,quality), frame,_ = receiver_frame(tracker,center,guard)
    labels = [classify_aperture(f,original)]
    for solve in solvers:
        update = (solve(y,quality)-y)*frame
        for fraction in (.5,1.):
            restored = original.clone()
            restored[...,:CENTER] += real_iq(fraction*update)
            labels.append(classify_aperture(f,restored))
    return np.stack(labels)


def predict(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    fit = read(root/'fit-complete.json')
    if (fit['status'],fit['initialization'],fit['steps']) != ('SOURCE_FROZEN','scratch',UPDATES) or any(
            fit[k] for k in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
        raise ValueError('Source-only flow weights must be frozen before W24')
    model = PacketVelocity().to(device).eval()
    model.load_state_dict(torch.load(root/'packet-flow.pt',map_location=device,weights_only=True))
    solvers = [lambda y,q,step=s: integrate(model,y,q,step) for s in (1,4)]
    predict_panel(rt,solvers,MODES[:-1],render_batch=render)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action',required=True)
    phases = ('fit-source','predict-source','score-source')
    for phase in ('dispatch',*phases):
        subs.add_parser(phase).add_argument('runtime',type=Path)
    pub = subs.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+field,type=Path,required=True)
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
        ctx = read(args.context)
        rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs',project/'releases'
        rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'),run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3,6,0],cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root','classifier_root','w24_release_root','training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo,args.output,rt,args.ssh_config)


if __name__ == '__main__':
    main()
