"""Condition scratch packet recovery on a source-supervised received noise estimate."""

import argparse
from functools import partial
from math import log
from pathlib import Path, PurePosixPath

import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

if not __package__:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.w24_edit_budget_audit import score
from experiments.pamf_lora_main.w24_native_innovation import (
    AUG_RUN, DATA_RUN, PARENT, STEPS, TRAIN_RUN, W24_RUN, PacketReconstructor, dispatch,
    fit, metrics, observation, predict_panel, render, training_pair,
)


RUN = '20261126-phase1-lora-w24-packet-noise-calibrated-s2026112991-r01'
RECOVERY = '20261126-phase1-lora-w24-packet-noise-calibrated-s2026112991-r03'
FIT_COMMIT = '6b9ab67d34b039a179cb0a1ea435668543d343d9'
ARMS = ('aperture_power','native_power')
MODES = ('direct','aperture_half','aperture_full','native_half','native_full','outer_half')
CONTRASTS = (('native_half','aperture_half'), ('native_full','aperture_full'),
             ('native_half','outer_half'), ('native_half','direct'), ('native_full','native_half'))
WEIGHTS = (1., .35, .15, .15, .15, .12, .30, .55, .10)


def packet_log_power(fields):
    log_local = 8*fields[:,:,2,:].flatten(1)
    return torch.logsumexp(log_local,dim=-1) - log(log_local.shape[-1])


def measured_fields(aperture, native):
    # The local complex residual and local log power are held fixed.
    return tuple(torch.cat((aperture,(packet_log_power(f)/8.)[:,None,None,None].expand(
        -1,aperture.shape[1],1,aperture.shape[-1])),dim=-2) for f in (aperture,native))


def packet_observation(tracker, raw, taps):
    inputs, frame, rms, fields, scale = observation(tracker,raw,taps)
    return inputs,frame,rms,measured_fields(*fields),scale


class CalibratedRestorer(PacketReconstructor):
    def __init__(self):
        super().__init__(True,receive_fields=4)
        self.noise_predictor = nn.Sequential(nn.Linear(2,24), nn.SiLU(), nn.Linear(24,24),
                                             nn.SiLU(), nn.Linear(24,1))
        nn.init.zeros_(self.noise_predictor[-1].weight)
        nn.init.zeros_(self.noise_predictor[-1].bias)

    def forward(self, center, edges, quality, fields):
        log_power = fields[:,0,3,0]
        packet_quality = quality.squeeze(-1).median(-1).values
        features = torch.stack((log_power,packet_quality),dim=-1)
        correction = self.noise_predictor(features).squeeze(-1)
        estimate = 8*log_power + correction
        noise_field = (estimate.clamp(-12.,3.)/8.)[:,None,None,None].expand(
            -1,fields.shape[1],1,fields.shape[-1])
        output = super().forward(center,edges,quality,torch.cat((fields[:,:,:3],noise_field),dim=-2))
        return (*output,estimate)


def calibrated_metrics(output, target, edges, inputs):
    waveform = metrics(output[:4],target,edges,inputs)
    observation_error = (target-inputs[0]).abs().square().mean((-1,-2))
    paired_log_power = (observation_error + 1e-5).log()
    k = smooth_l1_loss(output[-1],paired_log_power)
    return torch.cat((waveform,k[None]))


def frozen_fit(rt):
    root = Path(rt['run_root'])
    if root.name == RECOVERY:
        archive = Path(rt['project'])/'releases'/f'{RUN}-release'
        previous = read(archive/'runtime.json')
        fit_root = root.parent/RUN
        if (previous['code_commit'],previous['run_root'],previous['source_root'],
                previous['classifier_root'],previous['training_release_root']) != (
                FIT_COMMIT,str(fit_root),rt['source_root'],rt['classifier_root'],rt['training_release_root']):
            raise ValueError('Unavailable exact frozen source training')
        if read(fit_root/'fit-smoke.json')['status'] != 'PASS':
            raise ValueError('Original real source fit smoke did not pass')
        root = fit_root
    proof = read(root/'fit-complete.json')
    if (proof['status'],proof['steps'],proof['arms'],proof['initialization']) != (
            'SOURCE_FROZEN',STEPS,list(ARMS),'scratch') or any(
            proof[k] for k in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
        raise ValueError('Missing frozen source-only noise estimators and waveform models')
    return root


def predict(rt):
    root,device=frozen_fit(rt),'cuda:0'
    models=[CalibratedRestorer().to(device).eval() for _ in ARMS]
    states=torch.load(root/'native-innovation.pt',map_location=device,weights_only=True)
    for model,state in zip(models,states,strict=True):
        model.load_state_dict(state)
    predict_panel(rt,models,MODES[:-1],render_batch=partial(render,observe=packet_observation))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    phases=('fit-source','predict-source','score-source')
    for phase in ('dispatch',*phases):
        sub.add_parser(phase).add_argument('runtime',type=Path)
    pub=sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+field,type=Path,required=True)
    pub.add_argument('--reuse-frozen-source',action='store_true')
    args=parser.parse_args()
    if args.action=='dispatch':
        if Path(read(args.runtime)['run_root']).name == RECOVERY:
            dispatch(args.runtime,entry=__file__)
        else:
            dispatch(args.runtime,entry=__file__,phases=phases,gpu_phases=phases[:2])
    elif args.action=='fit-source':
        fit(read(args.runtime),make_model=CalibratedRestorer,
            pair=partial(training_pair,observe=packet_observation),
            measure=calibrated_metrics,loss_weights=WEIGHTS,field_count=4,arms=ARMS)
    elif args.action=='predict-source':
        predict(read(args.runtime))
    elif args.action=='score-source':
        score(args.runtime,modes=MODES,contrasts=CONTRASTS)
    else:
        ctx=read(args.context)
        rt={k:ctx[k] for k in ('project','python','frozen_sources')}
        project=PurePosixPath(rt['project'])
        runs,releases=project/'runs',project/'releases'
        run = RECOVERY if args.reuse_frozen_source else RUN
        rt.update(run_id=run,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'),run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3,6,0],cpu_workers=8)
        rt['external_releases']=[rt[k] for k in ('source_root','classifier_root','w24_release_root',
                                                'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        if args.reuse_frozen_source:
            rt['external_releases'] += [str(runs/RUN),str(releases/f'{RUN}-release')]
            rt['cpu_workers'] = 0
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
