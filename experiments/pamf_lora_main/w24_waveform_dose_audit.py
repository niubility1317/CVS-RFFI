"""Source-only audit of dose selection with exact noiseless receive components."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

if not __package__:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import FULL, full_components, score
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, panel
from experiments.pamf_lora_main.w24_formal import wide_delta
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, BATCH, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as PARENT, load_candidate as load_parent
from experiments.pamf_lora_main.w24_third_prior import UNET_FIDELITY_ARMS, UNET_FIDELITY_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import load_restorer, received_aperture, training_runtime


RUN = '20261126-diagnostic-lora-w24-waveform-dose-source-s2026112559-r01'
SPECTRAL_RUN = '20261126-diagnostic-lora-w24-spectral-simplex-source-s2026112559-r01'
BASE = ('direct', 'outer_half', 'standard_half', 'standard_full', 'complex_half', 'complex_full')
MODES = (*BASE, 'minimum_mse', 'minimum_balanced', 'continuous_mse')
CONTRASTS = tuple((x,'outer_half') for x in MODES[2:]) + tuple(
    (x,'standard_half') for x in MODES[len(BASE):])
BANDS = ((0,0),(1,8),(9,32),(33,64),(65,192))
EXPERT_BANDS = ((1,8),(9,32),(33,80),(81,192))
SPECTRAL_MODES = (*BASE,'packet_band','chirp_band','chirp_band_discrete')
SPECTRAL_CONTRASTS = tuple((x,'outer_half') for x in SPECTRAL_MODES[len(BASE):]) + (
    ('chirp_band','packet_band'),('chirp_band','chirp_band_discrete'))


def mse(edits, target):
    """Batch-major candidate bank, with all branches in the same receive scale."""
    return (edits-target[:,None]).abs().square().mean((-1,-2))


def balanced(edits, target, signal, phase):
    fft = torch.fft.fft((edits-target[:,None])*phase[:,None].conj(),norm='ortho')
    signal = torch.fft.fft(signal*phase.conj(),norm='ortho')
    bins = torch.arange(CENTER,device=edits.device)
    bins = torch.minimum(bins,CENTER-bins)
    values = []
    for low,high in BANDS:
        mask = (bins>=low) & (bins<=high)
        risk = fft[...,mask].abs().square().mean((-1,-2))
        energy = signal[...,mask].abs().square().mean((-1,-2))
        values.append(torch.log1p(risk/(.003+energy[:,None])))
    return torch.stack(values,dim=-1).mean(-1)


def select(bank, risk):
    idx = risk.argmin(-1)
    return bank[torch.arange(len(bank),device=bank.device),idx],idx


def optimal_rays(directions, target):
    dot = (directions.conj()*target[:,None]).real.sum((-1,-2))
    squared = directions.abs().square().sum((-1,-2))
    strength = (dot/squared.clamp_min(1e-12)).clamp(0.,1.)
    return directions*strength[...,None,None], strength


def simplex_fit(a, b, truth):
    """Solve the two-variable real least squares problem on a closed simplex."""
    def dot(x,y):
        return (x.to(torch.complex128).conj()*y.to(torch.complex128)).real.sum(-1)

    aa,bb,ab=dot(a,a),dot(b,b),dot(a,b)
    ya,yb=dot(a,truth),dot(b,truth)
    zero=torch.zeros_like(aa)
    pa=(ya/aa.clamp_min(1e-20)).clamp(0.,1.)
    pb=(yb/bb.clamp_min(1e-20)).clamp(0.,1.)
    d=a-b
    edge=(dot(d,truth-b)/dot(d,d).clamp_min(1e-20)).clamp(0.,1.)
    det=aa*bb-ab.square()
    x=(bb*ya-ab*yb)/det.clamp_min(1e-20)
    y=(aa*yb-ab*ya)/det.clamp_min(1e-20)
    valid=(det>16*torch.finfo(det.dtype).eps*aa*bb)&(x>=0.)&(y>=0.)&(x+y<=1.)
    x=torch.where(valid,x,zero)
    y=torch.where(valid,y,zero)
    wa=torch.stack((pa,zero,edge,x),dim=-1).to(a.real.dtype)
    wb=torch.stack((zero,pb,1.-edge,y),dim=-1).to(b.real.dtype)
    edit=wa[...,None]*a[...,None,:]+wb[...,None]*b[...,None,:]
    risk=(edit-truth[...,None,:]).abs().square().sum(-1)
    return edit.gather(-2,risk.argmin(-1)[...,None,None].expand(*risk.shape[:-1],1,edit.shape[-1])).squeeze(-2)


def spectral_edits(a,b,truth):
    distance=torch.arange(CENTER,device=a.device)
    distance=torch.minimum(distance,CENTER-distance)
    packet,local,discrete=(torch.zeros_like(a) for _ in range(3))
    for low,high in EXPERT_BANDS:
        mask=(distance>=low)&(distance<=high)
        x,y,target=a[...,mask],b[...,mask],truth[...,mask]
        packet[...,mask]=simplex_fit(x.flatten(1),y.flatten(1),target.flatten(1)).reshape_as(x)
        local[...,mask]=simplex_fit(x,y,target)
        bank=torch.stack((torch.zeros_like(x),.5*x,x,.5*y,y),dim=-2)
        idx=(bank-target[...,None,:]).abs().square().mean(-1).argmin(-1)
        discrete[...,mask]=bank.gather(-2,idx[...,None,None].expand(*idx.shape,1,x.shape[-1])).squeeze(-2)
    return packet,local,discrete


def candidate_bank(tracker,models,received,signal,device):
    original = received_aperture(received,device)
    observed = complex_iq(original[...,:CENTER])
    ideal = complex_iq(tensor(model_iq(native_aperture(signal),
                                      center_rms(native_aperture(received))),device)[...,:CENTER])
    expected = ideal-observed
    directions = [complex_iq(wide_delta(model,tracker,original)) for model in models]
    outer, standard, rich = directions
    bank = torch.stack((torch.zeros_like(outer), .5*outer, .5*standard, standard, .5*rich, rich),dim=1)
    _, phase, _, _ = coordinates(tracker,original[...,:CENTER])
    return original,ideal,expected,directions,bank,phase


def frozen_labels(readout,original,bank,extra):
    labels=[classify_aperture(readout,original)]
    for delta in (*bank.unbind(1)[1:],*extra):
        edited=original.clone()
        edited[...,:CENTER]+=real_iq(delta)
        labels.append(classify_aperture(readout,edited))
    return np.stack(labels)


@torch.no_grad()
def evaluate(readout, tracker, models, received, signal, device):
    original,ideal,expected,directions,bank,phase=candidate_bank(tracker,models,received,signal,device)
    nearest, i_mse = select(bank,mse(bank,expected))
    colored, i_balanced = select(bank,balanced(bank,expected,ideal,phase))
    rays, weights = optimal_rays(torch.stack(directions,dim=1),expected)
    optimal, i_ray = select(rays,mse(rays,expected))
    extra=(nearest,colored,optimal)
    v = torch.arange(len(weights),device=weights.device)
    info = dict(choices=torch.stack((i_mse,i_balanced,i_ray),dim=1).tolist(),
                strength=weights[v,i_ray].tolist(),
                scores=torch.cat((mse(bank,expected),
                                  mse(torch.stack((nearest,colored,optimal),dim=1),expected)),dim=1).tolist())
    return frozen_labels(readout,original,bank,extra),info


@torch.no_grad()
def evaluate_spectral(readout,tracker,models,received,signal,device):
    original,_,expected,directions,bank,phase=candidate_bank(tracker,models,received,signal,device)
    spectrum=[torch.fft.fft(value*phase.conj(),norm='ortho') for value in (*directions[:2],expected)]
    extra=tuple(torch.fft.ifft(value,norm='ortho')*phase for value in spectral_edits(*spectrum))
    risk=mse(torch.cat((bank,torch.stack(extra,dim=1)),dim=1),expected)
    return frozen_labels(readout,original,bank,extra),dict(scores=risk.tolist())


def summarize(records):
    choices=np.concatenate([x['choices'] for x in records])
    strengths=np.concatenate([x['strength'] for x in records])
    values=np.concatenate([x['scores'] for x in records])
    common = {name: int(n) for name,n in zip(BASE,np.bincount(choices[:,0],minlength=len(BASE)),strict=True)}
    color = {name: int(n) for name,n in zip(BASE,np.bincount(choices[:,1],minlength=len(BASE)),strict=True)}
    rays = {name: int(n) for name,n in zip(('outer','standard','complex'),
                                         np.bincount(choices[:,2],minlength=3),strict=True)}
    return dict(mse_selected=common,balanced_selected=color,continuous_direction=rays,
                continuous_fraction_q10_q50_q90=np.quantile(strengths,[.1,.5,.9]).tolist(),
                center_mse_q10_q50_q90={name:q.tolist() for name,q in
                    zip(MODES,np.quantile(values,[.1,.5,.9],axis=0).T,strict=True)})


def summarize_spectral(records):
    values=np.concatenate([x['scores'] for x in records])
    return dict(center_mse_q10_q50_q90={name:q.tolist() for name,q in zip(
                SPECTRAL_MODES,np.quantile(values,[.1,.5,.9],axis=0).T,strict=True)})


def predict(rt):
    activate(rt)
    root, source, device = Path(rt['run_root']),Path(rt['source_root']),'cuda:0'
    contract = source_contract(source)
    _, tracker, _, _ = load_restorer(*training_runtime(rt),contract,device)
    parent,_=load_parent(root.parent/PARENT,contract,device)
    unet_root=root.parent/UNET_FIDELITY_RUN
    if read(unet_root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('U-Net source baseline is not complete')
    models=[parent,*(load_candidate(unet_root,contract,device,arm)[0] for arm in UNET_FIDELITY_ARMS)]
    readout=load_classifier(rt,contract,device)
    spectral=rt.get('spectral_simplex',False)
    modes=SPECTRAL_MODES if spectral else MODES
    run_case=evaluate_spectral if spectral else evaluate
    summarize_case=summarize_spectral if spectral else summarize
    u=native(source,'U_s',contract['counts']['U_s'])[0]
    for scene in ('clean',VIEWS[-1]):
        y,truth_wave=full_components(scene,(0,SOURCE[0],u),verify=True)
        d,meta=run_case(readout,tracker,models,y[None],truth_wave[None],device)
        if d.shape!=(len(modes),1) or not np.isfinite(meta['scores']).all():
            raise ValueError('Invalid real source selection smoke')
        if scene=='clean' and not np.all(d[len(BASE):]==d[0]):
            raise ValueError('Clean oracle does not preserve the observation')
    write(root/'smoke.json',dict(status='PASS',source_u_packets=1,selection_source_diagnostic_only=True,
          target_reads=0,source_truth_reads=0))
    print('W24_WAVEFORM_DOSE_SMOKE PASS',flush=True)
    ix,rx,ids=panel(source,contract)
    rows=native(source,'V',contract['counts']['V'])[ix]
    jobs=[(int(i),SOURCE[int(j)],p) for i,j,p in zip(ix,rx,rows,strict=True)]
    matrix,detail={},{}
    for view in VIEWS:
        stream=map(partial(full_components,view),jobs)
        guesses, records=[],[]
        for start in range(0,len(jobs),BATCH):
            chunk=[next(stream) for _ in range(min(BATCH,len(jobs)-start))]
            received,signal=(np.stack(t) for t in zip(*chunk,strict=True))
            rows,meta=run_case(readout,tracker,models,received,signal,device)
            if not np.isfinite(meta['scores']).all():
                raise ValueError('Nonfinite source oracle risk')
            guesses.append(rows)
            records.append(meta)
        matrix[view]=np.concatenate(guesses,axis=1)
        detail[view]=summarize_case(records)
        print('W24_WAVEFORM_DOSE_PREDICTED',view,len(jobs),flush=True)
    check_controls(unet_root/FULL,('direct','outer_parent_half','unet_standard_long_half',
                   'unet_standard_long_full','unet_complex_long_half','unet_complex_long_full'),
                   list(range(len(BASE))),ids,matrix)
    out=root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz',**matrix)
    write(out/'source-predictions.json',dict(status='PREDICTIONS_FROZEN',modes=list(modes),views=list(VIEWS),
          physical_ids=ids,target_reads=0,source_truth_reads=0,waveform=dict(route=FULL,
          genuine_known_noiseless_source_only=True,statistics=detail)))


def main():
    from experiments.pamf_lora_main.publish import release,release_commit

    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for phase in ('dispatch','predict-source','score-source'):
        sub.add_parser(phase).add_argument('runtime',type=Path)
    pub=sub.add_parser('publish')
    for field in ('repo','context','output','ssh-config'):
        pub.add_argument('--'+field,type=Path,required=True)
    pub.add_argument('--spectral-simplex',action='store_true')
    args=parser.parse_args()
    if args.action=='dispatch':
        dispatch(args.runtime,entry=__file__)
    elif args.action=='predict-source':
        predict(read(args.runtime))
    elif args.action=='score-source':
        if read(args.runtime).get('spectral_simplex',False):
            score(args.runtime,modes=SPECTRAL_MODES,contrasts=SPECTRAL_CONTRASTS)
        else:
            score(args.runtime,modes=MODES,contrasts=CONTRASTS)
    else:
        ctx=read(args.context)
        rt={k:ctx[k] for k in ('project','python','frozen_sources')}
        project=PurePosixPath(rt['project'])
        runs,releases=project/'runs',project/'releases'
        run=SPECTRAL_RUN if args.spectral_simplex else RUN
        rt.update(run_id=run,runner=__spec__.name,code_commit=release_commit(args.repo),
                  spectral_simplex=args.spectral_simplex,
                  release_root=str(releases/f'{run}-release'),run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),allowed_gpus=[3,6,0])
        rt['external_releases']=[rt[k] for k in ('source_root','classifier_root','w24_release_root',
                                                'training_release_root')]
        rt['external_releases'] += [str(runs/r) for r in (PARENT,UNET_FIDELITY_RUN)]
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
