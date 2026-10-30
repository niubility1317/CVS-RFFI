"""Source-only public L-STF phase-reference feasibility; no G or TX truth."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.cvs_rff_physics import known_excitation
from experiments.cvs_reference_identity.model import ExcitationResponse
from experiments.pamf_dura_direct_train import runner as training, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices, tensor, numpy_values
from experiments.pamf_restoration.models import complex_iq, pair
from .phase_sensitivity import metrics

RUN='20261030-diagnostic-DURA-source-phase-anchor-s2026092701-r01'
RX=(1,3,4,6,8)
SESSIONS={f'{r}:{d}' for r in RX for d in (1,2,3)}
RULE=('Calibration excludes fixed source512. ALL62488: valid>=.999,resultant>=.99; '
      'EACH15RX/day:valid>=.999,resultant>=.95,mean direction within5degrees global. '
      'Disjoint512 F preservation:overallchoice>=.995,EACHRX>=.99,meanFcos<=.001,meanKL<=.001. '
      'Any undefined phase or failure: no global-anchor deployment/training from this study;no RX-specific fallback.')


@torch.no_grad()
def public_gain(x, correlator):
    w=complex_iq(x)[:,80:160]
    cross=(w[:,20:]*w[:,:-20].conj()).sum(-1)
    omega=torch.atan2(cross.imag,cross.real+correlator.epsilon**2)/20
    n=torch.arange(80,device=x.device,dtype=x.dtype)
    unit_phase=torch.polar(torch.ones_like(omega[:,None]*n),-omega[:,None]*n)
    mean=(w*unit_phase).reshape(-1,4,20).mean(1)
    unit=mean/(mean.abs().square().mean(1)+correlator.epsilon**2).sqrt()[:,None]
    bank=torch.complex(correlator.bank_real.to(x.dtype),correlator.bank_imag.to(x.dtype))
    corr=unit@bank.conj().T/20
    chosen=corr.abs().square().argmax(1)
    return corr.gather(1,chosen[:,None]).squeeze(1)


def circular_summary(gain):
    gain=np.asarray(gain,np.complex128)
    valid=np.abs(gain)>1e-6
    if not np.isfinite(gain).all() or gain.ndim!=1 or not len(gain):raise ValueError('Finite nonempty source gain vector required')
    mean=(gain[valid]/np.abs(gain[valid])).mean() if valid.any() else None
    resultant=float(abs(mean)) if mean is not None else 0.
    angle=float(np.angle(mean)) if resultant>1e-12 else None
    errors=np.abs(np.angle(gain[valid]*np.exp(-1j*angle))) if angle is not None else None
    return dict(count=len(gain),valid_count=int(valid.sum()),valid_fraction=float(valid.mean()),resultant=resultant,
        mean_degrees=math.degrees(angle) if angle is not None else None,
        p95_absolute_error_degrees=float(np.rad2deg(np.quantile(errors,.95))) if errors is not None else None)


def anchor_decision(calibration, fidelity):
    if set(calibration['sessions'])!=SESSIONS:raise ValueError('All15 original sourceRX/day groups required')
    total=calibration['all']
    phase_ok=(total['mean_degrees'] is not None and total['valid_fraction']>=.999 and total['resultant']>=.99
        and all(g['valid_fraction']>=.999 and g['resultant']>=.95
                and g['mean_error_to_global_degrees'] is not None and g['mean_error_to_global_degrees']<=5
                for g in calibration['sessions'].values()))
    f_ok=False
    if fidelity is not None:
        if (set(fidelity['receivers'])!=set(map(str,RX)) or fidelity['all']['count']!=512
                or sum(g['count'] for g in fidelity['receivers'].values())!=512):
            raise ValueError('All512/allfive sourceRX F comparison required')
        total_f=fidelity['all']
        f_ok=(total_f['teacher_correct']/512>=.995
            and all(g['teacher_correct']/g['count']>=.99 for g in fidelity['receivers'].values())
            and total_f['feature_cosine_error']<=.001 and total_f['posterior_KL']<=.001)
    return dict(phase_calibration_passed=bool(phase_ok),heldout_F_preserved=bool(f_ok),
                source_anchor_study_eligible=bool(phase_ok and f_ok))


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    p=training.paths(args.root);clean,roles=training.source_training(p)
    ids=f_guided.source_ids(p,roles);probe=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    if rx.shape!=(63000,) or day.shape!=(63000,) or set(rx.tolist())!=set(RX) or set(day.tolist())!={1,2,3}:
        raise ValueError('Original source RX/day metadata required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    frozen={k:v.clone() for k,v in reader.state_dict().items()}
    correlator=ExcitationResponse('reference_response').eval()
    if list(correlator.parameters()):raise ValueError('Public correlator must have no learned parameters')
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference()))
    smoke=reader(tensor(clean[probe[:2]],'cpu'))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual source2Ls F checkpoint smoke failed')
    print('{"event":"REAL_SOURCE_F_CHECKPOINT_SMOKE_PASS","Ls":2,"G_reads":0,"target_reads":0}',flush=True)
    started=time.perf_counter();gains=[]
    for offset in range(0,63000,256):
        gain=public_gain(tensor(clean[offset:offset+256],'cpu'),correlator)
        if not torch.isfinite(gain).all():raise FloatingPointError('Nonfinite public source correlation')
        gains.append(numpy_values(torch.stack((gain.real,gain.imag),1)))
        if (offset//256+1)%32==0:print(f'{{"event":"SOURCE_CORRELATION_PROGRESS","count":{min(offset+256,63000)}}}',flush=True)
    gain_pair=np.concatenate(gains);g=gain_pair[:,0].astype(np.float64)+1j*gain_pair[:,1].astype(np.float64)
    calibration=np.ones(63000,dtype=bool);calibration[probe]=False
    assert calibration.sum()==62488
    all_stats=circular_summary(g[calibration]);mu=all_stats['mean_degrees']
    session_stats={}
    for r in RX:
        for d in (1,2,3):
            stats=circular_summary(g[calibration&(rx==r)&(day==d)])
            stats['mean_error_to_global_degrees']=(abs(math.degrees(math.atan2(
                math.sin(math.radians(stats['mean_degrees']-mu)),math.cos(math.radians(stats['mean_degrees']-mu)))))
                if mu is not None and stats['mean_degrees'] is not None else None)
            session_stats[f'{r}:{d}']=stats
    phase=dict(all=all_stats,sessions=session_stats)
    fidelity=None;fv=np.empty((0,512,160),np.float32);lz=np.empty((0,512,6),np.float32);fm={}
    valid_probe=bool((np.abs(g[probe])>1e-6).all())
    power_error=0.
    if mu is not None and valid_probe:
        features=[[],[]];logits=[[],[]]
        rotation=np.exp(1j*(math.radians(mu)-np.angle(g[probe])))
        for offset in range(0,512,32):
            x=tensor(clean[probe[offset:offset+32]],'cpu')
            q=rotation[offset:offset+32]
            qp=tensor(np.stack((q.real,q.imag),1),'cpu');unit=torch.complex(qp[:,0],qp[:,1])
            corrected=pair(complex_iq(x)*unit[:,None])
            power_error=max(power_error,float(((corrected.square().sum(1)-x.square().sum(1)).abs().mean(1)
                        /x.square().sum(1).mean(1).clamp_min(1e-8)).max()))
            for i,value in enumerate((x,corrected)):
                f=reader.features(value);z=reader.classify_features(f)
                if f.shape!=(len(x),160) or z.shape!=(len(x),6) or not torch.isfinite(f).all() or not torch.isfinite(z).all():
                    raise FloatingPointError('Nonfinite source F preservation probe')
                features[i].append(f.clone());logits[i].append(z.clone())
        ft=[torch.cat(v) for v in features];zt=[torch.cat(v) for v in logits]
        fm={k:numpy_values(v) for k,v in metrics(ft[1],zt[1],ft[0],zt[0]).items()}
        fv=numpy_values(torch.stack(ft));lz=numpy_values(torch.stack(zt))
        def group(mask):
            return dict(count=int(mask.sum()),teacher_correct=int(fm['teacher_agreement'][mask].sum()),
                feature_cosine_error=float(fm['feature_cosine_error'][mask].mean(dtype=np.float64)),
                posterior_KL=float(fm['posterior_KL'][mask].mean(dtype=np.float64)))
        fidelity=dict(all=group(np.ones(512,dtype=bool)),receivers={str(r):group(rx[probe]==r) for r in RX})
    if power_error>2e-6:raise ValueError('Phase projection changed source power')
    if any(not torch.equal(v,frozen[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Frozen F state or gradients changed')
    np.savez(args.output/'values.npz',gain_pair=gain_pair,rx=rx,day=day,roles=roles,calibration_mask=calibration,
        probe_positions=probe,probe_physical_ids=ids[probe],F_features=fv,F_logits=lz,**fm)
    decision=anchor_decision(phase,fidelity)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',calibration=phase,
        heldout_phase=circular_summary(g[probe]),heldout_F=fidelity,decision=decision,source_rule=RULE,
        source_records=63000,calibration_records=62488,probe_records=512,source_public_correlator_inputs=63000,
        F_feature_analysis_inputs=1024 if fidelity is not None else 0,F_smoke_inputs=2,F_teacher=f_guided.teacher_reference(),
        public_bank_reference=known_excitation.REFERENCE,public_bank_learned_parameters=0,F_unchanged=True,
        optimizer_updates=0,G_checkpoint_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,power_error_max=power_error,
        seconds=time.perf_counter()-started,
        scope='Source phase-reference feasibility and originalF teacher-choice preservation;not TX/hardware calibration,target phase prior or restoration performance'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',F_unchanged=True,
        optimizer_updates=0,target_reads=0,goal_complete=False,**decision))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
