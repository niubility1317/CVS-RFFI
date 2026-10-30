"""Original frozen-F phase sensitivity on fixed source L/U; no G or truth."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices, tensor, numpy_values
from experiments.pamf_restoration.models import complex_iq, pair

RUN='20261030-diagnostic-DURA-source-Fphase-s2026092701-r01'
RX=(1,3,4,6,8)
RULE=('All seven nonzero fixed angles need clean-F choice agreement >=99.5percent overall, '
      '>=99percent at EACH sourceRX, overall mean feature cosine error<=.001 and '
      'mean KL(clean||rotated)<=.001. Otherwise no global-phase-quotient training from this diagnostic. '
      'Passing only supports a separately preregistered study, not deployment/target invariance.')


def rotate(x, angle):
    if angle==0.:return x
    phase=x.new_tensor(angle)
    unit=torch.complex(phase.cos(),phase.sin())
    return pair(unit*complex_iq(x))


def metrics(features, logits, reference_features, reference_logits):
    if (features.shape!=reference_features.shape or features.ndim!=2
            or logits.shape!=reference_logits.shape or logits.shape!=(len(features),6)):
        raise ValueError('Paired source features and original six-class F logits required')
    logp=F.log_softmax(reference_logits,dim=1);logq=F.log_softmax(logits,dim=1)
    return dict(feature_cosine_error=(1-F.cosine_similarity(features,reference_features,dim=1,eps=1e-8)).clamp_min(0),
                posterior_KL=F.kl_div(logq,logp,log_target=True,reduction='none').sum(1).clamp_min(0),
                teacher_agreement=(logits.argmax(1)==reference_logits.argmax(1)).to(features.dtype))


def phase_decision(summary):
    checks=[]
    if set(summary)!=set(map(str,range(8))):raise ValueError('All eight fixed angles required')
    for a in range(8):
        groups=summary[str(a)]
        if (groups['all']['count']!=512 or not all(f'rx:{r}' in groups for r in RX)
                or sum(groups[f'rx:{r}']['count'] for r in RX)!=512):
            raise ValueError('Complete fixed512/allfive sourceRX groups required')
        if a==0:continue
        overall=groups['all']
        passed=(overall['teacher_correct']/512>=.995
                and all(groups[f'rx:{r}']['teacher_correct']/groups[f'rx:{r}']['count']>=.99 for r in RX)
                and overall['metrics']['feature_cosine_error']['mean']<=.001
                and overall['metrics']['posterior_KL']['mean']<=.001)
        checks.append(dict(angle_code=a,passed=bool(passed)))
    return dict(angles=checks,phase_quotient_study_eligible=all(c['passed'] for c in checks))


def aggregate(values, mask):
    return dict(count=int(mask.sum()),teacher_correct=int(values['teacher_agreement'][mask].sum()),
        metrics={key:dict(mean=float(v[mask].mean(dtype=np.float64)),median=float(np.median(v[mask])),
                          p95=float(np.quantile(v[mask],.95))) for key,v in values.items()})


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    p=training.paths(args.root);clean,roles=training.source_training(p)
    ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices]
    day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    if set(rx.tolist())!=set(RX) or set(day.tolist())!={1,2,3}:
        raise ValueError('Original source-only RX/day panel required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:v.clone() for key,v in reader.state_dict().items()}
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        commit=args.commit,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0,
        F_teacher=f_guided.teacher_reference(),source_role_match=True))
    smoke=reader(tensor(clean[indices[:2]],'cpu'))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real source2Ls F checkpoint noquery smoke failed')
    print('{"event":"REAL_SOURCE_F_CHECKPOINT_SMOKE_PASS","Ls":2,"G_reads":0,"target_reads":0}',flush=True)
    started=time.perf_counter();features=[];logits=[];power_error=0.
    for code in range(8):
        current_f=[];current_z=[];angle=code*math.pi/4
        for offset in range(0,512,32):
            x=tensor(clean[indices[offset:offset+32]],'cpu');rotated=rotate(x,angle)
            relative=(rotated.square().sum(1)-x.square().sum(1)).abs().mean(1)/x.square().sum(1).mean(1).clamp_min(1e-8)
            power_error=max(power_error,float(relative.max()))
            feature=reader.features(rotated);score=reader.classify_features(feature)
            if feature.shape!=(len(x),160) or score.shape!=(len(x),6) or not torch.isfinite(feature).all() or not torch.isfinite(score).all():
                raise FloatingPointError('Nonfinite or incomplete source frozen-F output')
            current_f.append(feature.clone());current_z.append(score.clone())
        features.append(torch.cat(current_f));logits.append(torch.cat(current_z))
        print(f'{{"event":"SOURCE_PHASE_ANGLE_COMPLETE","code":{code},"count":512,"seconds":{time.perf_counter()-started}}}',flush=True)
    if power_error>2e-6:raise ValueError('Source phase transform changed power')
    values={};summary={}
    masks={'all':np.ones(512,dtype=bool)}
    for key,labels in dict(role=roles[indices],rx=rx,day=day).items():
        for label in np.unique(labels):masks[f'{key}:{label}']=labels==label
    for code in range(8):
        scores={key:numpy_values(v) for key,v in metrics(features[code],logits[code],features[0],logits[0]).items()}
        if any(v.shape!=(512,) or not np.isfinite(v).all() for v in scores.values()):raise FloatingPointError('Nonfinite source phase metrics')
        for key,v in scores.items():values[f'angle{code}/{key}']=v
        summary[str(code)]={key:aggregate(scores,mask) for key,mask in masks.items()}
    if any(not torch.equal(v,before[key]) for key,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Frozen source F state or gradient changed')
    np.savez(args.output/'values.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
             features=numpy_values(torch.stack(features)),logits=numpy_values(torch.stack(logits)),**values)
    decision=phase_decision(summary)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decision=decision,
        source_rule=RULE,angles_degrees=[45*a for a in range(8)],source_records=512,L_s=128,U_s=384,
        F_feature_analysis_inputs=4096,F_smoke_inputs=2,F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        optimizer_updates=0,G_checkpoint_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        power_error_max=power_error,seconds=time.perf_counter()-started,
        scope='Authorized frozenF sourceclean sensitivity;teacher choices not TX truth;not F-free design,target invariance or restoration performance'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',F_unchanged=True,
        optimizer_updates=0,target_reads=0,goal_complete=False,**decision))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
