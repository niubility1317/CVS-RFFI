"""Frozen-source F geometry; soft posterior coordinates are never TX truth."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import runner as training, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices, tensor, numpy_values

RUN='20261031-diagnostic-DURA-source-F-RXgeometry-s2026092701-r01'
CACHE_RUN='20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01'
RX=(1,3,4,6,8)
RULE=('No optimizer/TX/V/target. Fit62488source records,holdout fixed512(L128/U384). '
      'Original frozen F soft6D posterior weights conditional RX feature means;no hard pseudoTX. '
      'Remove original centered-head rank5 directions from shared RX offsets. '
      'Eligibility: protectedshared/conditionalRXenergy>=.05, leaveoneposteriorcoordinate '
      'mean cosine>.1 and positivefraction>.5, fit-vs-holdout RXdirection mean cosine>.1, '
      'leaveoneRX projectioncapture mean>=.20 and >=.10 for at least4/5 RX. '
      'Head-preserving feature transform alpha.25 is an algebra check,not evidence of actual G/identity efficacy. '
      'Pass permits only subsequent source loss-gradient calibration;no training is launched here.')


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-4)


def basis(rows,max_rank):
    _,singular,vh=np.linalg.svd(rows,full_matrices=False)
    rank=int((singular>singular.max()*1e-7).sum()) if singular.max()>0 else 0
    if rank>max_rank:raise ValueError('Unexpected centered geometry rank')
    return vh[:rank].T.copy()


def protected(x,head_basis):return x-(x@head_basis)@head_basis.T


def cosine(a,b):
    denominator=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(-1),denominator,out=np.zeros_like(denominator),where=denominator>1e-20)


def conditional_means(features,probabilities,rx):
    if features.shape!=(len(rx),160) or probabilities.shape!=(len(rx),6) or set(rx.tolist())!=set(RX):
        raise ValueError('Full five-sourceRX/160D/soft6D rows required')
    means=[];masses=[]
    for receiver in RX:
        take=rx==receiver;weights=probabilities[take];mass=weights.sum(0)
        if np.any(mass<=0):raise ValueError('Nonzero soft-posterior coordinate mass required')
        means.append(weights.T@features[take]/mass[:,None]);masses.append(mass)
    return np.stack(means),np.stack(masses)


def geometry(fit,probe,head_basis):
    if fit.shape!=(5,6,160) or probe.shape!=fit.shape or head_basis.shape!=(160,5):
        raise ValueError('FiveRX/sixsoftcoordinates/originalrank5 head required')
    conditional=fit-fit.mean(0,keepdims=True)
    shared=protected(conditional.mean(1),head_basis)
    denominator=float(np.square(conditional).sum())
    if denominator<=1e-20:raise ValueError('Nonzero conditional RX variation required')
    fraction=6*float(np.square(shared).sum())/denominator
    leave_coordinate=[]
    for coordinate in range(6):
        others=np.arange(6)!=coordinate
        estimate=protected(conditional[:,others].mean(1),head_basis)
        held=protected(conditional[:,coordinate],head_basis)
        leave_coordinate.extend(cosine(estimate,held).tolist())
    probe_shared=protected((probe-probe.mean(0,keepdims=True)).mean(1),head_basis)
    transfer=cosine(shared,probe_shared)
    captures=[];loo_bases=[];held_energies=[]
    for receiver in range(5):
        others=np.arange(5)!=receiver;mean=fit[others].mean(0)
        offsets=protected((fit[others]-mean).mean(1),head_basis)
        q=basis(offsets,3);loo_bases.append(q)
        held=protected((fit[receiver]-mean).mean(0),head_basis)
        energy=float(np.square(held).sum())
        held_energies.append(energy)
        captures.append(float(np.square(held@q).sum())/energy if energy>1e-20 else 0.)
    result=dict(protected_shared_energy_fraction=fraction,
        leave_posterior_coordinate_cosines=leave_coordinate,
        leave_posterior_coordinate_mean_cosine=float(np.mean(leave_coordinate)),
        leave_posterior_coordinate_positive_fraction=float(np.mean(np.asarray(leave_coordinate)>0)),
        holdout_RX_cosines=transfer.tolist(),holdout_RX_mean_cosine=float(transfer.mean()),
        leave_RX_capture=captures,leave_RX_mean_capture=float(np.mean(captures)),
        leave_RX_energy=held_energies,leave_RX_defined=[v>1e-20 for v in held_energies],
        leave_RX_count_capture_ge10percent=int((np.asarray(captures)>=.10).sum()))
    eligible=(fraction>=.05 and result['leave_posterior_coordinate_mean_cosine']>.1
        and result['leave_posterior_coordinate_positive_fraction']>.5 and result['holdout_RX_mean_cosine']>.1
        and result['leave_RX_mean_capture']>=.20 and result['leave_RX_count_capture_ge10percent']>=4)
    result['source_canonical_feature_calibration_eligible']=bool(eligible)
    return result,basis(shared,4),loo_bases


def canonical_feature(features,mean,rx_basis,alpha=.25):
    return features-alpha*((features-mean)@rx_basis)@rx_basis.T


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    p=training.paths(args.root);clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles)
    indices=source_indices(roles);fit_mask=np.ones(63000,dtype=bool);fit_mask[indices]=False
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)
    cache=args.root/'runs'/CACHE_RUN/'scratch_joint/source_feature_cache'
    metadata=training.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,TX_labels_read=0,
                  V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference())
    if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Original authorized source F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: cached feature source physical order')
    cached=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if cached.shape!=(63000,160) or cached.dtype!=np.float32 or not np.isfinite(cached).all():
        raise ValueError('Complete finite original FP32 source F features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference()))
    smoke_features=reader.features(tensor(clean[indices[:2]],'cpu'));smoke=reader.classify_features(smoke_features)
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual source2Ls F checkpoint smoke failed')
    smoke_error=float(np.max(np.abs(numpy_values(smoke_features)-cached[indices[:2]])))
    if smoke_error>3e-5:raise ValueError('Cached source features do not match actual F smoke')
    weight=reader.core.id_backbone.cls_head.weight.detach().double().tolist()
    print(json.dumps(dict(event='REAL_SOURCE_F_CHECKPOINT_SMOKE_PASS',Ls=2,cache_max_abs_error=smoke_error)),flush=True)
    started=time.perf_counter();features=unit(np.asarray(cached,dtype=np.float64));weight=unit(np.asarray(weight))
    head=basis(weight-weight.mean(0),5)
    if head.shape!=(160,5):raise ValueError('Original nondegenerate rank5 head required')
    scores=30*features@weight.T;scores-=scores.max(1,keepdims=True);probabilities=np.exp(scores);probabilities/=probabilities.sum(1,keepdims=True)
    fit,mass_fit=conditional_means(features[fit_mask],probabilities[fit_mask],rx[fit_mask])
    probe,mass_probe=conditional_means(features[indices],probabilities[indices],rx[indices])
    result,rx_basis,loo=geometry(fit,probe,head)
    mean=features[fit_mask].mean(0);canonical=canonical_feature(features[indices],mean,rx_basis)
    old_logits=30*features[indices]@weight.T;new_logits=30*unit(canonical)@weight.T
    agreement=int((old_logits.argmax(1)==new_logits.argmax(1)).sum())
    orthogonality=float(np.max(np.abs(head.T@rx_basis))) if rx_basis.shape[1] else 0.
    if agreement!=512 or orthogonality>1e-10:raise ValueError('Head-preserving feature algebra failed')
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Frozen source F changed')
    arrays=dict(fit_centroids=fit,probe_centroids=probe,mass_fit=mass_fit,mass_probe=mass_probe,
        head_basis=head,RX_basis=rx_basis,normalized_head_weight=weight,fit_feature_mean=mean,
        probe_features=features[indices],probe_probabilities=probabilities[indices],canonical_features=canonical,
        source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx[indices])
    for index,q in enumerate(loo):arrays[f'leave_RX_{RX[index]}_basis']=q
    np.savez(args.output/'geometry.npz',**arrays)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_rule=RULE,geometry=result,
        source_records=63000,fit_records=62488,holdout_records=512,holdout_L=128,holdout_U=384,RX=list(RX),
        soft_posterior_coordinates=6,hard_pseudo_TX_labels=False,head_rank=5,RX_rank=rx_basis.shape[1],
        canonical_alpha=.25,canonical_teacher_choices_preserved=agreement,head_RX_orthogonality_max=orthogonality,
        F_new_feature_inputs=2,G_checkpoint_reads=0,cache_features_reused=63000,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,F_teacher=f_guided.teacher_reference(),
        cache_smoke_max_abs_error=smoke_error,seconds=time.perf_counter()-started,
        scope='Source RX association under soft F coordinates,not causal hardware/TX truth or unseenRX G generalization;F-informed'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',F_unchanged=True,optimizer_updates=0,
        target_reads=0,goal_complete=False,source_canonical_feature_calibration_eligible=result['source_canonical_feature_calibration_eligible']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
