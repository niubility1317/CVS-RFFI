"""Bounded local representation and cross-day source RX relation diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import lsq_linear
import torch

from . import design as d
from .receiver import _source,_build_groups,_check_features,FOLDS
from .conditional_audit import metric,center_blocks,tangent_oracle

RUN='20261011-phase1-stage2-relation-identifiability-source-manysig-m3-r01'
RELEASE='cvs_stage2_relation_identifiability_source_20261011_r01'
SHRINK=.4
BOX_TOL=1e-10
BOX_MAX_ITER=1000
SPLIT_SALT='source-relation-split-v1'


def solve_box(matrix,target):
    a=np.asarray(matrix,dtype=np.float64);b=np.asarray(target,dtype=np.float64)
    solution=lsq_linear(a,b,bounds=(-1.,1.),method='bvls',tol=BOX_TOL,
                        max_iter=BOX_MAX_ITER,lsq_solver='exact')
    x=solution.x;gradient=a.T@(a@x-b)
    projected=float(np.max(np.abs(x-np.clip(x-gradient,-1.,1.))))
    box_violation=float(max(0.,np.abs(x).max()-1.))
    # Convex first-order lower bound over the entire box: min_y g.(y-x).
    # This independently bounds primal suboptimality without relying on status.
    dual_gap=float(np.sum(np.abs(gradient)+gradient*x))
    cost=float(.5*np.square(a@x-b).sum())
    diag=dict(success=bool(solution.success),status=int(solution.status),
        iterations=int(solution.nit),solver_optimality=float(solution.optimality),
        projected_gradient_inf=projected,box_violation=box_violation,
        objective=cost,convex_gap_upper_bound=max(0.,dual_gap),
        gap_relative_to_zero=max(0.,dual_gap)/max(.5*float(b@b),1e-12),
        active_coordinates=int(np.count_nonzero(np.abs(x)>=1-1e-8)),
        coordinates=len(x),max_abs=float(np.abs(x).max()))
    if not solution.success or box_violation>1e-9 or projected>1e-7 or diag['gap_relative_to_zero']>1e-6:
        raise RuntimeError('Box oracle not numerically verified: '+repr(diag))
    # N607 has a historical Torch/NumPy ABI mismatch. These small CPU matrices
    # cross via explicit values, never the tensor/ndarray shared-memory bridge.
    return torch.tensor(x.tolist(),dtype=torch.float64),diag


def bounded_oracle(jacobian,truth,blocks):
    prediction=torch.zeros_like(truth);details=[]
    for b in blocks:
        ids=b['group_ids'];g=len(ids);dim=truth.shape[-1];k=jacobian.shape[-1]
        operator=truth.new_zeros((g,dim,g,k))
        for pos,gid in enumerate(ids):operator[pos,:,pos,:]=jacobian[gid]
        operator-=operator.mean(0,keepdim=True)
        matrix=operator.reshape(g*dim,g*k)
        target=(truth[ids]-truth[ids].mean(0)).reshape(-1)
        solution,diag=solve_box(matrix.tolist(),target.tolist())
        prediction[ids]=(matrix@solution).reshape(g,dim)
        details.append(dict(block_id=b['block_id'],tx=b['tx'],day=b['day'],**diag))
    return prediction,dict(predictive=False,heldout_truth_used=True,
        scope='group-constant 8D local linear actions in [-1,1]; NOT finite-IQ optimality',
        solver='scipy.optimize.lsq_linear bvls',tol=BOX_TOL,max_iter=BOX_MAX_ITER,blocks=details)


def split_group_means(cache,groups):
    z=cache['parent_feature'].detach().cpu().double();a=[];b=[];splits=[]
    for group in groups:
        ordered=sorted(group['indices'],key=lambda i:hashlib.sha256(
            (SPLIT_SALT+'|'+str(cache['ids'][i])).encode('utf-8')).digest())
        # The registered source cache is balanced, with an even 70 per group.
        if len(ordered)%2:raise ValueError('Equal disjoint halves require even source group counts')
        left,right=ordered[:len(ordered)//2],ordered[len(ordered)//2:]
        if set(left)&set(right):raise ValueError('Source half overlap')
        a.append(z[left].mean(0));b.append(z[right].mean(0))
        splits.append(dict(group_id=group['group_id'],half_a_indices=left,half_b_indices=right))
    return torch.stack(a),torch.stack(b),splits


def grid(means,groups):
    txs=sorted({g['tx'] for g in groups});days=sorted({g['day'] for g in groups});rxs=sorted({g['rx'] for g in groups})
    out=means.new_empty((len(txs),len(days),len(rxs),means.shape[-1]));seen=set()
    for i,g in enumerate(groups):
        key=(txs.index(g['tx']),days.index(g['day']),rxs.index(g['rx']))
        if key in seen:raise ValueError('Multiple quality cells: grid decomposition not applicable')
        seen.add(key);out[key]=means[i]
    if len(seen)!=len(txs)*len(days)*len(rxs):raise ValueError('Incomplete factorial grid')
    return out,(txs,days,rxs)


def components(value):
    """Orthogonal descriptive ANOVA of RX-centered balanced group means."""
    v=value-value.mean(2,keepdim=True)
    receiver=v.mean((0,1),keepdim=True).expand_as(v)
    rx_day=v.mean(0,keepdim=True).expand_as(v)-receiver
    tx_rx=v.mean(1,keepdim=True).expand_as(v)-receiver
    interaction=v-receiver-rx_day-tx_rx
    return dict(receiver=receiver,receiver_day=rx_day,tx_receiver=tx_rx,
                tx_receiver_day=interaction),v


def component_evidence(full,left,right):
    parts,centered=components(full);pa,_=components(left);pb,_=components(right)
    total=float(centered.square().sum(-1).mean())
    rows={}
    for name,p in parts.items():
        energy=float(p.square().sum(-1).mean())
        cross=float((pa[name]*pb[name]).sum(-1).mean())
        split_noise=float(.25*(pa[name]-pb[name]).square().sum(-1).mean())
        if abs(energy-cross-split_noise)>1e-10:raise ValueError('Disjoint-half energy identity differs')
        rows[name]=dict(full_energy=energy,full_fraction=energy/total if total else None,
            half_cross_energy=cross,half_difference_energy=split_noise,
            half_cross_fraction_of_full_total=cross/total if total else None)
    if abs(sum(r['full_energy'] for r in rows.values())-total)>1e-10:
        raise ValueError('Balanced component energies do not sum')
    return dict(total_rx_centered_energy=total,components=rows,
        interpretation='descriptive second moments; cross-half noise interpretation assumes independent zero-mean packet errors, not guaranteed by disjoint IDs',
        negative_cross_energy_retained=True)


def shared_prediction(means,groups,blocks,fit_tx,fit_days,condition_day):
    """Source categorical controls fit only requested TX/day cells.

    RX-only effects average equal source cells. RX/day needs the same source
    days in fit and evaluation; no new/target receiver prediction is implied.
    """
    centered=center_blocks(means,blocks)
    fit=[i for i,g in enumerate(groups) if g['tx'] in fit_tx and g['day'] in fit_days]
    bank={}
    for i in fit:
        g=groups[i];key=(g['rx'],g['day']) if condition_day else (g['rx'],)
        bank.setdefault(key,[]).append(centered[i])
    bank={key:torch.stack(values).mean(0) for key,values in bank.items()}
    output=torch.zeros_like(means);supported=[]
    for i,g in enumerate(groups):
        key=(g['rx'],g['day']) if condition_day else (g['rx'],)
        if key in bank:output[i]=bank[key];supported.append(i)
    # Center only within each block, never across fit and held TX/day.
    return center_blocks(output,blocks),supported


def transfer_evidence(means,half_a,half_b,groups,blocks):
    days=sorted({g['day'] for g in groups});rows=[]
    for fold,held in enumerate(FOLDS):
        fit_tx=[tx for tx in range(6) if tx not in held]
        for mode in ('rx_day_same_days','rx_only_same_days','rx_only_held_day'):
            for held_day in (days if mode=='rx_only_held_day' else [None]):
                fit_days=[day for day in days if day!=held_day]
                eval_blocks=[b for b in blocks if b['tx'] in held and (held_day is None or b['day']==held_day)]
                for fit_role,fit_value,eval_role,eval_value in [('full',means,'full',means),
                    ('half_a',half_a,'half_b',half_b),('half_b',half_b,'half_a',half_a)]:
                    pred,supported=shared_prediction(fit_value,groups,blocks,fit_tx,fit_days,mode=='rx_day_same_days')
                    required={i for b in eval_blocks for i in b['group_ids']}
                    if not required.issubset(supported):raise ValueError('Unseen categorical environment in evaluation')
                    for scale in (1.,SHRINK):
                        rows.append(dict(fold=fold,heldout_tx=list(held),heldout_day=held_day,mode=mode,
                            fit_tx=fit_tx,fit_days=fit_days,fit_role=fit_role,evaluation_role=eval_role,
                            scale=scale,**metric(eval_value,scale*pred,eval_blocks)))
    return rows


def analyze(cache,jacobian):
    data=_source(cache);z=cache['parent_feature'].detach().cpu().double()
    _check_features(z,len(data['state']))
    j=torch.as_tensor(jacobian).detach().cpu().double()
    if j.shape!=(len(z),160,8) or not torch.isfinite(j).all():raise ValueError('Jacobian mismatch')
    groups,blocks,decisions,_=_build_groups(data)
    means=torch.stack([z[g['indices']].mean(0) for g in groups])
    js=torch.stack([j[g['indices']].mean(0) for g in groups])
    a,b,splits=split_group_means(cache,groups)
    if not torch.allclose(means,(a+b)/2,atol=1e-12,rtol=0):raise ValueError('Half means differ from full cache')
    full_grid,axes=grid(means,groups);a_grid,_=grid(a,groups);b_grid,_=grid(b,groups)
    box,details=bounded_oracle(js,means,blocks)
    raw,shrink,oracle=tangent_oracle(js,means,blocks)
    if metric(means,box,blocks)['mse_pred']>metric(means,shrink,blocks)['mse_pred']+1e-9:
        raise ValueError('Box optimum worse than feasible uniformly scaled solution')
    result=dict(source_count=len(z),groups=len(groups),blocks=len(blocks),axes=axes,
        grouping_decisions=decisions,components=component_evidence(full_grid,a_grid,b_grid),
        transfer=transfer_evidence(means,a,b,groups,blocks),
        oracle=dict(box=metric(means,box,blocks),unbounded=metric(means,raw,blocks),
            uniform_shrink=metric(means,shrink,blocks),**details),
        target_access=False,identity_updates=0,real_iq_verification=False,
        split_contract=dict(salt=SPLIT_SALT,method='SHA256 ordered physical IDs, disjoint equal halves; same split for all model seeds',
                            statistically_independent_packets_verified=False))
    return result,splits


def run(output=None):
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_stage2_decouple.common import verify_parent
    out=Path(output) if output else Path(d.PROJECT)/'runs'/RUN
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Preregistration mismatch')
    spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit
    d.write(out/'experiment.json',spec)
    def event(status,**kwargs):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kwargs))+'\n')
    event('RUNNING');summary=[]
    try:
        for seed in d.SEEDS:
            begin=time.perf_counter();source=d.AUDIT_BASE/f's{seed}';done=d.read(source/'preparation_complete.json');c=d.parent_config(seed)
            if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or done['identity_updates'] or done['parent_contract']!=c:
                raise ValueError('Source preparation provenance mismatch')
            provenance=verify_parent(c);packed=torch.load(source/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));cache=packed['labeled']
            if len(cache['ids'])!=6300 or sorted(cache['ids'])!=sorted(packed['contract']['role_ids']['L_s']):raise ValueError('Source IDs differ')
            response=torch.load(source/'source_responses.pt',map_location='cpu',weights_only=False)
            if response['source_ids']!=cache['ids']:raise ValueError('Source response order differs')
            result,splits=analyze(cache,response['jacobian']);row=out/f's{seed}';row.mkdir()
            result.update(seed=seed,parent_provenance=provenance,input_source=str(source),execution_commit=commit,
                          elapsed_seconds=time.perf_counter()-begin,environment=dict(scipy=scipy.__version__,torch=torch.__version__,cuda_used=False,threads=2))
            d.write(row/'audit.json',result);d.write(row/'source_split_indices.json',splits)
            d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,target_access=False,identity_updates=0,audit=str(row/'audit.json')))
            summary.append(dict(seed=seed,oracle_box=result['oracle']['box'],components=result['components'],elapsed_seconds=result['elapsed_seconds']))
            print(json.dumps(summary[-1]),flush=True);event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        d.write(out/'summary.json',summary);spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec)
        d.write(out/'completion.json',dict(run_id=RUN,status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,
            target_access=False,identity_updates=0,execution_commit=commit,elapsed_seconds=time.perf_counter()-started))
        event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();run(a.output)
