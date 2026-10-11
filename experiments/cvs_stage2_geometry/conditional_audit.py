"""Source-only cross-TX audit of sample-dependent receiver displacement.

This module fits no identity weights and exports no training targets. The oracle
explicitly uses held-out displacements and is NEVER a predictive result.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import torch

from . import design as d
from .receiver import FOLDS, _source, _build_groups, _fit_fold, _check_features

RUN = '20261011-phase1-stage2-conditional-geometry-source-manysig-m3-r01'
RELEASE = 'cvs_stage2_conditional_geometry_source_20261011_r01'
RIDGE = .1
ORACLE_RCOND = 1e-6
METHODS = ('global_rank2', 'global_full', 'tangent_state',
           'global_environment', 'tangent_environment')


def center_blocks(value, blocks):
    result = value.clone()
    for b in blocks:
        ids = b['group_ids']
        result[ids] -= value[ids].mean(0)
    return result


def group_weights(blocks, count):
    w = torch.zeros(count, dtype=torch.float64)
    for b in blocks:
        w[b['group_ids']] = 1. / (len(blocks) * len(b['group_ids']))
    return w


def fit_linear(design, target, weights, *, normalize_columns):
    """Weighted group objective; last axis is coefficients, no held-out fit.

    design[G,D,P], target[G,D], weights[G]. Column RMS uses only fit groups,
    summing output energy. Ridge is fixed; no validation-based tuning.
    """
    keep = weights > 0
    x = design[keep].double()
    y = target[keep].double()
    w = weights[keep].double()
    scale = (torch.einsum('gdp,gdp,g->p', x, x, w).sqrt().clamp_min(1e-8)
             if normalize_columns else x.new_ones(x.shape[-1]))
    x = x / scale
    gram = torch.einsum('gdp,gdq,g->pq', x, x, w)
    rhs = torch.einsum('gdp,gd,g->p', x, y, w)
    coef = torch.linalg.solve(gram + RIDGE * torch.eye(len(rhs), dtype=x.dtype), rhs) / scale
    return coef, dict(ridge=RIDGE, columns=len(coef), fit_groups=int(keep.sum()),
                      column_scale=scale.tolist(), normalized_columns=normalize_columns)


def metric(truth, prediction, blocks):
    mse, zero, energy, directional = [], [], [], []
    n = 0
    for b in blocks:
        pairs = list(combinations(b['group_ids'], 2))
        delta = torch.stack([truth[g]-truth[h] for g,h in pairs])
        pred = torch.stack([prediction[g]-prediction[h] for g,h in pairs])
        mse.append((pred-delta).square().sum(1).mean())
        zero.append(delta.square().sum(1).mean())
        energy.append(pred.square().sum(1).mean())
        denom = pred.norm(dim=1) * delta.norm(dim=1)
        directional.append(torch.where(denom > 1e-12,
            (pred*delta).sum(1)/denom.clamp_min(1e-12), torch.zeros_like(denom)).mean())
        n += len(pairs)
    mp, mz = float(torch.stack(mse).mean()), float(torch.stack(zero).mean())
    ratio = mp/mz if mz > 1e-12 else None
    return dict(mse_pred=mp, mse_zero=mz, mse_ratio=ratio,
        c=max(0., min(1., 1.-ratio)) if ratio is not None else 0.,
        prediction_energy=float(torch.stack(energy).mean()),
        direction_cosine_zero_for_zero_prediction=float(torch.stack(directional).mean()),
        blocks=len(blocks), pairs=n, weighting='equal_block_then_equal_RX_pair')


def tangent_prediction(jacobian, actions, blocks, bounded=False):
    actions = actions.clone()
    factors = []
    for b in blocks:
        ids = b['group_ids']
        maximum = float(actions[ids].abs().max())
        beta = min(1., 1./maximum) if maximum else 1.
        if bounded:
            actions[ids] *= beta
        factors.append(dict(block_id=b['block_id'], beta=beta,
                            raw_max_abs_coefficient=maximum))
    prediction = center_blocks(torch.einsum('gdk,gk->gd', jacobian, actions), blocks)
    return prediction, factors


def tangent_oracle(jacobian, truth, blocks):
    """Per-block group action oracle: upper bound on local-span representation.

    Uses truth in the held-out block. Its coefficients are never reusable model
    coefficients or reliability estimates. Bounding the minimum-norm solution
    is a diagnostic, NOT the optimal box-constrained oracle.
    """
    pred = torch.zeros_like(truth)
    actions = jacobian.new_zeros((len(jacobian), jacobian.shape[-1]))
    ranks = []
    for b in blocks:
        ids=b['group_ids'];g=len(ids);dim=truth.shape[-1];k=jacobian.shape[-1]
        operator=jacobian.new_zeros((g,dim,g,k))
        for pos,gid in enumerate(ids):operator[pos,:,pos,:]=jacobian[gid]
        operator-=operator.mean(0,keepdim=True)
        matrix=operator.reshape(g*dim,g*k)
        target=(truth[ids]-truth[ids].mean(0)).reshape(-1)
        result=torch.linalg.lstsq(matrix,target,driver='gelsd',rcond=ORACLE_RCOND)
        actions[ids]=result.solution.reshape(g,k)
        pred[ids]=(matrix@result.solution).reshape(g,dim)
        ranks.append(dict(block_id=b['block_id'],rank=int(result.rank),columns=g*k))
    bounded,factors=tangent_prediction(jacobian,actions,blocks,bounded=True)
    return pred,bounded,dict(ranks=ranks,bounds=factors,rcond=ORACLE_RCOND,
        predictive=False,heldout_truth_used=True,
        bounded_interpretation='uniformly shrunk unconstrained minimum-norm solution; not constrained optimum')


def analyze(cache, jacobian):
    data=_source(cache);z=cache['parent_feature'].detach().cpu().double()
    _check_features(z,len(data['state']))
    j=torch.as_tensor(jacobian).detach().cpu().double()
    if j.shape!=(len(z),160,8) or not torch.isfinite(j).all():
        raise ValueError('Finite cached own-sample joint160 Jacobian [N,160,8] required')
    groups,blocks,decisions,_=_build_groups(data)
    if not blocks:raise ValueError('No legal source relation blocks')
    means=torch.stack([z[g['indices']].mean(0) for g in groups])
    js=torch.stack([j[g['indices']].mean(0) for g in groups])
    state=torch.stack([data['state'][g['indices']].double().mean(0) for g in groups])
    target=center_blocks(means,blocks)
    for g,m in zip(groups,means):g['parent_feature_mean']=m
    environments=sorted({(g['rx'],g['day']) for g in groups})
    env=torch.zeros((len(groups),len(environments)),dtype=torch.float64)
    for gi,g in enumerate(groups):env[gi,environments.index((g['rx'],g['day']))]=1.
    envc=center_blocks(env,blocks)
    folds=[];fitted={}
    for fi,held in enumerate(FOLDS):
        fit_tx=tuple(tx for tx in range(6) if tx not in held)
        fb=[b for b in blocks if b['tx'] in fit_tx];hb=[b for b in blocks if b['tx'] in held]
        weights=group_weights(fb,len(groups));old=_fit_fold(data,z,groups,blocks,fit_tx)
        ss=center_blocks((state-old['standard_mean'])/old['standard_scale'],blocks)
        # Each group uses its own mean Jacobian: E[J_i a_g] = E[J_i] a_g.
        ts=center_blocks(torch.einsum('gdk,gs->gdks',js,ss).flatten(2),blocks)
        te=center_blocks(torch.einsum('gdk,ge->gdke',js,env).flatten(2),blocks)
        ws,diag_s=fit_linear(ts,target,weights,normalize_columns=True)
        we,diag_e=fit_linear(te,target,weights,normalize_columns=True)
        # Full-vector global comparators use exactly the old state scale and
        # ridge; the environment comparison uses the same weighted objective.
        def global_fit(x):
            xx=x.T@(weights[:,None]*x)
            return torch.linalg.solve(xx+RIDGE*torch.eye(x.shape[1],dtype=x.dtype),
                                      x.T@(weights[:,None]*target))
        full=global_fit(ss);global_env=global_fit(envc)
        state_actions=ss@ws.reshape(8,40).T
        env_actions=env@we.reshape(8,len(environments)).T
        predictions=dict(global_rank2=ss@old['matrix'].T,global_full=ss@full,
                         tangent_state=ts@ws,global_environment=envc@global_env,
                         tangent_environment=te@we)
        results={name:dict(heldout=metric(means,p,hb),fit=metric(means,p,fb))
                 for name,p in predictions.items()}
        for name,actions in [('tangent_state',state_actions),('tangent_environment',env_actions)]:
            raw,_=tangent_prediction(js,actions,blocks)
            if not torch.allclose(raw,predictions[name],atol=1e-9,rtol=1e-9):
                raise ValueError('Action parameterization and linear design disagree')
            bound,factors=tangent_prediction(js,actions,blocks,bounded=True)
            results[name].update(bounded_heldout=metric(means,bound,hb),
                bounded_fit=metric(means,bound,fb),
                bounds=[v for v in factors if blocks[v['block_id']]['tx'] in held])
        oracle,bounded,oracle_info=tangent_oracle(js,means,hb)
        folds.append(dict(fold=fi,fit_tx=list(fit_tx),heldout_tx=list(held),
            predictive=results,oracle=dict(raw=metric(means,oracle,hb),
                bounded_shrink=metric(means,bounded,hb),**oracle_info),
            fit_diagnostics=dict(tangent_state=diag_s,tangent_environment=diag_e),
            fit_physical_ids=sum(g['count'] for g in groups if g['tx'] in fit_tx),
            heldout_physical_ids=sum(g['count'] for g in groups if g['tx'] in held)))
        fitted[str(fi)]=dict(tangent_state=ws,tangent_environment=we,global_full=full,
            global_environment=global_env,state_scale=old['standard_scale'],
            state_center=old['standard_mean'],fit_tx=fit_tx,heldout_tx=held,
            status='DIAGNOSTIC_ONLY_NOT_A_TRAINING_TARGET')
    return dict(folds=folds,source_count=len(z),groups=len(groups),blocks=len(blocks),
        environments=[list(e) for e in environments],grouping_decisions=decisions,
        scope='auxiliary cross-TX only; parent identity network saw all six source TX',
        target_access=False,validation_access=False,identity_updates=0,
        real_iq_verification=False,readout_parameter_search=False),fitted


def run(output=None):
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_stage2_decouple.common import verify_parent
    out=Path(output) if output else Path(d.PROJECT)/'runs'/RUN
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(2)
    commit=(d.ROOT/'release_commit.txt').read_text(encoding='utf-8').strip()
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Preregistration differs')
    spec.update(status='RUNNING',execution_commit=commit)
    d.write(out/'experiment.json',spec)
    def event(status,**kw):
        with (out/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(at=time.time(),status=status,**kw))+'\n')
    event('RUNNING',execution_commit=commit)
    results=[]
    try:
        for seed in d.SEEDS:
            begin=time.perf_counter();old=d.AUDIT_BASE/f's{seed}'
            done=d.read(old/'preparation_complete.json');c=d.parent_config(seed)
            if (done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or
                done['parent_contract']!=c or done['identity_updates']!=0):
                raise ValueError('Cached geometry provenance mismatch')
            provenance=verify_parent(c)
            packed=torch.load(old/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE))
            cache=packed['labeled']
            if len(cache['ids'])!=6300 or sorted(cache['ids'])!=sorted(packed['contract']['role_ids']['L_s']):
                raise ValueError('Source physical IDs differ')
            response=torch.load(old/'source_responses.pt',map_location='cpu',weights_only=False)
            if response['source_ids']!=cache['ids']:raise ValueError('Jacobian sample ordering differs')
            result,fitted=analyze(cache,response['jacobian'])
            result.update(seed=seed,execution_commit=commit,parent_provenance=provenance,
                input_source=str(old),elapsed_seconds=time.perf_counter()-begin,
                hardware=dict(cpu=platform.processor(),cuda_used=False,threads=2))
            row=out/f's{seed}';row.mkdir();torch.save(fitted,row/'diagnostic_coefficients.pt')
            d.write(row/'audit.json',result)
            d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,
                target_access=False,identity_updates=0,source_count=6300,audit=str(row/'audit.json')))
            results.append(dict(seed=seed,folds=[dict(fold=f['fold'],
                predictive={name:dict(raw=x['heldout']['mse_ratio'],
                bounded=x.get('bounded_heldout',{}).get('mse_ratio')) for name,x in f['predictive'].items()},
                oracle=f['oracle']['raw']['mse_ratio'],
                oracle_bounded_shrink=f['oracle']['bounded_shrink']['mse_ratio']) for f in result['folds']]))
            event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
            print(json.dumps(results[-1]),flush=True)
        d.write(out/'summary.json',results)
        spec.update(status='SOURCE_DIAGNOSTIC_COMPLETE')
        for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec)
        d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=RUN,
            seeds=list(d.SEEDS),rows_completed=3,target_access=False,identity_updates=0,
            elapsed_seconds=time.perf_counter()-started,execution_commit=commit))
        event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)))
        event('FAILED',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    run(args.output)
