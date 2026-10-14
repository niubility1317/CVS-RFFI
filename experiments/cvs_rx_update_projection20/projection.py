"""Euclidean projection of a supplied AdamW increment onto five risk halfspaces.

No optimizer/model construction, data access, fitted state, learning rate or
radius is introduced. Source/parameter scope legality belongs to the caller.
"""
from collections.abc import Mapping
import math

import torch

SOURCE_RX = (1, 3, 4, 6, 8)
_ROUNDING_FACTOR = 512.  # FP64 numerical check, not a scientific hyperparameter.


def _finite(value, description):
    if not math.isfinite(value): raise ValueError(f'{description} is nonfinite')
    return value


def _norm(values):
    # Scaling protects tiny/large finite rows from squaring under/overflow.
    maximum = max((float(v.detach().abs().max()) for v in values if v is not None and v.numel()),default=0.)
    if maximum == 0.: return 0.
    squares = math.fsum(float((v.detach().double()/maximum).square().sum())
                        for v in values if v is not None)
    return _finite(maximum*math.sqrt(squares),'Vector norm')


def _dot(left,right):
    return _finite(math.fsum(float((a.detach().double()*b.detach().double()).sum())
                            for a,b in zip(left,right) if a is not None and b is not None),'Directional dot product')


def _inputs(d0,gradients):
    if not isinstance(d0,tuple) or not d0:
        raise ValueError('d0 must be a nonempty tuple of existing floating parameter increments')
    for value in d0:
        if (not isinstance(value,torch.Tensor) or value.layout != torch.strided
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError('Every d0 item must be finite, real, floating and dense')
    if not isinstance(gradients,Mapping):
        raise ValueError('Five source RX gradient tuples are required')
    normalized_keys = {}
    for key,row in gradients.items():
        if type(key) is int and key in SOURCE_RX: rx = key
        elif type(key) is str and key in tuple(map(str,SOURCE_RX)): rx = int(key)
        else: raise ValueError('Only native source RX1/3/4/6/8 gradient keys are permitted')
        if rx in normalized_keys: raise ValueError('Duplicate source RX gradient key')
        if not isinstance(row,tuple) or len(row) != len(d0):
            raise ValueError('Each RX gradient must be a same-length tuple; unused items may be None')
        for gradient,delta in zip(row,d0):
            if gradient is not None and (not isinstance(gradient,torch.Tensor)
                    or gradient.layout != torch.strided or gradient.shape != delta.shape
                    or gradient.dtype != delta.dtype or gradient.device != delta.device
                    or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Gradient shape/dtype/device/finite contract differs from d0')
        normalized_keys[rx] = row
    if set(normalized_keys) != set(SOURCE_RX):
        raise ValueError('All five source RX risk gradients must be supplied')
    return normalized_keys


def _candidate(base,units,weights):
    result=[]
    for column,value in enumerate(base):
        projected=value.clone()
        for row,weight in zip(units,weights.tolist()):
            if row[column] is not None and weight:
                projected.sub_(row[column]*weight)
        result.append(projected)
    return tuple(result)


def _stationarity(base,delta,units,weights):
    residual=[]
    for column,(original,projected) in enumerate(zip(base,delta)):
        value=projected-original
        for row,weight in zip(units,weights.tolist()):
            if row[column] is not None and weight: value=value+row[column]*weight
        residual.append(value)
    return _norm(residual)


def project(d0,gradients_by_RX,role='L_s'):
    """min .5||d-d0||² subject to g_RX dot d <=0, with FP64 KKT checks.

    Input tuples exclude W/fusion gains as asserted by the caller. None means
    an unused parameter. Positive row normalization changes no halfspace.
    At most32 active sets use only a <=5x5 FP64 Gram/pseudoinverse. Candidate
    residuals are checked again in the supplied high-dimensional coordinates.
    Returned projected_delta preserves each input dtype/device; its cast risks
    are separate from FP64 risks. Actual theta_after-theta_before is not known
    to this function and must be measured independently by the caller.
    """
    if role != 'L_s': raise ValueError('Only caller-validated visible source L_s risk gradients are permitted')
    raw = _inputs(d0,gradients_by_RX)
    norms = {rx:_norm(raw[rx]) for rx in SOURCE_RX}
    nonzero = [rx for rx in SOURCE_RX if norms[rx] != 0.]
    units = [tuple(None if value is None else value.detach().double()/norms[rx] for value in raw[rx])
             for rx in nonzero]
    base = tuple(value.detach().double() for value in d0)
    original_norm = _norm(base)
    tolerance = _ROUNDING_FACTOR*torch.finfo(torch.float64).eps*original_norm
    h = torch.tensor([_dot(row,base) for row in units],dtype=torch.float64)
    count=len(units); sets=valid=rank_deficient=0
    best = None; best_weights=torch.zeros(count,dtype=torch.float64); best_active=[]
    best_risks=h.tolist(); best_stationarity=0.; best_active_residual=0.
    if bool((h <= 0).all()):
        # Preserve the exact tuple, values, dtypes and devices of feasible d0.
        output = d0; ideal = base; status='already_feasible'; projection_norm=0.
    else:
        gram=torch.empty((count,count),dtype=torch.float64)
        for i,row in enumerate(units):
            for j in range(i+1): gram[i,j]=gram[j,i]=_dot(row,units[j])
        for mask in range(1<<count):
            sets += 1
            active=[i for i in range(count) if mask & (1<<i)]
            weights=torch.zeros(count,dtype=torch.float64)
            if active:
                sub=gram[active][:,active]
                rank=int(torch.linalg.matrix_rank(sub,hermitian=True))
                rank_deficient += int(rank < len(active))
                inverse=torch.linalg.pinv(sub,hermitian=True)
                selected=inverse@h[active]
                if not bool(torch.isfinite(selected).all()): continue
                # Refine using residuals measured in original coordinates.
                # Gram cancellation near opposite rows otherwise makes a
                # small linear-solve residual hide a larger actual dot risk.
                for _ in range(3):
                    weights[active]=selected
                    candidate=_candidate(base,units,weights)
                    residual=torch.tensor([_dot(units[i],candidate) for i in active],dtype=torch.float64)
                    if float(residual.abs().max()) <= tolerance: break
                    selected=selected+inverse@residual
                if not bool(torch.isfinite(selected).all()) or float(selected.min()) < -tolerance: continue
                weights[active]=selected.clamp_min(0.)
            candidate=_candidate(base,units,weights)
            if not all(bool(torch.isfinite(value).all()) for value in candidate): continue
            risks=[_dot(row,candidate) for row in units]
            if max(risks,default=0.) > tolerance: continue
            active_residual=max((abs(risks[i]) for i in active),default=0.)
            if active_residual > tolerance: continue
            stationarity=_stationarity(base,candidate,units,weights)
            if stationarity > tolerance: continue
            valid += 1
            distance=_norm(tuple(projected-original for projected,original in zip(candidate,base)))
            if best is None or distance < projection_norm:
                best=candidate;projection_norm=distance;best_weights=weights;best_active=active
                best_risks=risks;best_stationarity=stationarity;best_active_residual=active_residual
        if best is None:
            raise RuntimeError(f'No FP64 KKT-feasible projection among{sets} active sets at d0-scaled tolerance{tolerance}')
        ideal=best;output=tuple(value.to(dtype=original.dtype,device=original.device).detach() for value,original in zip(best,d0))
        status='projected'
    ideal_norm=_norm(ideal);cast_norm=_norm(output)
    ideal_objective=_finite(.5*projection_norm*projection_norm,'Projection objective')
    by_rx={};cast_normalized_risks=[];ideal_normalized_risks=[]
    index_by_rx={rx:i for i,rx in enumerate(nonzero)}
    for rx in SOURCE_RX:
        i=index_by_rx.get(rx)
        normalized_ideal=0. if i is None else _dot(units[i],ideal)
        normalized_cast=0. if i is None else _dot(units[i],output)
        cast_normalized_risks.append(normalized_cast);ideal_normalized_risks.append(normalized_ideal)
        by_rx[str(rx)]=dict(gradient_norm=norms[rx],zero_gradient_constraint=norms[rx]==0.,
            unused_parameter_tensors=sum(value is None for value in raw[rx]),
            raw_directional_risk=_dot(raw[rx],d0),FP64_projected_directional_risk=_dot(raw[rx],ideal),
            cast_directional_risk=_dot(raw[rx],output),
            FP64_normalized_directional_risk=normalized_ideal,cast_normalized_directional_risk=normalized_cast,
            FP64_raw_risk_tolerance=_finite(tolerance*norms[rx],'Risk tolerance'),
            normalized_dual=0. if i is None else float(best_weights[i]),
            active_set_member=i in best_active if i is not None else False)
    summary=dict(status=status,role=role,parameter_tensors=len(d0),coordinates=sum(v.numel() for v in d0),
        nonzero_constraints=count,ignored_zero_constraints=5-count,original_norm=original_norm,
        projected_FP64_norm=ideal_norm,projected_cast_norm=cast_norm,projection_norm=projection_norm,
        retained_norm_ratio=ideal_norm/original_norm if original_norm else 1.,
        cast_retained_norm_ratio=cast_norm/original_norm if original_norm else 1.,
        relative_projection_norm=projection_norm/original_norm if original_norm else 0.,
        objective=ideal_objective,active_RX=[nonzero[i] for i in best_active],
        active_sets_enumerated=sets,KKT_feasible_candidates=valid,rank_deficient_active_sets=rank_deficient,
        normalized_risk_tolerance=tolerance,solver_stationarity_residual_norm=best_stationarity,
        solver_active_constraint_residual=best_active_residual,
        solver_primal_violation=max(0.,max(ideal_normalized_risks,default=0.)),
        FP64_strictly_nonpositive=max(ideal_normalized_risks,default=0.)<=0.,
        FP64_feasible_with_numeric_tolerance=all(risk<=tolerance for risk in ideal_normalized_risks),
        cast_primal_violation=max(0.,max(cast_normalized_risks,default=0.)),
        cast_feasible_with_FP64_tolerance=all(risk<=tolerance for risk in cast_normalized_risks),
        by_RX=by_rx,row_normalization='positive global L2 norm; halfspaces unchanged',
        solver='FP64 <=5x5 Gram, <=32 active sets, pseudoinverse plus original-coordinate residual refinement',
        actual_parameter_addition_not_measured=True,new_parameters=0,optimizer_updates=0,
        source_provenance_asserted_by_caller=True,parameter_scope_asserted_by_caller=True,
        boundary='Pure solver cannot prove source data/physicalIDs or W/gain exclusion; cast delta is not actual parameter displacement. Related gradient projection priors exist; no novelty or efficacy claim')
    return dict(projected_delta=output,summary=summary)
