"""Static deployed feature scope and a gradient-cone comparator to update projection."""
from collections.abc import Sequence

import torch
from torch import nn

from .projection import project, SOURCE_RX, _norm, _dot

FEATURE_PREFIX = 'id_backbone.encoder.'


def feature_scope(model, optimizer_parameters, optimizer_names=None):
    """All actual encoder parameters, except existing W/PA/ref/SIG fusion gains.

    Selection is static by model/optimizer identity, never by current gradient
    presence or magnitude. Norm affine, PA/frequency/reference/SIG_TIME and
    Sinc transformations are retained. Other networks are outside the prefix.
    """
    from .materialize import protected_names
    if not isinstance(model, nn.Module) or not isinstance(optimizer_parameters, (tuple,list)) or not optimizer_parameters:
        raise ValueError('Existing model and exact optimizer parameter traversal required')
    named=list(model.named_parameters());by_id={id(p):name for name,p in named}
    if len({id(p) for p in optimizer_parameters}) != len(optimizer_parameters):
        raise ValueError('Duplicate optimizer parameter object')
    if any(id(p) not in by_id for p in optimizer_parameters):
        raise ValueError('Optimizer parameter is outside the supplied actual model')
    names=tuple(by_id[id(p)] for p in optimizer_parameters)
    if optimizer_names is not None and (not isinstance(optimizer_names,Sequence)
            or isinstance(optimizer_names,(str,bytes)) or tuple(optimizer_names) != names):
        raise ValueError('Optimizer names must match each actual parameter identity and order')
    protected=tuple(protected_names(model))
    if len(protected) != 4 or len(set(protected)) != 4 or any(name not in dict(named) for name in protected):
        raise ValueError('Four actual default W/PA/reference/SIG fusion parameter names required')
    indices=tuple(i for i,name in enumerate(names) if name.startswith(FEATURE_PREFIX) and name not in protected)
    if not indices: raise ValueError('Actual deployed identity encoder feature scope is empty')
    feature_names=tuple(names[i] for i in indices)
    summary=dict(optimizer_tensors=len(names),feature_tensors=len(indices),
        feature_coordinates=sum(optimizer_parameters[i].numel() for i in indices),
        protected_tensor_count=4,protected_tensors_in_optimizer=sum(name in names for name in protected),
        other_optimizer_tensors=len(names)-len(indices),static_selection=True,
        selected_prefix=FEATURE_PREFIX,nonzero_gradient_selection=False,
        default_W_PA_reference_SIG_gains_excluded=True,other_networks_excluded=True,
        new_parameters=0,model_parameter_mutations=0)
    return dict(indices=indices,names=feature_names,summary=summary)


def gradient_cone(base_grad_tuple,parameters,feature_indices,risk_by_RX_F64,role='L_s'):
    """Project only F's -g_L in FP64, then return its negative as native gradients.

    parameters is the full exact optimizer order; feature_indices comes from
    feature_scope. Risk tuples use the same F order and FP64 dtype, with unused
    items None. No optimizer runs here. Non-F gradient objects are unchanged.
    An original None with zero representable projected F gradient stays None;
    an original zero Tensor stays a Tensor. Risk feasibility of -g is distinct
    from feasibility of an AdamW displacement using existing moments/decay.
    """
    if role != 'L_s': raise ValueError('Only visible source L_s risk gradients are permitted')
    if (not isinstance(base_grad_tuple,tuple) or not isinstance(parameters,(list,tuple))
            or not parameters or len(base_grad_tuple) != len(parameters)):
        raise ValueError('Complete base gradient tuple must match actual optimizer parameters')
    for gradient,parameter in zip(base_grad_tuple,parameters):
        if (not isinstance(parameter,torch.Tensor) or not parameter.is_floating_point()
                or parameter.layout != torch.strided or not bool(torch.isfinite(parameter).all())):
            raise ValueError('Finite real dense optimizer parameters required')
        if gradient is not None and (not isinstance(gradient,torch.Tensor) or gradient.layout != torch.strided
                or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                or gradient.device != parameter.device or not bool(torch.isfinite(gradient).all())):
            raise ValueError('Base gradient shape/dtype/device/finite contract differs')
    if (not isinstance(feature_indices,(tuple,list)) or not feature_indices
            or any(type(i) is not int or i<0 or i>=len(parameters) for i in feature_indices)
            or tuple(feature_indices) != tuple(sorted(set(feature_indices)))):
        raise ValueError('Fixed F indices must be unique ascending actual optimizer indices')
    indices=tuple(feature_indices)
    original=tuple(base_grad_tuple[i] for i in indices)
    descent=tuple(torch.zeros_like(parameters[i],dtype=torch.float64) if base_grad_tuple[i] is None
                  else -base_grad_tuple[i].detach().double() for i in indices)
    projected=project(descent,risk_by_RX_F64,role=role)
    ideal=projected['projected_delta']
    output=list(base_grad_tuple);cast_feature=[];none_preserved=introduced=zero_cast_from_nonzero=0
    for position,i in enumerate(indices):
        gradient=-ideal[position].to(dtype=parameters[i].dtype,device=parameters[i].device).detach()
        cast_zero=not bool(torch.count_nonzero(gradient))
        if cast_zero and bool(torch.count_nonzero(ideal[position])): zero_cast_from_nonzero += 1
        if base_grad_tuple[i] is None and cast_zero:
            output[i]=None;none_preserved += 1
        elif base_grad_tuple[i] is not None and torch.equal(gradient,base_grad_tuple[i]):
            output[i]=base_grad_tuple[i]
        else:
            output[i]=gradient;introduced += int(base_grad_tuple[i] is None)
        cast_feature.append(torch.zeros_like(ideal[position]) if output[i] is None else output[i].detach().double())
    result=tuple(output)
    cast_descent=tuple(-gradient for gradient in cast_feature)
    differences=tuple(cast-old for cast,old in zip(cast_descent,descent))
    cast_error=tuple(cast-expected for cast,expected in zip(cast_descent,ideal))
    # project() verified legal exact five keys and FP64 shapes. Normalize only
    # key spelling for reporting the actual returned-gradient risk separately.
    raw={int(rx):values for rx,values in risk_by_RX_F64.items()}
    by_rx={str(rx):dict(original_gradient_descent_directional_risk=_dot(raw[rx],descent),
        projected_FP64_gradient_descent_directional_risk=_dot(raw[rx],ideal),
        cast_gradient_descent_directional_risk=_dot(raw[rx],cast_descent)) for rx in SOURCE_RX}
    summary=dict(feature_tensors=len(indices),feature_coordinates=sum(parameters[i].numel() for i in indices),
        original_feature_gradient_norm=_norm(original),cast_feature_gradient_norm=_norm(cast_feature),
        cast_gradient_difference_norm=_norm(differences),cast_vs_projected_FP64_gradient_error_norm=_norm(cast_error),
        original_feature_None=sum(g is None for g in original),projected_feature_None=sum(output[i] is None for i in indices),
        original_None_zero_projection_preserved=none_preserved,original_None_nonzero_feature_gradient_introduced=introduced,
        ideal_nonzero_cast_zero_tensors=zero_cast_from_nonzero,nonfeature_gradients_original_objects=True,
        by_RX=by_rx,optimizer_updates=0,live_parameter_mutations=0,new_head_gradients=False,
        parameter_scope_must_come_from_feature_scope=True,source_provenance_asserted_by_caller=True,
        boundary='Projected gradient descent feasibility does not imply saved-moment AdamW displacement feasibility; no actual parameter addition is measured')
    return dict(gradients=result,gradient_tuple=result,summary=summary,
                projection_summary=projected['summary'],projected_feature_descent_FP64=ideal)
