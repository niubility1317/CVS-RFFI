"""Fixed real scope plus synthetic gradient/AdamW semantics, no data loading."""
import copy
import json

import torch
from torch import nn

from .gradient_control import feature_scope, gradient_cone
from .projection import SOURCE_RX
from experiments.cvs_u_schedule50.counterfactual import simulate, _equal, _rng_snapshot


def _risks(rows, count):
    result={rx:tuple(None for _ in range(count)) for rx in SOURCE_RX}
    for rx,values in rows.items():
        result[rx]=tuple(None if value is None else torch.tensor([value],dtype=torch.float64) for value in values)
    return result


def _raises(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected scope/shape/nonfinite rejection')


def check():
    from experiments.cvs_crossrx_sharpness20 import design as d, model as model_api
    from .materialize import protected_names
    torch.set_num_threads(2)
    c=d.config(next(row for row in d.rows() if row['arm']=='BASE_CE' and row['model_seed']==d.SEEDS[0]))
    model=model_api.build(c,'cpu').eval()
    named=list(model.named_parameters());parameters=[p for name,p in named];names=[name for name,p in named]
    before=copy.deepcopy(model.state_dict());rng=_rng_snapshot()
    scope=feature_scope(model,parameters,names);protected=protected_names(model)
    assert len(protected)==4 and scope['summary']['protected_tensors_in_optimizer']==4
    assert all(name.startswith('id_backbone.encoder.') and name not in protected for name in scope['names'])
    for required in ('id_backbone.encoder.core.id_backbone.sinc.low_coordinate',
                     'id_backbone.encoder.core.id_backbone.sinc.band_coordinate',
                     'id_backbone.encoder.core.id_backbone.t1.dw.weight',
                     'id_backbone.encoder.core.id_backbone.cls_head.base_norm.weight',
                     'id_backbone.encoder.core.id_backbone.cls_head.pa_norm.weight',
                     'id_backbone.encoder.response_projection.0.weight',
                     'id_backbone.encoder.sequence_pool.local_projection.weight',
                     'id_backbone.encoder.sequence_pool.output_norm.weight'):
        assert required in scope['names']
    assert all(names[i]==name for i,name in zip(scope['indices'],scope['names']))
    assert scope['indices']==feature_scope(model,parameters)['indices']
    assert _equal(before,model.state_dict()) and _equal(rng,_rng_snapshot()) and all(p.grad is None for p in parameters)
    _raises(lambda:feature_scope(model,parameters,names[::-1]))
    _raises(lambda:feature_scope(model,parameters+[parameters[0]]))
    # Independent sign check: downward vector(2,3) projected onto x+y<=0
    # is(-.5,.5); the gradient passed to AdamW must therefore be(.5,-.5).
    small=[nn.Parameter(torch.zeros(1)) for _ in range(4)]
    nonfeature=torch.tensor([.7]);base=(torch.tensor([-2.]),torch.tensor([-3.]),nonfeature,None)
    risk=_risks({1:(1.,1.)},2)
    values=copy.deepcopy([p.detach() for p in small]);base_values=tuple(None if g is None else g.clone() for g in base)
    result=gradient_cone(base,small,(0,1),risk)
    assert result['gradients'] is result['gradient_tuple']
    assert torch.equal(result['gradients'][0],torch.tensor([.5])) and torch.equal(result['gradients'][1],torch.tensor([-.5]))
    assert result['gradients'][2] is nonfeature and result['gradients'][3] is None
    assert result['summary']['by_RX']['1']['cast_gradient_descent_directional_risk']==0.
    assert all(torch.equal(p.detach(),v) and p.grad is None for p,v in zip(small,values))
    assert _equal(base_values,base)
    # An original None remains None if its projected representable gradient is
    # exactly0; a supplied zero Tensor remains a Tensor for AdamW WD/moments.
    none_base=(None,torch.tensor([-1.]),torch.tensor([0.]),None)
    none_result=gradient_cone(none_base,small,(0,1),_risks({1:(None,1.)},2))
    assert none_result['gradients'][0] is None and torch.count_nonzero(none_result['gradients'][1])==0
    assert none_result['gradients'][2] is none_base[2] and none_result['gradients'][3] is None
    # Projection can legitimately introduce a nonzero F gradient into an
    # original-unused coordinate; it never introduces one outside fixed F.
    introduced=gradient_cone(none_base,small,(0,1),risk)
    assert torch.equal(introduced['gradients'][0],torch.tensor([.5]))
    assert introduced['gradients'][2] is none_base[2] and introduced['gradients'][3] is None
    # Concrete saved moments: seven negative-gradient updates precede a
    # positive CE gradient. Although -g satisfies the risk halfspace, AdamW's
    # remaining negative moment gives a positive actual displacement.
    feature=nn.Parameter(torch.zeros(1));head=nn.Parameter(torch.zeros(1))
    optimizer=torch.optim.AdamW([feature,head],lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    for _ in range(7):
        feature.grad=torch.tensor([-1.]);optimizer.step();optimizer.zero_grad(set_to_none=True)
    raw=(torch.tensor([1.]),None);risk1=_risks({1:(1.,)},1)
    cone=gradient_cone(raw,[feature,head],(0,),risk1)
    assert cone['summary']['by_RX']['1']['cast_gradient_descent_directional_risk']==-1.
    parameter_before=feature.detach().clone();state=copy.deepcopy(optimizer.state_dict())
    shadow=simulate(optimizer,raw,cone['gradients'],names=['feature','protected_head'],max_grad_norm=5.)
    actual=shadow['total_after'][0].double()-parameter_before.double()
    actual_risk=float(actual)
    assert actual_risk>0.05
    moments=optimizer.state[feature];t=int(moments['step'])+1
    expected=-.1*float((.9*moments['exp_avg']+.1)/(1-.9**t))/(
        float(torch.sqrt((.999*moments['exp_avg_sq']+.001)/(1-.999**t)))+1e-8)
    assert abs(actual_risk-expected)<1e-7
    assert torch.equal(feature.detach(),parameter_before) and _equal(state,optimizer.state_dict())
    assert feature.grad is None and head.grad is None
    no_gradient=gradient_cone((None,None),[feature,head],(0,),risk1)
    zero_gradient=gradient_cone((torch.zeros_like(feature),None),[feature,head],(0,),risk1)
    semantic=simulate(optimizer,no_gradient['gradients'],zero_gradient['gradients'],
                      names=['feature','protected_head'],max_grad_norm=5.)
    assert torch.equal(semantic['base_after'][0],parameter_before)
    zero_tensor_displacement=float(semantic['total_after'][0].double()-parameter_before.double())
    assert zero_tensor_displacement>0 and _equal(state,optimizer.state_dict())
    json.dumps(result['summary'],allow_nan=False)
    _raises(lambda:gradient_cone(base,small,(0,0),risk))
    _raises(lambda:gradient_cone(base,small,(9,),risk))
    _raises(lambda:gradient_cone(base[:-1],small,(0,1),risk))
    bad=(torch.ones(2),*base[1:]);_raises(lambda:gradient_cone(bad,small,(0,1),risk))
    bad=(torch.tensor([float('nan')]),*base[1:]);_raises(lambda:gradient_cone(bad,small,(0,1),risk))
    wrong_risk=dict(risk);wrong_risk[1]=(torch.tensor([1.]),torch.tensor([1.]))
    _raises(lambda:gradient_cone(base,small,(0,1),wrong_risk))
    return dict(status='PASS',torch=torch.__version__,device='cpu',actual_constructor_noquery=True,
        fixed_feature_tensor_count=len(scope['indices']),fixed_feature_coordinates=scope['summary']['feature_coordinates'],
        W_three_gains_and_other_networks_excluded=True,actual_joint_transformations_included=True,
        negative_gradient_sign_closed_form=True,nonfeature_gradients_original_objects=True,
        None_vs_zero_AdamW_semantics_preserved=True,source_feature_None_can_gain_nonzero=True,
        saved_moment_gradient_descent_risk=-1.,saved_moment_actual_AdamW_directional_risk=actual_risk,
        saved_moment_None_displacement=0.,saved_moment_zero_Tensor_displacement=zero_tensor_displacement,
        gradient_feasibility_not_update_feasibility=True,live_state_unchanged=True,negative_cases=True,
        real_data_reads=0,checkpoint_reads=0,training_launches=0,
        boundary='Synthetic mathematics/constructor only; no source efficacy claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
