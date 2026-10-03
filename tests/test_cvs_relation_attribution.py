import copy
import numpy as np
import pytest
import torch
from experiments.cvs_relation_attribution.run import components,load_source_state,validate_config,source_identity,PROJECT,RUN
from experiments.cvs_relation_attribution.statistics import analyze_arrays,unit


def build(v):
    if v.startswith('mirror_'):
        from experiments.cvs_mirror_subspace_identity.model import build as factory
    else:
        from experiments.cvs_spectral_relation_identity.model import build as factory
    return factory(v)


@pytest.mark.parametrize('variant',['mirror_energy','mirror_subspace','relation_frequency_energy'])
def test_exact_frozen_counterfactual_and_reconstructed_logits(variant):
    torch.set_num_threads(2);torch.manual_seed(7);model=build(variant).eval()
    branch=getattr(model.core,'mirror_relation',getattr(model.core,'spectral_relation',None))
    with torch.no_grad():branch.project.weight.normal_(0,.01)
    x=torch.randn(24,2,256);old={k:v.clone() for k,v in model.state_dict().items()}
    logits,features=components(model,x)
    assert torch.equal(logits['all_on'],model(x))
    assert not torch.equal(logits['all_on'],logits['auxiliary_off'])
    assert all(torch.equal(old[k],v) for k,v in model.state_dict().items())
    assert not branch._forward_hooks and not model.core.id_backbone.f_proj._forward_hooks
    head=model.core.id_backbone.cls_head
    q=dict(ids=np.array([str(i) for i in range(24)]),truth=np.arange(24)%6,receiver=np.arange(24)%2,day=np.arange(24)%3,
        classifier_weight=head.weight.detach().numpy(),classifier_scale=np.array(30.),
        **{k:v.detach().numpy() for k,v in {**logits,**features}.items()})
    r=analyze_arrays(q,False);assert r['count']==24 and max(r['recomputed_logit_max_errors'].values())<5e-5
    assert sum(v['count'] for v in r['partitions'].values())==24
    for item in r['geometry'].values():assert abs(item['TX']+item['RX_day_within_TX']+item['within_cell']-1)<1e-10
    q['all_on'][0,0]+=1
    with pytest.raises(ValueError):analyze_arrays(q,False)


def config(v='mirror_energy'):
    family,run,commit=source_identity(v);rid=v+'-s2026092701'
    return dict(source_release=PROJECT+'/releases/'+family+'_20261003_r01',source_commit=commit,
        source_output=PROJECT+'/runs/'+run+'/'+rid+'/source',variant=v,model_seed=2026092701,
        output_root=PROJECT+'/runs/'+RUN+'/'+rid,conditions=['all_on','auxiliary_off'],role='V',launch_owner='codex/root/relation-source-attribution-20261003')


@pytest.mark.parametrize('mutation',[None,'target','role','source','output','commit','conditions','owner'])
def test_config_scope_and_exact_paths(mutation):
    c=config()
    if mutation=='target':c['target_truth']='x'
    elif mutation=='role':c['role']='L_s'
    elif mutation=='source':c['source_output']+='/other'
    elif mutation=='output':c['output_root']+='/other'
    elif mutation=='commit':c['source_commit']='wrong'
    elif mutation=='conditions':c['conditions']=['all_on']
    elif mutation=='owner':c['launch_owner']='another'
    if mutation is None:assert validate_config(c)==c
    else:
        with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('mutation',['fp64','hann','nan','contract'])
def test_actual_state_cannot_be_silently_cast_or_misdeclared(mutation):
    model=build('mirror_energy');state={k:v.clone() for k,v in model.state_dict().items()};resolved=dict(mirror_relation_actual=model.contract())
    if mutation=='fp64':state={k:v.double() if v.is_floating_point() else v for k,v in state.items()}
    elif mutation=='hann':state['core.mirror_relation.window'].fill_(1)
    elif mutation=='nan':state['core.mirror_relation.mix_real'][0,0]=float('nan')
    else:resolved['mirror_relation_actual']['mirror_relation_active']=False
    with pytest.raises(ValueError):load_source_state(model,dict(model=state),resolved,'mirror_energy')
