import io
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r46_stat_transport.normalization import StatisticGroupNorm
from experiments.pamf_r46_stat_transport.models import CompactSuccessor
from experiments.pamf_r46_stat_transport import design as d
from experiments.pamf_r46_stat_transport import models as m,runtime as r,evaluation as e,runner
from experiments.pamf_r46_stat_transport.calibration import collect,validate_calibration,aligned_links
from experiments.pamf_r46_fold_compensate.calibration import collect as collect_fold
from experiments.pamf_r46_fold_compensate.models import compact_from_parent as full_fold
from experiments.pamf_r46_stat_transport.normalization import convert_normalization
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)

@pytest.mark.parametrize('shape',[(1,8,32),(3,12,256),(2,28,16)])
def test_identity_matches_original_GN_and_uses_only_each_own_sample(shape):
    torch.manual_seed(58);original=nn.GroupNorm(4,shape[1]).double();model=StatisticGroupNorm(4,shape[1]).double()
    with torch.no_grad():model.weight.copy_(original.weight);model.bias.copy_(original.bias)
    x=torch.randn(shape,dtype=torch.float64)*.7+.3
    assert torch.allclose(original(x),model(x),rtol=1e-12,atol=1e-12)
    independent=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(independent,model(x),rtol=1e-12,atol=1e-12)

def test_calibrated_formula_gradients_and_strict_state_include_all_new_parameters():
    torch.manual_seed(58);model=StatisticGroupNorm(4,12).double()
    mean=dict(slope=[.7,1.2,1.4,.8],intercept=[.1,-.2,.3,-.1]);logvar=dict(slope=[.9,.6,1.2,1.1],intercept=[.2,-.1,.1,.3])
    model.set_calibration(mean,logvar);x=(torch.randn(3,12,32,dtype=torch.float64)*.8+.4).requires_grad_(True)
    z=np.array(x.detach().tolist()).reshape(3,4,-1);mu=z.mean(-1);lv=np.log(z.var(-1)+model.eps)
    center=np.asarray(mean['slope'])*mu+np.asarray(mean['intercept']);lv=np.maximum(np.asarray(logvar['slope'])*lv+np.asarray(logvar['intercept']),np.log(model.eps))
    expected=((z-center[:,:,None])*np.exp(-.5*lv)[:,:,None]).reshape(x.shape)
    np.testing.assert_allclose(np.array(model(x).detach().tolist()),expected,rtol=1e-12,atol=1e-12)
    (model(x)-torch.randn_like(x)).square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(model.stat_parameters.grad).all() and model.stat_parameters.grad.abs().sum()>0
    assert model.stat_parameters.requires_grad and model.stat_parameters.numel()==16
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0);fresh=StatisticGroupNorm(4,12).double();fresh.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    with torch.no_grad():assert torch.equal(model(x),fresh(x))

@pytest.mark.parametrize('bad',['nonfinite','shape','outside'])
def test_wrong_calibration_cannot_enter_layer(bad):
    fit=dict(slope=[1.,1.,1.,1.],intercept=[0.,0.,0.,0.]);other=dict(fit)
    if bad=='nonfinite':fit['slope']=[float('nan'),1.,1.,1.]
    elif bad=='shape':fit['intercept']=[0.]
    else:fit['slope']=[0.,1.,1.,1.]
    with pytest.raises(ValueError):StatisticGroupNorm(4,12).set_calibration(fit,other)

@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_compact_graph_full_cost_counts_and_shared_prior_semantics(arm):
    model=CompactSuccessor(arm)
    assert sum(p.numel() for p in model.parameters())==276453 and sum(p.numel() for p in model.frozen.parameters())==186959
    layers=[v for v in model.modules() if isinstance(v,StatisticGroupNorm)]
    assert len(layers)==168 and sum(v.stat_parameters.numel() for v in layers)==2688 and all(v.stat_parameters.requires_grad for v in layers)
    a,b=model.prior[:2];assert a.encoder.blocks[1][0].net[0].weight is b.encoder.blocks[1][0].net[0].weight
    assert a.encoder.blocks[1][0].net[1].stat_parameters is not b.encoder.blocks[1][0].net[1].stat_parameters
    with torch.no_grad():
        x=torch.zeros(1,2,256);y=model.eval()(x);assert y.shape==x.shape and torch.isfinite(y).all()

def test_fixed_matrix_keeps_original_and_new_parameter_budgets_separate():
    cfg=d.configuration();assert cfg['statistic_parameters']==2688 and cfg['training']['new_stat_parameters_epochs']==20
    assert cfg['training']['original_parameters_cumulative_epochs']==100
    assert cfg['comparison'].startswith('Calibrated versus identity')
    assert all(cfg['calibration'][k]==0 for k in ('TX_labels_read','F_reads','V_reads','target_reads','clean_IQ_reads','optimizer_updates'))
    assert set(cfg['widths'])==set(d.ARMS) and cfg['predecessor_by_GPU']['2']['run']==d.PREDECESSOR_RUN

@pytest.fixture(scope='module')
def calibrated():
    torch.manual_seed(58);parent=ParentModel('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in (*parent.prior,parent.frozen.coarse.spectral,parent.frozen.coarse.temporal,parent.frozen.estimator.core):core.out.weight.normal_(std=.001)
        parent.frozen.estimator.global_head.weight.normal_(std=.001)
    iq=torch.randn(128,2,256)*.1;views=[iq,iq*.98];observed=collect_fold(parent,views,batch=8)
    folding=dict(schema='R46_source_channel_reconstruction_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_conv_input=16,
        ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
        parent_unchanged=True,no_intercept=True,weight_only_maps=True,calibration_shared_once=True,**{k:v for k,v in observed.items() if k!='moments'})
    model=convert_normalization(full_fold(parent,'fold_full',folding)).eval().requires_grad_(False)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    observed=collect(parent,model,maps,views,batch=8)
    cal=dict(schema='R46_GN_stat_transport_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,fold_calibration=folding,source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],
        sampling_seed=d.CAL_SEED,total_new_G_inputs=512,reused_fold_parent_G_inputs=256,ridge_fraction=d.RIDGE_FRACTION,slope_bounds=list(d.SLOPE_BOUNDS),new_stat_parameters=2688,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,calibration_shared_once=True,**{k:v for k,v in observed.items() if k!='moments'})
    validate_calibration(cal);return parent,model,maps,cal

def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

@pytest.mark.parametrize('arm',d.ARMS)
def test_full_chain_source_statistics_only_initialization_joint_gradients_and_strict_reload(calibrated,tmp_path,arm):
    parent,base,_,cal=calibrated;before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm,cal)
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in model.state_dict().items() if not k.endswith('stat_parameters'))
    values=[v for k,v in model.named_parameters() if k.endswith('stat_parameters')]
    assert len(values)==168 and sum(v.numel() for v in values)==2688
    assert (any(torch.count_nonzero(v) for v in values))==(arm=='calibrated')
    assert all(f[k]['sampled_channel_vectors']==256 for f in cal['stat_fits'].values() for k in ('mean','logvariance'))
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():teacher=parent.stage_outputs(received)[0]
    teacher=[v.requires_grad_(True) for v in teacher];student=model.stage_outputs(received)[0];loss,terms=m.objective(student,teacher,clean,1,arm)
    assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse']);loss.mean().backward()
    for module in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads) and any(v.abs().sum()>0 for v in grads)
        stats=[v.grad for k,v in module.named_parameters() if k.endswith('stat_parameters')]
        assert stats and all(v is not None and torch.isfinite(v).all() for v in stats) and any(v.abs().sum()>0 for v in stats)
    assert clean.grad is None and all(v.grad is None for v in teacher) and all(p.grad is None for p in parent.parameters())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    model.initialization_report['seconds']=0.;model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,lineage());fresh,payload=r.load_checkpoint(path,'cpu')
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    assert payload['new_stat_parameters_epochs']==20 and payload['inherited_stat_parameters_epochs']==0 and payload['optimizer'] is None

@pytest.mark.parametrize('field,bad',[('TX_labels_read',1),('F_reads',1),('V_reads',1),('target_reads',1),('clean_IQ_reads',1),('identity_child_G_inputs',255),('new_stat_parameters',2687),('fold_calibration_commit','unknown')])
def test_source_calibration_permissions_and_full_actual_counts_reject(calibrated,field,bad):
    value=copy.deepcopy(calibrated[3]);value[field]=bad
    with pytest.raises(ValueError):validate_calibration(value)

@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('new_stat_parameters_epochs',100),('inherited_stat_parameters_epochs',80)])
def test_checkpoint_permissions_endpoint_and_new_parameter_budget_reject(calibrated,field,bad):
    model=m.compact_from_parent(calibrated[0],d.ARMS[0],calibrated[3]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)

def test_wrong_group_quota_or_nonidentity_calibration_child_rejects(calibrated):
    parent,base,maps,_=calibrated;bad=copy.deepcopy(maps);keep=bad['action'][1];old_quota=parent.frozen.estimator.core.encoder.blocks[1][0].net[1].num_channels//4
    replacement=next(v for v in range(old_quota,2*old_quota) if v not in keep.tolist());keep[0]=replacement;bad['action'][1]=keep.sort().values
    with pytest.raises(ValueError):aligned_links(parent,base,bad)
    changed=copy.deepcopy(base)
    with torch.no_grad():next(v for v in changed.modules() if isinstance(v,StatisticGroupNorm)).stat_parameters[0,0]=.1
    with pytest.raises(ValueError):collect(parent,changed,maps,[torch.randn(1,2,256)])

@pytest.mark.parametrize('arm',d.ARMS)
def test_every_fixed_row_auto_tests_after_source_without_performance_selection(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'));monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'));monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']

def test_target_truth_stays_closed_until_all_declared_predictions_exist(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def absent(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',absent);monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Targettruth must remain closed'))
    with pytest.raises(Missing):e.score(args,'target')
