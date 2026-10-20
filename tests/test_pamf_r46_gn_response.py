import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r46_gn_response import design as d
from experiments.pamf_r46_gn_response.normalization import PairedMoment,affine_fit,apply_affine,norm_links,collect
from experiments.pamf_r46_gn_response import models as m,runtime as r,evaluation as e,runner
from experiments.pamf_r46_gn_response.calibration import validate_calibration
from experiments.pamf_r46_fold_compensate.calibration import collect as collect_fold
from experiments.pamf_r46_fold_compensate.models import compact_from_parent as full_fold
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)

@pytest.mark.parametrize('strength',[1.,.5])
def test_moment_fit_matches_numpy_identity_ridge_and_folds_existing_affine(strength):
    rng=np.random.default_rng(57);x=rng.normal(size=(4,3,32));y=x*np.array([1.4,.8,2.1])[None,:,None]+np.array([.3,-.1,.2])[None,:,None]
    stat=PairedMoment(3);stat.update(torch.tensor(x),torch.tensor(y))
    item=affine_fit(stat.value(),ridge_fraction=1e-3,bounds=(.25,4.))
    indices=np.arange(16)*31//15;xx=x[:,:,indices].transpose(1,0,2).reshape(3,-1);yy=y[:,:,indices].transpose(1,0,2).reshape(3,-1)
    mx=xx.mean(1);my=yy.mean(1);var=((xx-mx[:,None])**2).mean(1);cov=((xx-mx[:,None])*(yy-my[:,None])).mean(1)
    ridge=1e-3*max(float(var.mean()),1e-12);slope=np.clip((cov+ridge)/(var+ridge),.25,4.);bias=my-slope*mx
    np.testing.assert_allclose(item['slope'],slope,rtol=1e-12);np.testing.assert_allclose(item['intercept'],bias,rtol=1e-12)
    assert item['fitted_MSE']<=item['uncorrected_MSE']
    module=nn.GroupNorm(1,3).double();module.weight.data.copy_(torch.tensor([.8,1.2,.5]));module.bias.data.copy_(torch.tensor([.2,-.4,.3]))
    old=copy.deepcopy(module);z=torch.randn(2,3,17,dtype=torch.float64);before=module(z);apply_affine(module,item,strength)
    expected=before+strength*((torch.tensor(slope)[:,None]-1)*before+torch.tensor(bias)[:,None])
    assert torch.allclose(module(z),expected,atol=1e-12,rtol=1e-12)
    assert set(module.state_dict())==set(old.state_dict()) and sum(p.numel() for p in module.parameters())==6

def test_zero_variance_clipped_slope_and_missing_observations():
    stat=PairedMoment(2)
    with pytest.raises(ValueError):stat.value()
    x=torch.ones(2,2,16,dtype=torch.float64);y=x*torch.tensor([0.,20.],dtype=torch.float64)[None,:,None]
    stat.update(x,y);fit=affine_fit(stat.value(),1e-3,(.25,4.))
    assert fit['slope']==[1.,1.] and fit['fitted_MSE']==pytest.approx(0.)
    stat=PairedMoment(2);x=torch.randn(2,2,16,dtype=torch.float64);stat.update(x,-x*20)
    fit=affine_fit(stat.value(),1e-3,(.25,4.));assert fit['slope']==[.25,.25] and fit['fitted_MSE']<=fit['uncorrected_MSE']

@pytest.mark.parametrize('bad',['nonfinite','mismatch','too_short','bad_ridge','bad_bounds'])
def test_invalid_calibration_values_reject(bad):
    stat=PairedMoment(2);x=torch.ones(2,2,16,dtype=torch.float64);y=x.clone()
    if bad=='nonfinite':x[0,0,0]=float('nan')
    elif bad=='mismatch':y=y[:,:1]
    elif bad=='too_short':x=x[:,:,:8];y=y[:,:,:8]
    if bad in ('bad_ridge','bad_bounds'):
        stat.update(x,y)
        with pytest.raises(ValueError):affine_fit(stat.value(),0. if bad=='bad_ridge' else 1e-3,(2.,4.) if bad=='bad_bounds' else (.25,4.))
    else:
        with pytest.raises(ValueError):stat.update(x,y)

def test_fixed_rule_has_no_F_TX_target_or_architecture_growth():
    cfg=d.configuration();assert d.PARAMETERS=={'gn_full':273765,'gn_half':273765}
    assert d.GN_STRENGTH=={'gn_full':1.,'gn_half':.5} and all(v==1 for v in d.FOLD_STRENGTH.values())
    assert d.REUSED_CONTROL['row']=='fold_full' and d.REUSED_CONTROL['run']==d.FOLD_CAL_RUN
    assert d.CAL_SEED==2026102156 and cfg['calibration']['total_new_G_inputs']==512
    assert all(cfg['calibration'][k]==0 for k in ('TX_labels_read','F_reads','V_reads','target_reads','clean_IQ_reads','optimizer_updates'))

@pytest.fixture(scope='module')
def calibrated():
    torch.manual_seed(57);parent=ParentModel('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in (*parent.prior,parent.frozen.coarse.spectral,parent.frozen.coarse.temporal,parent.frozen.estimator.core):core.out.weight.normal_(std=.001)
        parent.frozen.estimator.global_head.weight.normal_(std=.001)
    iq=torch.randn(128,2,256)*.1;views=[iq,iq*.98];observed=collect_fold(parent,views,batch=8)
    folding=dict(schema='R46_source_channel_reconstruction_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_conv_input=d.CAL_POSITIONS,
        ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
        parent_unchanged=True,no_intercept=True,weight_only_maps=True,calibration_shared_once=True,**{k:v for k,v in observed.items() if k!='moments'})
    child=full_fold(parent,'fold_full',folding).eval().requires_grad_(False)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in child.initialization_report['channel_maps'].items()}
    observed=collect(parent,child,maps,views,batch=8)
    value=dict(schema='R46_GN_response_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        fold_calibration_run=d.FOLD_CAL_RUN,fold_calibration_commit=d.FOLD_CAL_COMMIT,fold_calibration=folding,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_GN_output=16,
        total_new_G_inputs=512,reused_fold_parent_G_inputs=256,GN_ridge_fraction=d.GN_RIDGE_FRACTION,GN_slope_bounds=list(d.GN_SLOPE_BOUNDS),
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,calibration_shared_once=True,GN_affine_has_intercept=True,
        **{k:v for k,v in observed.items() if k!='moments'})
    validate_calibration(value);return parent,child,value

def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

@pytest.mark.parametrize('arm',d.ARMS)
def test_complete_GN_maps_only_affine_changes_joint_gradients_and_standalone_checkpoint(calibrated,tmp_path,arm):
    parent,base,cal=calibrated;before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm,cal)
    assert cal['GN_modules']==sum(isinstance(v,nn.GroupNorm) for v in model.modules())
    assert cal['GN_modules']>0 and all(v['sampled_channel_vectors']==4096 and v['fitted_MSE']<=v['uncorrected_MSE']+1e-8 for v in cal['GN_fits'].values())
    assert all(p.grad is None for p in base.parameters())
    gn_keys={name+'.'+key for name,module in model.named_modules() if isinstance(module,nn.GroupNorm) for key in ('weight','bias')}
    assert all(torch.equal(value,base.state_dict()[key]) for key,value in model.state_dict().items() if key not in gn_keys)
    assert set(model.state_dict())==set(base.state_dict()) and sum(p.numel() for p in model.parameters())==273765
    assert any(not torch.equal(value,base.state_dict()[key]) for key,value in model.state_dict().items() if key in gn_keys)
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():teacher=parent.stage_outputs(received)[0]
    teacher=[v.requires_grad_(True) for v in teacher];student=model.stage_outputs(received)[0]
    loss,terms=m.objective(student,teacher,clean,1,arm);assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse'])
    loss.mean().backward()
    for module in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads) and any(v.abs().sum()>0 for v in grads)
    assert clean.grad is None and all(v.grad is None for v in teacher)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    model.initialization_report['seconds']=0.;model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,lineage());fresh,payload=r.load_checkpoint(path,'cpu')
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    assert payload['optimizer'] is None and not payload['training_F_access']
    a,b=model.prior[:2];assert a.encoder.blocks[1][0].net[0].weight is b.encoder.blocks[1][0].net[0].weight
    assert a.encoder.blocks[1][0].net[1].weight is not b.encoder.blocks[1][0].net[1].weight
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,lineage())

@pytest.mark.parametrize('field,bad',[('TX_labels_read',1),('F_reads',1),('V_reads',1),('target_reads',1),('clean_IQ_reads',1),('source_unique_IQ',127),('student_IQ_inputs',255),('fold_calibration_commit','unknown')])
def test_source_permission_and_actual_calibration_budget_reject(calibrated,field,bad):
    value=copy.deepcopy(calibrated[2]);value[field]=bad
    with pytest.raises(ValueError):validate_calibration(value)

@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('parent_R46_row','retain_refine')])
def test_checkpoint_source_permissions_and_fixed_endpoint_reject(calibrated,field,bad):
    model=m.compact_from_parent(calibrated[0],d.ARMS[0],calibrated[2]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)

@pytest.mark.parametrize('field,bad',[('GN_modules',167),('GN_strength',.7),('new_student_G_inputs',255),('fold_calibration_commit','unknown')])
def test_checkpoint_cannot_change_fixed_GN_rule_or_source_origin(calibrated,field,bad):
    model=m.compact_from_parent(calibrated[0],d.ARMS[0],calibrated[2]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);value['initialization'][field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)

def test_wrong_GN_channel_map_or_nonfinite_fit_rejects(calibrated):
    cal=copy.deepcopy(calibrated[2]);name=next(iter(cal['GN_fits']));cal['GN_fits'][name]['outputs']=cal['GN_fits'][name]['outputs'][::-1]
    with pytest.raises(ValueError):m.compact_from_parent(calibrated[0],d.ARMS[0],cal)
    cal=copy.deepcopy(calibrated[2]);cal['GN_fits'][name]['slope'][0]=float('nan')
    with pytest.raises(ValueError):validate_calibration(cal)

@pytest.mark.parametrize('arm',d.ARMS)
def test_all_fixed_rows_automatically_test_without_source_performance_gate(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'));monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'));monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']

def test_target_truth_waits_for_all_seven_prediction_artifacts(monkeypatch):
    from pathlib import Path
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def absent(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',absent);monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth must remain closed'))
    with pytest.raises(Missing):e.score(args,'target')
