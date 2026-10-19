import copy
import torch
import pytest
from experiments.pamf_analytic_restore.views import AnalyticViews, ViewLoss, scales_from_sums
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from experiments.standard_ce_baselines.sigtime import NoExplicitCFO


def scales(front, x):
    values = front(x)
    sums = {key: value.double().square().sum((0, *range(2, value.ndim)), keepdim=False) for key, value in values.items()}
    counts = {key: len(x)*(value.shape[-1] if value.ndim == 3 else 1) for key, value in values.items()}
    return scales_from_sums(sums, counts)


def test_parameterless_views_match_original_PAMF_frontend_exactly():
    torch.manual_seed(51); x = torch.randn(3, 2, 256)*.6
    native = IdentityOnlyCVS().id_backbone.eval(); front = AnalyticViews().eval()
    values = front(x); frequency, rho, dac, pa = native._mirror_compressed_features(x)
    assert list(front.parameters()) == []
    assert torch.equal(values['frequency'], frequency)
    assert torch.equal(values['polynomial'], native.pa_lift(x))
    assert torch.equal(values['statistics'], torch.cat((dac, pa, rho), 1))
    response = NoExplicitCFO('reference_response')(x)
    assert torch.equal(values['response'], torch.cat((response[:, :27], response[:, 28:]), 1))
    assert values['frequency'].shape == (3, 4, 32) and values['polynomial'].shape == (3, 24, 256)
    assert values['statistics'].shape == (3, 7) and values['response'].shape == (3, 28)


def test_view_loss_is_zero_on_identity_and_detaches_clean_targets():
    torch.manual_seed(72); front = AnalyticViews(); clean = torch.randn(4, 2, 256)*.5
    loss_module = ViewLoss(scales(front, clean))
    loss, parts = loss_module(clean, clean)
    assert torch.equal(loss, torch.zeros_like(loss)) and all(torch.equal(value, torch.zeros_like(value)) for value in parts.values())
    target = clean.clone().requires_grad_(); estimate = (clean+.05*torch.randn_like(clean)).requires_grad_()
    value, parts = loss_module(estimate, target); value.mean().backward()
    assert estimate.grad is not None and torch.isfinite(estimate.grad).all() and estimate.grad.norm() > 0
    assert target.grad is None and list(loss_module.parameters()) == []


def test_zero_and_clipped_PA_inputs_remain_finite_without_clipping_IQ_output():
    front = AnalyticViews(); calibration = torch.randn(5, 2, 256)*.5
    module = ViewLoss(scales(front, calibration))
    x = torch.stack((torch.zeros(2, 256), torch.full((2, 256), 3.))).requires_grad_()
    value, _ = module(x, calibration[:2]); value.mean().backward()
    assert torch.isfinite(value).all() and torch.isfinite(x.grad).all()
    assert float(x.detach().max()) == 3.


def test_source_RMS_shapes_floors_and_group_weights_are_fixed():
    front = AnalyticViews(); values = front(torch.zeros(2, 2, 256))
    sums = {key: value.double().square().sum((0, *range(2, value.ndim))) for key, value in values.items()}
    counts = {key: len(value)*(value.shape[-1] if value.ndim == 3 else 1) for key, value in values.items()}
    norm = scales_from_sums(sums, counts)
    assert sum(value.numel() for value in norm.values()) == 63
    assert all(torch.isfinite(value).all() and (value >= .01).all() for value in norm.values())
    bad = dict(norm); bad['statistics'] = torch.zeros_like(norm['statistics'])
    with pytest.raises(ValueError):
        ViewLoss(bad)


def test_calibration_and_checkpoint_cannot_inherit_V_TX_F_or_target_updates():
    from experiments.pamf_analytic_restore.runtime import validate_calibration, validate_payload
    from experiments.pamf_analytic_restore import design as d
    from experiments.pamf_analytic_restore.views import SHAPES
    norm = {key: torch.ones(shape).tolist() for key, shape in SHAPES.items()}
    value = dict(status='ANALYTIC_RMS_FROZEN', count=8192, seed=d.CALIBRATION_SEED, source_V_reads=0,
        TX_labels_read=0, F_access=0, target_reads=0, optimizer_updates=0, fitted_scalars=63, L_exposures=820, U_exposures=7372, scales=norm)
    validate_calibration(value)
    for change in (dict(source_V_reads=1), dict(F_access=1), dict(TX_labels_read=1), dict(target_reads=1), dict(count=8191), dict(optimizer_updates=1)):
        with pytest.raises(ValueError):
            validate_calibration(dict(value, **change))
    # Reject permission mismatches before requiring any weight tensors/lineage.
    for change in (dict(training_F_access=True), dict(training_TX_labels_read=1), dict(target_training=True)):
        payload=dict(schema='analytic_restore_checkpoint_v1', config=d.configuration(),
            candidate='analytic_views', kind='native_map_prior', epoch=20, updates=4940,
            training_F_access=False, training_TX_labels_read=0, target_training=False)
        payload.update(change)
        with pytest.raises(ValueError):
            validate_payload(payload)


def source_summary(accuracy, mid=92., nmse=.028):
    from experiments.pamf_uncertainty_restore.evaluation import VIEWS
    from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
    views = {view: dict(restored=dict(waveform=dict(iq_nmse=nmse))) for view in VIEWS}
    views['clean']['restored']['waveform']['iq_nmse'] = .001
    records = [dict(view=view, readout='restored', dimension='overall', value='all', accuracy=(
        98.5 if view=='clean' else accuracy if view=='practical_low_urban' else mid if view=='practical_mid_urban' else 96.),
        baseline_accuracy=98.5 if view=='clean' else 40.) for view in VIEWS]
    records += [dict(view='practical_low_urban', readout='restored', dimension='TX', value=str(i), accuracy=accuracy) for i in range(6)]
    return dict(status='ANALYZED', kind='native_map_prior', role='source', count=1800, F_run=REFERENCE_RUN, F_seed=2026092701,
        original_default_head=True, all_predictions_before_truth=True, training_TX_labels_read=0, training_F_access=False,
        head_fit_updates=0, target_access=False, views=views, records=records)


def test_source_decision_requires_identity_and_waveform_and_rejects_class_damage():
    from experiments.pamf_analytic_restore.evaluation import qualify_source
    parent=source_summary(82.,mid=90.)
    control=source_summary(82.2,mid=90.2)
    good=source_summary(83.,mid=91.)
    decision=qualify_source(control,good,parent)
    assert decision['qualified'] and decision['target_rows']==['pair_control','analytic_views']
    assert not qualify_source(control,source_summary(83.,mid=90.4),parent)['qualified']
    assert not qualify_source(control,source_summary(83.,mid=91.,nmse=.04),parent)['qualified']
    bad=copy.deepcopy(good);bad['records'][-1]['accuracy']=80.9
    assert not qualify_source(control,bad,parent)['qualified']
    with pytest.raises(ValueError):
        qualify_source(control,dict(good,role='target'),parent)


def test_unqualified_or_wrong_row_is_blocked_before_target_IQ(tmp_path):
    import json
    from types import SimpleNamespace
    from experiments.pamf_analytic_restore.evaluation import prepare_target
    args=SimpleNamespace(output=tmp_path,row='analytic_views')
    (tmp_path/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',qualified=False,target_rows=[])),encoding='utf-8')
    with pytest.raises(ValueError):prepare_target(args)
    (tmp_path/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',qualified=True,target_rows=['pair_control'])),encoding='utf-8')
    with pytest.raises(ValueError):prepare_target(args)
    assert not (tmp_path/'analytic_views').exists()


def test_complete_native_state_roundtrip_and_ancestor_contamination_rejection(tmp_path):
    from experiments.pamf_analytic_restore.runtime import save_checkpoint,load_checkpoint,validate_payload
    from experiments.pamf_analytic_restore import design as d
    from experiments.pamf_uncertainty_restore import design as old
    from experiments.pamf_uncertainty_restore.models import UncertaintyRestorer
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    model=UncertaintyRestorer(d.KIND,ActionRestorer(ComplexTFUNet()))
    parent=dict(run=d.PARENT,commit=d.PARENT_COMMIT,candidate=d.KIND,source_contract='original7/63/30',
        training_TX_F_reads=0,target_training=False,inherited_prior_epochs=20)
    ancestor=dict(schema='uncertainty_checkpoint_v1',config=old.configuration(),kind=d.KIND,
        training_TX_labels_read=0,training_F_access=False,target_training=False,epoch=20,updates=4940,
        parent=dict(coarse_run=old.COARSE_RUN,coarse_commit=old.COARSE_COMMIT,action_run=old.ACTION_RUN,
            action_commit=old.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False))
    optimizer=torch.optim.AdamW(model.prior.parameters(),lr=3e-5)
    path=tmp_path/'final.pt';save_checkpoint(path,model,optimizer,1,1,'pair_control',parent,ancestor,None)
    fresh,payload=load_checkpoint(path,'cpu')
    assert all(torch.equal(value,fresh.state_dict()[key]) for key,value in model.state_dict().items())
    bad=copy.deepcopy(payload);bad['parent_payload']['training_F_access']=True
    with pytest.raises(ValueError):validate_payload(bad)


def test_original_predictor_accepts_new_loader_and_fixes_all_views_before_truth(tmp_path,monkeypatch):
    import json
    import numpy as np
    from types import SimpleNamespace
    from torch import nn
    import experiments.pamf_uncertainty_restore.evaluation as original
    source=tmp_path/'input';source.mkdir()
    for view in original.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32)*.5)
    output=tmp_path/'output';(output/'native_map_prior').mkdir(parents=True)
    args=SimpleNamespace(kind='native_map_prior',output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def features(self,x):return x.flatten(1)[:,:160]
        def classify_features(self,z):return z[:,:6]
    monkeypatch.setattr(original,'reference',lambda args,role:(source,np.arange(16),None,None))
    monkeypatch.setattr(original,'load_reader',lambda *a,**k:Reader())
    monkeypatch.setattr(original,'load_checkpoint',lambda *a,**k:pytest.fail('Old checkpoint schema must not load new continuation'))
    monkeypatch.setattr(original,'array_tensor',lambda value,device:torch.tensor(value.tolist(),dtype=torch.float32))
    monkeypatch.setattr(original,'tensor_array',lambda value:np.asarray(value.detach().tolist(),np.float32))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    calls=[]
    def loader(path,device):
        calls.append(path)
        return Toy(),dict(kind='native_map_prior',epoch=20,updates=4940)
    original.predict(args,'source',checkpoint_loader=loader)
    arrays=original.prediction_arrays(output/'native_map_prior/prediction',np.arange(16),'native_map_prior','source')
    assert list(arrays)==list(original.VIEWS) and len(calls)==1
    complete=json.loads((output/'native_map_prior/prediction/complete.json').read_text())
    assert complete['TX_truth_reads']==complete['state_updates']==complete['head_fit_updates']==0


def test_control_budget_and_public_module_entry_do_not_enable_unused_branches():
    import os,subprocess,sys
    from pathlib import Path
    from experiments.pamf_analytic_restore import design as d
    from experiments.pamf_analytic_restore.training import learning_rate
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','experiments.pamf_analytic_restore.runner','--help'],cwd=root,
        env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    config=d.configuration()
    assert config['updates']==4940 and config['epochs']==20 and config['component_cumulative_epochs']==80
    assert config['training_TX_labels_read']==0 and not config['training_F_access']
    assert abs(learning_rate(4939)-3e-6)<1e-12
    with pytest.raises(ValueError):learning_rate(4940)
