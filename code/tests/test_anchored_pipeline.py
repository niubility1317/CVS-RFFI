import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_anchored_cache import toy_rows
from cvsrffi.anchored_pipeline import FrozenAnchoredSystem, validate_config, state_identity
from cvsrffi.anchored_geometry import AnchoredMetricHead
from cvsrffi.anchored_calibration import FinalProbabilityCalibrator


def test_frozen_cache_prediction_roundtrip_and_binding(tmp_path):
    rows=toy_rows();w=torch.randn(3,8)
    system=FrozenAnchoredSystem(None,AnchoredMetricHead(w,10).export_state(),fixed_action=0)
    raw=system.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)
    assert torch.equal(raw['top_class'],rows.baseline_inference_logits.argmax(-1))
    cal=FinalProbabilityCalibrator().fit(raw['log_probabilities'],rows.labels,system_identity=system.identity)
    system.attach_calibrator(cal)
    before=state_identity(system.export_state())
    result=system.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)
    parts=[system.predict_features(rows.h[i:i+1],rows.baseline_inference_logits[i:i+1],rows.quality[i:i+1],rows.valid[i:i+1])['probabilities'] for i in range(len(rows))]
    torch.testing.assert_close(result['probabilities'],torch.cat(parts))
    assert state_identity(system.export_state())==before
    exported=system.export_state()
    assert exported['stage']=='Phase1'
    assert not {'labels','physical_ids','receiver','day','source_cache','dataset_path'}&set(exported)
    loaded=FrozenAnchoredSystem.from_state(exported)
    torch.testing.assert_close(loaded.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)['probabilities'],result['probabilities'])
    with pytest.raises(ValueError,match='identity'):
        system.attach_calibrator(FinalProbabilityCalibrator().fit(raw['log_probabilities'],rows.labels,system_identity='changed'))


def test_config_strict_source_only():
    import json
    path=Path(__file__).resolve().parents[1]/'configs/core90_anchored_geometry_v1.json'
    config=json.loads(path.read_text(encoding='utf-8'))
    validate_config(config)
    with pytest.raises(ValueError):validate_config(dict(config,target=True))
    changed=dict(config);changed['stages']=config['stages']+['target']
    with pytest.raises(ValueError):validate_config(changed)


def test_standalone_and_fused_paths_and_parameter_mutation():
    rows=toy_rows();state=AnchoredMetricHead(torch.randn(3,8),10).export_state()
    expert=FrozenAnchoredSystem(None,state,fixed_action=4,mode='expert')
    result=expert.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)
    torch.testing.assert_close(result['log_probabilities'],expert.head(rows.h).double().log_softmax(-1))
    fusion=FrozenAnchoredSystem(None,state,fixed_action=2)
    assert torch.isfinite(fusion.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)['probabilities']).all()
    fusion.head.a.add_(.01)
    with pytest.raises(ValueError,match='changed'):fusion.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)


def test_actual_iq_cache_fit_calibration_export(tmp_path):
    import json
    from dataclasses import replace
    from test_evidence_integration import model_args
    from post_stage_common import build_baseline_model
    from cvsrffi.anchored_cache import IdentityAnchor,CacheIdentity,build_source_cache
    from cvsrffi.anchored_fit import fit_expert,FitConfig
    from cvsrffi.anchored_calibration import physical_weights
    from cvsrffi.partial_evidence_fit import fit_partial_evidence
    from cvsrffi.mask_pattern_calibration import PatternSpec,PatternCalibrator
    from cvsrffi.anchored_source import FrozenPartialSystem
    torch.set_num_threads(2);torch.manual_seed(95)
    args=model_args();model=build_baseline_model(args,torch.device('cpu'));anchor=IdentityAnchor(model)
    ids=[];labels=[];rx=[]
    for y in range(3):
        for r in (1,3,4):
            for i in range(2):ids.append(f'{y}:{r}:1:{i}');labels.append(y);rx.append(r)
    v_ids=[p+'v' for p in ids]
    contract=dict(roles={'L_s':ids,'V':v_ids},source_receivers=[1,3,4],source_days=[1])
    identity=CacheIdentity('a'*64,'b'*64,'received_iq_v1',{'version':'paired_leo_v1','seed':392005},392005,'L_s')
    batch=dict(x=torch.randn(len(ids),2,256),y=torch.tensor(labels),physical_ids=ids,receiver=torch.tensor(rx),day=torch.ones(len(ids),dtype=torch.long))
    rows=build_source_cache(anchor,[batch],identity,identity.view_recipe,tmp_path/'Ls',contract=contract,batch_size=7)
    vbatch=dict(batch,x=torch.randn(len(ids),2,256),physical_ids=v_ids)
    vid=replace(identity,role='V')
    validation=build_source_cache(anchor,[vbatch],vid,vid.view_recipe,tmp_path/'V',contract=contract,batch_size=7)
    fit=fit_expert(rows,(1,3),FitConfig(epochs=2),3,tmp_path/'fit',w0=anchor.w0,tau0=anchor.tau0,validation_rx=(4,))
    system=FrozenAnchoredSystem(anchor,fit.state,fixed_action=2,architecture=vars(args))
    out=system.predict_features(validation.h,validation.baseline_inference_logits,validation.quality,validation.valid)
    cal=FinalProbabilityCalibrator().fit(out['log_probabilities'],validation.labels,weights=physical_weights(validation.physical_ids),system_identity=system.identity)
    system.attach_calibrator(cal);system.save(tmp_path/'frozen_system.pt')
    restored=FrozenAnchoredSystem.from_state(torch.load(tmp_path/'frozen_system.pt',weights_only=True))
    actual=restored.predict(batch['x'][:3])
    torch.testing.assert_close(actual['probabilities'],system.predict(batch['x'][:3])['probabilities'])
    single=torch.cat([restored.predict(x[None])['probabilities'] for x in batch['x'][:3]])
    torch.testing.assert_close(actual['probabilities'],single,atol=1e-5,rtol=1e-5)
    block_id=replace(identity,preprocessing_version='received_iq_v1:blocks_t_f_pa_fixed_v1')
    blocks=build_source_cache(anchor,[batch],block_id,block_id.view_recipe,tmp_path/'blocks',contract=contract,layout='blocks')
    schema=json.loads((tmp_path/'blocks'/'block_schema.json').read_text())
    spec=PatternSpec(tuple(schema['names']),tuple(schema['sizes']))
    head=fit_partial_evidence(blocks,spec,rank=4)
    assert sum(spec.block_sizes)==blocks.h.shape[1]
    assert head(blocks.h,spec.mask('missing_t',len(blocks)),blocks.quality)['solver_calls']==1
