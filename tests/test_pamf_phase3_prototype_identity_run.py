"""Frozen source prototypes route both density and calibration; F0 remains baseline."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.prototype_identity_run')


def test_prototype_is_actual_unit_feature_mean_not_normalized_center_and_batch_independent():
    m=api();x=np.zeros((12,160),np.float32);y=np.repeat(np.arange(6),2)
    for c in range(6):x[2*c,c]=1;x[2*c+1,c]=.6;x[2*c+1,10]=.8
    p=m.fit_prototypes(x,y);np.testing.assert_allclose(p['means'][:,10],.4);np.testing.assert_allclose(p['means'][np.arange(6),np.arange(6)],.8)
    assert np.all(np.linalg.norm(p['means'],axis=1)<1)
    z=x[[8,2,0]];closed,distance=m.classify(z,p);np.testing.assert_array_equal(closed,[4,1,0])
    for i in range(len(z)):
        own,d=m.classify(z[i:i+1],p);np.testing.assert_array_equal(own,closed[i:i+1]);np.testing.assert_array_equal(d,distance[i:i+1])
    with pytest.raises(ValueError):m.fit_prototypes(x,np.zeros(12,np.int64))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as parent
    with parent(root,mp) as (fresh,ref,_,b,events,identity):
        m=api();ref['code_commit']=m.REFERENCE_CODE;ref['train_config']['code_commit']=m.REFERENCE_CODE;identity['code_commit']=m.REFERENCE_CODE;pm=fresh.run(ref,'cpu')
        mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT);mp.setattr(m,'SOURCE_COUNT',fresh.COUNTS['L']);mp.setattr(m.geometry,'SOURCE_COUNT',fresh.COUNTS['L']);mp.setattr(m.geometry,'CELL_COUNT',4)
        model,_=fresh._load_model(ref,'cpu');mp.setattr(m,'_load_model',lambda *a:(model,identity));mp.setattr(m,'_backend',lambda *a:b)
        role=fresh._iq_role(ref,'L',pm['source_control_prediction_identity']);z=fresh.source_eval._project(model,role['restored'],'cpu');meta=role['metadata'];ids=role['ids']
        geo=root/'geometry';geo.mkdir();lc=Path(ref['source_control_config']['reference_config']['train_config']['IQ_root'])/'L';manifest=m._read(lc/'manifest.json')
        gconfig=dict(run_id=m.geometry.RUN_ID,code_commit=m.GEOMETRY_CODE,reference_config=deepcopy(ref),output_root=str(geo),batch_size=128)
        gd=dict(schema=m.geometry.SCHEMA,status=m.geometry.STATUS,run_id=m.geometry.RUN_ID,code_commit=m.GEOMETRY_CODE,config=gconfig,source_L_reads=len(ids),source_only_inputs=True,
            model_source_only=False,PCD_target_U_ancestor_count=576,original_models_unchanged=True,source_states_unchanged=True,private_truth_reads=0,target_U_reads=0,target_Q_reads=0)
        m._write(geo/'complete.json',gd);m._write(geo/'summary.json',dict(source_manifest=manifest,checkpoint_identities=dict(PCD=identity,F0=pm['parent_F0_identity'],r03=b['model_identity'],DURA=pm['dura_identity'])))
        np.savez(geo/'source_features.npz',ids=ids,rx_i=meta['rx_i'],day_i=meta['day_i'],F0_closed=np.zeros(len(ids),np.int64),PCD_unit=z,F0_unit=z,r03_unit=z,PCD_white=z.astype(np.float64))
        c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=deepcopy(ref),source_geometry_root=str(geo),output_root=str(root/'prototype'),batch_size=128)
        events.clear();original=m._parent_rows
        def rows(config,key):
            out=Path(c['output_root']);assert (out/'models/complete.json').exists() and (out/'models/prototypes.npz').exists()
            if key!='source_V':assert (out/'calibration/complete.json').exists() and (out/'thresholds.npz').exists()
            events.append(key);return original(config,key)
        mp.setattr(m,'_parent_rows',rows)
        yield m,c,pm,b,model,events,meta,gd


def test_actual_producer_own_closed_route_exact_controls_all_frozen_then_cpu_gate(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,model,events,meta,gd):
        original=m.risk.calibrate_known;calls=[]
        def calibrate(ids,y,closed,baseline,score,rx,scene):
            with np.load(Path(c['output_root'])/'models/prototypes.npz') as saved:p={k:saved[k] for k in saved.files}
            parent=m.reg._compact(m._parent_rows(c,'source_V'));expected,_=m.classify(parent['features'],p)
            np.testing.assert_array_equal(closed,expected);np.testing.assert_array_equal(baseline,parent['baseline_closed']);calls.append(1)
            return original(ids,y,closed,baseline,score,rx,scene)
        monkeypatch.setattr(m.risk,'calibrate_known',calibrate);done=m.run(c,'cpu');out=Path(c['output_root'])
        assert len(calls)==2 and done['source_prototypes_frozen_before_V'] and done['all_thresholds_frozen_before_novel_and_Q']
        assert done['new_threshold_sets_fitted']==2 and done['source_prototype_fits']==1 and not done['classification_unchanged']
        assert all(done[k]==0 for k in ('target_U_reads','private_truth_reads','real_F_forward_rows','real_signal_forward_rows','G_forward_rows','optimizer_updates','new_density_fits'))
        # Remove the test-only calibration instrument: the CPU gate must not recalibrate.
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No gate calibration'))
        gated,state,p,tau,cal,rows=m._prediction_gate(out);assert gated==done and tau.shape==(4,6) and set(rows)==set(m.ROW_KEYS)
        for key,pair in rows.items():
            old=m._parent_rows(c,key)
            for j,arm in enumerate(m.ARMS):
                a=pair[arm];np.testing.assert_array_equal(a['default_logits'],old['default_logits'])
                np.testing.assert_array_equal(a['closed'],a['default_logits'].argmax(1) if j==0 else a['prototype_distances'].argmin(1))
            np.testing.assert_array_equal(pair['OLD_PCD']['scores'],old['scores'][:,4:6])
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_geometry_weight_identity_or_private_config_rejected_before_V(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,model,events,meta,gd):
        for delta in (dict(truth='private'),dict(unknown_count=4),dict(batch_size=True),dict(output_root=c['source_geometry_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        path=Path(c['source_geometry_root'])/'summary.json';summary=m._read(path);summary['checkpoint_identities']['PCD']=dict(summary['checkpoint_identities']['PCD'],updates=445)
        path.unlink();m._write(path,summary)
        with pytest.raises(ValueError,match='geometry'):m.run(c,'cpu')
        assert events==[] and not Path(c['output_root'],'complete.json').exists()


def test_failed_V_calibration_never_opens_NV_or_Q(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,model,events,meta,gd):
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:(_ for _ in ()).throw(ValueError('calibration failed')))
        with pytest.raises(ValueError,match='calibration failed'):m.run(c,'cpu')
        assert events==['source_V'] and not Path(c['output_root'],'calibration/complete.json').exists() and not Path(c['output_root'],'complete.json').exists()
