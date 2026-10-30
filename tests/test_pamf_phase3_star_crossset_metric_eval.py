"""Synthetic strict checkpoint, feature-only density evaluation and immutable F0 control."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_actual_synthetic_projection_checkpoint_and_Torch_numpy_safe_bridge(tmp_path,monkeypatch):
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_crossset_metric_eval')
    from experiments.pamf_phase3_open_world.star_crossset_metric_model import MetricResidual
    model=MetricResidual().eval().requires_grad_(False);p=tmp_path/'synthetic_metric.pt'
    torch.save(model.state_dict(),p);loaded=MetricResidual().eval().requires_grad_(False)
    loaded.load_state_dict(torch.load(p,map_location='cpu',weights_only=False),strict=True)
    before=m._snapshot(loaded);rng=np.random.default_rng(71)
    u=rng.normal(size=(3,160)).astype(np.float32);c=rng.normal(size=(3,640)).astype(np.float32)
    original=torch.as_tensor
    def no_numpy(value,*a,**k):
        if isinstance(value,np.ndarray):raise RuntimeError('NumPy bridge unavailable')
        return original(value,*a,**k)
    monkeypatch.setattr(torch,'as_tensor',no_numpy);monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy ABI bridge forbidden'))
    z=m._project(loaded,u,c,'cpu');assert z.shape==(3,160) and z.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
    assert m._smoke(loaded,'cpu')['synthetic_open_forward_rows']==2
    m._unchanged(loaded,before)


@contextmanager
def fixture(root,monkeypatch,arm='S_SEP_TOPO'):
    from test_pamf_phase3_star_crossset_metric_train import fixture as train_fixture,frozen_checkpoint
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_crossset_metric_eval')
    with train_fixture(root,monkeypatch) as (t,prepare,configs,manifest):
        tc=deepcopy(configs[arm]);frozen_checkpoint(tc,manifest)
        L=t.load_role(tc['cache_root'],'L');OE=t.load_role(tc['cache_root'],'OE');core=m._core()
        counts=dict(L=len(L['ids']),OE=len(OE['ids']),V=180,novel=90)
        monkeypatch.setattr(m,'COUNTS',counts);monkeypatch.setattr(m.reference.parent,'COUNTS',counts)
        monkeypatch.setattr(core,'KNOWN_CELL_SIZE',counts['L']//90);monkeypatch.setattr(core,'OE_COUNT',counts['OE'])
        monkeypatch.setattr(core,'CAPACITY',4);monkeypatch.setattr(core,'LLOYD_ITERATIONS',2)
        known,_=core.fit_known(L['u'],L['metadata']['y'],L['metadata']['rx_i'],L['metadata']['day_i'])
        state,_=core.fit_background(known,OE['u'],OE['metadata']['rx_i'],OE['metadata']['day_i'])
        parent=root/'reference';(parent/'models').mkdir(parents=True);np.savez(parent/'models/fused160_density_ratio.npz',**state)
        scenarios=m.reference.SCENES;receivers=m.reference.RECEIVERS;rows=[(c,r,s) for c in range(6) for r in receivers for s in scenarios]
        y=np.asarray([v[0] for v in rows],np.int64);rx=np.asarray([v[1] for v in rows],np.int64);day=np.arange(180,dtype=np.int64)%3+1
        select=np.asarray([np.flatnonzero(L['metadata']['y']==c)[i%np.sum(L['metadata']['y']==c)] for i,c in enumerate(y)])
        v=dict(ids=np.asarray(['V'+str(i) for i in range(180)]),y=y,rx_i=rx,day_i=day,scenario=np.asarray([v[2] for v in rows]),
            unit_features=L['u'][select].copy(),contribution_features=L['c'][select].copy(),default_logits=np.eye(6,dtype=np.float32)[y]*3,closed=y.copy(),baseline_closed=y.copy())
        select=np.arange(90)%counts['OE'];rows=[(c,r,s) for c in range(3) for r in receivers for s in scenarios]
        n=dict(ids=np.asarray(['novel'+str(i) for i in range(90)]),rx_i=np.asarray([v[1] for v in rows],np.int64),day_i=np.arange(90,dtype=np.int64)%3+1,
            unit_features=OE['u'][select].copy(),contribution_features=OE['c'][select].copy(),default_logits=np.tile(np.asarray([3,0,0,0,0,0],np.float32),(90,1)),closed=np.zeros(90,np.int64))
        raw=core.score(state,v['unit_features'],'density_ratio');cal,report=m.risk.calibrate_known(v['ids'],y,y,y,raw,rx,v['scenario'])
        tau=np.stack((cal['conditional_thresholds'],cal['conditional_thresholds']+123.));glob=np.asarray([cal['global_threshold'],cal['global_threshold']+123.],np.float64)
        np.savez(parent/'thresholds.npz',conditional=tau,**{'global':glob})
        for role,a in (('known',v),('novel',n)):
            r=core.score(state,a['unit_features'],'density_ratio');a['raw_scores']=np.column_stack((r,r+999.));a['scores']=a['raw_scores']-tau[:,a['closed']].T
            np.savez(parent/(role+'_predictions.npz'),**a)
        fm=manifest['source_feature_manifest']
        marker=dict(run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,schema=m.reference.SCHEMA,status=m.reference.STATUS,
            model_identity=manifest['parent_F0_identity'],dura_identity=manifest['dura_identity'],source_feature_manifest=fm,
            method_names=list(m.reference.METHODS),classification_path='P1',source_only=True,private_truth_reads=0,target_reads=0,
            identity_predictions_unchanged=True,control_reused_exactly=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True)
        write(parent/'complete.json',marker)
        pc=dict(run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,status=m.reference.CALIBRATION_STATUS,
            methods={m.METHODS[0]:report['methods']['predicted_class']},global_controls={m.METHODS[0]:report['methods']['global_control']})
        write(parent/'calibration/complete.json',pc)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],arm=arm,train_config=tc,cache_root=tc['cache_root'],reference_prediction_root=str(parent),output_root=str(root/'evaluation'),batch_size=128)
        events=[];original_load=m._load_role;original_smoke=m._smoke;original_rows=m._reference_rows
        def load(path,role):
            assert events[0]=='smoke';events.append(role);return original_load(path,role)
        def smoke(model,device):events.append('smoke');return original_smoke(model,device)
        def saved_rows(path,role):
            if role=='known':assert (Path(c['output_root'])/'models/complete.json').exists()
            else:assert all((Path(c['output_root'])/p).exists() for p in ('known_predictions.npz','thresholds.npz','calibration/complete.json'))
            events.append(role);return original_rows(path,role)
        monkeypatch.setattr(m,'_load_role',load);monkeypatch.setattr(m,'_smoke',smoke);monkeypatch.setattr(m,'_reference_rows',saved_rows)
        yield m,c,counts,events,n


def test_strict_actual_toy_checkpoint_to_eval_without_F0_or_G_and_threshold_before_novel(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,novel):
        done=m.evaluate(c,'cpu');out=Path(c['output_root'])
        assert events==['smoke','L','OE','known','novel']
        assert done['real_F_forward_rows']==done['synthetic_F_forward_rows']==done['G_forward_rows']==0
        assert done['synthetic_open_forward_rows']==2 and done['open_projection_rows']==sum(counts.values())
        assert done['model_parameters_buffers_unchanged'] and done['control_reused_exactly'] and not done['source_bank_deployed']
        assert done['learned_known_statistics_shared'] and done['same_projection_for_learned_methods']
        with np.load(out/'models/learned_known_support.npz') as z:k={n:z[n].copy() for n in z.files}
        with np.load(out/'models/learned_density_ratio.npz') as z:r={n:z[n].copy() for n in z.files}
        assert len(k)==8 and len(r)==11
        assert all(np.array_equal(v,r[n]) for n,v in k.items())
        with np.load(out/'thresholds.npz') as z:assert z['conditional'].shape==(3,6) and z['global'].shape==(3,)
        for role in ('known','novel'):
            with np.load(Path(c['reference_prediction_root'])/(role+'_predictions.npz')) as z:old={k:z[k].copy() for k in z.files}
            with np.load(out/(role+'_predictions.npz')) as z:a={k:z[k].copy() for k in z.files}
            for k in old:
                np.testing.assert_array_equal(a[k][:,0] if k in ('scores','raw_scores') else a[k],old[k][:,0] if k in ('scores','raw_scores') else old[k])
            assert a['learned_features'].shape==(len(a['ids']),160)
            assert a['scores'].shape==(len(a['ids']),3) and a['raw_scores'].shape==(len(a['ids']),3)
        with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_invalid_config_checkpoint_identity_or_failed_smoke_precedes_source_reads(tmp_path,monkeypatch,subtests):
    for defect in ('truth','overlap','identity','smoke'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,counts,events,n):
                original_output=Path(c['output_root'])/'complete.json'
                cache_complete=Path(c['cache_root'])/'complete.json';cache_before=cache_complete.read_bytes()
                if defect=='truth':c['novel_truth']='forbidden'
                elif defect=='overlap':c['output_root']=c['cache_root']
                elif defect=='identity':
                    original=m._backend
                    def bad(*a):
                        model,i=original(*a);i=dict(i,updates=11100);return model,i
                    mp.setattr(m,'_backend',bad)
                else:mp.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
                with pytest.raises(ValueError):m.evaluate(c,'cpu')
                assert not original_output.exists() and 'L' not in events
                assert cache_complete.read_bytes()==cache_before


def test_S_SEP_embedding_control_still_uses_OE_density_and_model_cannot_mutate(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch,arm='S_SEP') as (m,c,counts,events,n):
        result=m.evaluate(c,'cpu')
        assert result['embedding_training_uses_OE'] and not result['OE_topology_regularization'] and result['density_background_uses_OE'] and 'OE' in events
    root=tmp_path/'changed';root.mkdir()
    with monkeypatch.context() as mp,fixture(root,mp) as (m,c,counts,events,n):
        original=m._project
        def changed(model,*a):
            result=original(model,*a)
            if len(result)>2:
                with torch.no_grad():next(model.parameters()).add_(.01)
            return result
        mp.setattr(m,'_project',changed)
        with pytest.raises(ValueError,match='changed'):m.evaluate(c,'cpu')
        assert 'novel' not in events and not (Path(c['output_root'])/'complete.json').exists()
