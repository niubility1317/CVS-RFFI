"""Synthetic density orchestration: source fit, known calibration, frozen novel."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@contextmanager
def fixture(root, monkeypatch):
    m = importlib.import_module('experiments.pamf_phase3_open_world.star_density_run')
    out = str(root/'out')
    source = dict(run_id=m.RUN_ID, row_id='star-density-s2026092701', code_commit='a'*40,
        model_seed=2026092701, batch_size=128, registered_classes=m.base.native.CLASSES.copy(),
        source_output=str(root/'F0'), source_contract=str(root/'contract'), source_cache=str(root/'unused'),
        base_release=str(root/'release'), frozen_sources={k:str(root/k) for k in ('F','AF','BASE','FOLD','G')}, output_root=out)
    c = dict(run_id=m.RUN_ID, code_commit='a'*40, source_config=source, feature_root=str(root/'features'),
        reference_prediction_root=str(root/'reference'), output_root=out, batch_size=128)
    identity = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701,
        **{k:source[k] for k in ('source_output','source_contract','base_release')})
    dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
    role_data = {}; handles = {}; counts = dict(L=90, OE=30, V=180, novel=90)
    for role in ('L','OE','V','novel'):
        rows = ([(y,r,d) for y in range(6) for r in m.RECEIVERS for d in (1,2,3)] if role=='L' else
            [(0,r,d) for r in m.RECEIVERS for d in (1,2,3) for _ in range(2)] if role=='OE' else
            [(y,r,i%3+1) for y in range(6) for r in m.RECEIVERS for i in range(6)] if role=='V' else
            [(y,r,i%3+1) for y in range(3) for r in m.RECEIVERS for i in range(6)])
        n=len(rows); y=np.array([x[0] for x in rows],np.int64); rx=np.array([x[1] for x in rows],np.int64); day=np.array([x[2] for x in rows],np.int64)
        meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted(m.RECEIVERS,rx)*3+day-1)
        if role in ('L','V'):meta['y']=y
        if role=='V':meta['scenario']=np.tile(np.asarray(m.SCENES),30)
        u=np.zeros((n,160),np.float32);u[np.arange(n),y if role in ('L','V') else 6]=1
        closed=y if role in ('L','V') else np.zeros(n,np.int64)
        logits=np.eye(6,dtype=np.float32)[closed]*3
        handle=dict(role=role,count=n,source_only=True,all_new_IQ_satellite=True,restored_input_path='P1',
            schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',parent_F0_identity=identity,dura_identity=dura)
        handles[role]=handle
        role_data[role]=dict(ids=np.asarray([role+str(i) for i in range(n)]),metadata=meta,
            raw_original_fused160=u*3,default_logits=logits,manifest=None)
        if role in ('V','novel'):
            a=dict(ids=role_data[role]['ids'],rx_i=rx,day_i=day,unit_features=u,default_logits=logits,
                closed=closed.copy(),raw_scores=np.zeros((n,3),np.float64),scores=np.zeros((n,3),np.float64))
            if role=='V':a.update(y=y,scenario=meta['scenario'],baseline_closed=closed.copy())
            Path(c['reference_prediction_root']).mkdir(exist_ok=True)
            np.savez(Path(c['reference_prediction_root'])/('known_predictions.npz' if role=='V' else 'novel_predictions.npz'),**a)
    feature_marker=dict(parent_F0_identity=identity,dura_identity=dura,source_manifests={k:handles[k] for k in ('L','OE')})
    for role in ('L','OE'):role_data[role]['manifest']=feature_marker
    reference=dict(schema=m.REFERENCE_SCHEMA,status=m.REFERENCE_STATUS,run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,
        classification_path='P1',model_identity=identity,dura_identity=dura,cache_identities=handles,
        source_only=True,target_reads=0,source_novel_truth_reads=0,source_U_reads=0,neural_updates=0,
        default_head_replaced=False,model_parameters_buffers_unchanged=True,known_count=180,novel_count=90)
    write(Path(c['reference_prediction_root'])/'complete.json',reference)
    events=[];reader=torch.nn.Linear(1,1).eval().requires_grad_(False)
    backend=dict(reader=reader,dura=None,reader_identity=identity)
    monkeypatch.setattr(m,'COUNTS',counts)
    monkeypatch.setattr(m,'_backend',lambda *a:(backend,{}))
    monkeypatch.setattr(m,'_smoke',lambda *a:events.append('smoke') or dict(status='PASS',actual_F0_checkpoint=True))
    def load(path,role):
        assert events[0]=='smoke';events.append('load_'+role)
        return role_data[role]
    monkeypatch.setattr(m,'load_features',load)
    original_reference=m._reference_data
    def reference_data(path,role):
        if role=='novel':
            assert all((Path(out)/p).is_file() for p in ('known_predictions.npz','thresholds.npz','calibration/complete.json','models/known_only.npz','models/density_ratio.npz'))
        events.append('reference_'+role)
        return original_reference(path,role)
    monkeypatch.setattr(m,'_reference_data',reference_data)
    def known(x,y,rx,day):
        assert events==['smoke','load_L'];events.append('fit_known')
        state=dict(schema=np.asarray('pamf_phase3_star_density_state_v1'),origin=np.zeros(160,np.float64),
            whitener=np.eye(160,dtype=np.float64),log_abs_det_whitener=np.asarray(0.,np.float64),pooled_L_variance=np.asarray(.01,np.float64),
            known_means=np.repeat(np.eye(160,dtype=np.float64)[:6],15,axis=0),known_variances=np.full(90,.01,np.float64),known_weights=np.full(90,1/90,np.float64))
        return state,dict(source_count=len(x))
    def background(state,x,rx,day):
        assert events.index('fit_known')<events.index('load_OE');events.append('fit_background')
        result=deepcopy(state);means=np.zeros((1,160),np.float64);means[0,6]=1
        result.update(background_means=means,background_variances=np.full(1,.01,np.float64),background_weights=np.ones(1,np.float64))
        return result,dict(source_count=len(x))
    monkeypatch.setattr(m,'_fit_known',known);monkeypatch.setattr(m,'_fit_background',background)
    yield m,c,role_data,backend,counts,events,reference


def test_smoke_source_fit_then_both_calibrations_before_novel_preserves_identity(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,roles,backend,counts,events,reference):
        original_load=m.np.load
        class Whitelist:
            def __init__(self,z):self.z=z;self.files=z.files
            def __enter__(self):return self
            def __exit__(self,*a):self.z.close()
            def __getitem__(self,k):
                assert k not in ('scores','raw_scores'), 'Old score/threshold cannot be consumed'
                return self.z[k]
        def load(path,*a,**kw):
            z=original_load(path,*a,**kw)
            return Whitelist(z) if Path(path).parent==Path(c['reference_prediction_root']) else z
        monkeypatch.setattr(m.np,'load',load)
        before={k:v.detach().clone() for k,v in backend['reader'].state_dict().items()}
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert events[:5]==['smoke','load_L','fit_known','load_OE','fit_background']
        assert events[-2:]==['reference_known','reference_novel']
        assert done['real_F_forward_rows']==done['G_forward_rows']==0 and done['synthetic_F_forward_rows']==2
        assert done['identity_predictions_unchanged'] and not done['source_bank_deployed']
        with original_load(out/'known_predictions.npz') as z:
            np.testing.assert_array_equal(z['closed'],roles['V']['metadata']['y'])
            assert z['raw_scores'].shape==(180,2) and z['scores'].shape==(180,2)
        assert all(torch.equal(v,backend['reader'].state_dict()[k]) for k,v in before.items())
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_config_private_wrong_source_or_output_collision_before_backend(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('backend opened'))
        for key in ('novel_truth','target_root','U_root'):
            with subtests.test(key=key),pytest.raises(ValueError):m.run(dict(c,**{key:'forbidden'}),'cpu')
        bad=deepcopy(c);bad['source_config']['row_id']='old-row'
        with pytest.raises(ValueError):m.run(bad,'cpu')
        bad=deepcopy(c);bad['output_root']=c['feature_root'];bad['source_config']['output_root']=bad['output_root']
        with pytest.raises(ValueError):m.run(bad,'cpu')


def test_failed_smoke_cannot_open_source_or_write_complete(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,roles,b,co,e,r):
        monkeypatch.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
        monkeypatch.setattr(m,'load_features',lambda *a:pytest.fail('source opened'))
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert not (Path(c['output_root'])/'complete.json').exists()


def test_wrong_reference_ancestor_and_extra_OE_labels_never_complete(tmp_path,monkeypatch,subtests):
    for defect in ('reference_code','F0','dura','OE_label'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,roles,b,co,e,r):
                if defect=='reference_code':r['code_commit']='b'*40
                elif defect=='F0':r['model_identity']=dict(r['model_identity'],epoch=60)
                elif defect=='dura':r['dura_identity']=dict(r['dura_identity'],updates=1)
                else:roles['OE']['metadata']['y']=np.zeros(co['OE'],np.int64)
                write(Path(c['reference_prediction_root'])/'complete.json',r)
                with pytest.raises(ValueError):m.run(c,'cpu')
                assert not (Path(c['output_root'])/'complete.json').exists()


def test_frozen_density_state_cannot_change_during_known_scoring(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        original=m._density_score
        def changed(state,x,method):
            result=original(state,x,method);state['origin'][0]+=.01;return result
        monkeypatch.setattr(m,'_density_score',changed)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert not (Path(c['output_root'])/'novel_predictions.npz').exists()


def test_reference_identity_and_unit_inputs_cannot_be_silently_reencoded(tmp_path,monkeypatch,subtests):
    for defect in ('argmax','unit','private'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,*_):
                p=Path(c['reference_prediction_root'])/'known_predictions.npz'
                with np.load(p) as z:a={k:z[k] for k in z.files}
                if defect=='argmax':a['closed'][0]=1
                elif defect=='unit':a['unit_features'][0]=0
                else:a['tx']=np.asarray(['private']*len(a['ids']))
                np.savez(p,**a)
                with pytest.raises(ValueError):m.run(c,'cpu')
                assert not (Path(c['output_root'])/'complete.json').exists()
