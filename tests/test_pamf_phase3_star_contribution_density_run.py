"""Synthetic four-contribution extraction and frozen-control source orchestration."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pamf_phase3_star_density_run import fixture as parent_fixture,write
from test_pamf_phase3_star_reference_open_model import fixture as native_fixture,hooks


@contextmanager
def fixture(root,monkeypatch):
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_run')
    prior=root/'prior';prior.mkdir()
    with parent_fixture(prior,monkeypatch) as (old,c,roles,backend,counts,e,ref):
        fm=roles['L']['manifest'];fm['config']=dict(L_root=str(root/'source_cache/L'))
        old.run(c,'cpu');p=Path(c['output_root']);marker=json.loads((p/'complete.json').read_text())
        marker['code_commit']=m.PARENT_CODE;marker['config']['code_commit']=m.PARENT_CODE;marker['config']['source_config']['code_commit']=m.PARENT_CODE
        write(p/'complete.json',marker)
        cal=json.loads((p/'calibration/complete.json').read_text());cal['code_commit']=m.PARENT_CODE;write(p/'calibration/complete.json',cal)
        source=deepcopy(c['source_config']);source.update(run_id=m.RUN_ID,row_id='star-contribution-density-s2026092701',code_commit='b'*40,output_root=str(root/'out'))
        config=dict(run_id=m.RUN_ID,code_commit='b'*40,source_config=source,feature_root=c['feature_root'],reference_prediction_root=str(p),output_root=source['output_root'],batch_size=128)
        for role in ('L','OE'):
            raw=roles[role]['raw_original_fused160'];roles[role]['contributions']=np.stack([raw]*4,axis=1)
        events=[]
        monkeypatch.setattr(m,'COUNTS',counts);monkeypatch.setattr(m,'_backend',lambda *a:(backend,{}))
        monkeypatch.setattr(m,'_smoke',lambda *a:events.append('smoke') or dict(status='PASS',actual_F0_checkpoint=True,contributions_checked=True))
        def load_features(path,role):
            assert events[0]=='smoke';events.append('fit_'+role);return roles[role]
        monkeypatch.setattr(m,'load_features',load_features)
        def load_role(path,role,identity,dura):
            assert role in ('V','novel')
            if role=='novel':assert all((root/'out'/f).exists() for f in ('known_predictions.npz','thresholds.npz','calibration/complete.json','models/contribution640_density_ratio.npz'))
            events.append(role);data=roles[role];meta=data['metadata'];n=len(data['ids'])
            y=meta['y'] if role=='V' else np.full(n,6,np.int64)
            values=np.zeros((n,2,256),np.float32);values[:,0,0]=y
            d=dict(ids=data['ids'],metadata=meta,manifest=marker['reference_prediction_identity']['cache_identities'][role],restored=values,received=None)
            if role=='V':d['baseline_closed']=np.column_stack((meta['y'],meta['y']))
            return d
        monkeypatch.setattr(m,'_load_role',load_role);monkeypatch.setattr(m,'close_role',lambda *a:None)
        def encode(reader,x,device):
            y=x[:,0,0].astype(int);u=np.eye(160,dtype=np.float32)[y];closed=np.where(y<6,y,0)
            return u*4,np.stack([u]*4,axis=1),np.eye(6,dtype=np.float32)[closed]*3
        monkeypatch.setattr(m,'_encode',encode)
        core=m._core();monkeypatch.setattr(core,'KNOWN_CELL_SIZE',1);monkeypatch.setattr(core,'OE_COUNT',30)
        monkeypatch.setattr(core,'CAPACITY',4);monkeypatch.setattr(core,'LLOYD_ITERATIONS',2)
        yield m,config,roles,backend,counts,events,p


def test_native_hooks_match_original_contribution_formula_without_installing_parts_and_cleanup(monkeypatch):
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_run')
    model,x=native_fixture();model.eval().requires_grad_(False);reader=model.id_backbone.encoder
    original={k:v.clone() for k,v in reader.state_dict().items()};initial=hooks(model)
    captured={};head=reader.core.id_backbone.cls_head
    targets=(head.base_norm,head.pa_norm,reader.response_projection,reader.sequence_pool)
    hs=[module.register_forward_hook(lambda mod,inputs,value,k=k:captured.update({k:value.detach().clone()})) for k,module in enumerate(targets)]
    try:
        with torch.no_grad():expected=reader.classify_features(reader.features(x))
    finally:
        for h in hs:h.remove()
    raw,contributions,logits=m._encode(reader,x,'cpu')
    expected_c=torch.stack((captured[0],head.gain.tanh()*captured[1],reader.response_gain.tanh()*captured[2],captured[3]),1)
    np.testing.assert_array_equal(contributions,np.asarray(expected_c.tolist(),np.float32))
    np.testing.assert_array_equal(logits,np.asarray(expected.tolist(),np.float32))
    np.testing.assert_array_equal(raw,np.asarray((((expected_c[:,0]+expected_c[:,1])+expected_c[:,2])+expected_c[:,3]).tolist(),np.float32))
    assert hooks(model)==initial and not hasattr(model,'open_world')
    assert all(torch.equal(value,reader.state_dict()[k]) for k,value in original.items())
    reader.response.unmask=True
    with pytest.raises(ValueError,match='27'):m._encode(reader,x,'cpu')
    assert hooks(model)==initial
    reader.response.unmask=False;reader.fail=True
    with pytest.raises(RuntimeError):m._encode(reader,x,'cpu')
    assert hooks(model)==initial


def test_toy_640_fit_full_producer_keeps_control_exact_and_opens_novel_after_freeze(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,roles,b,counts,events,parent):
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['smoke','fit_L','fit_OE','V','novel']
        assert done['real_F_forward_rows']==270 and done['synthetic_F_forward_rows']==2 and done['G_forward_rows']==0
        assert done['control_reused_exactly'] and done['identity_predictions_unchanged'] and not done['source_bank_deployed']
        for role in ('known','novel'):
            with np.load(parent/(role+'_predictions.npz')) as z:a={k:z[k].copy() for k in z.files}
            with np.load(out/(role+'_predictions.npz')) as z:d={k:z[k].copy() for k in z.files}
            for key in ('ids','closed','default_logits','unit_features'):np.testing.assert_array_equal(d[key],a[key])
            np.testing.assert_array_equal(d['scores'][:,0],a['scores'][:,1]);np.testing.assert_array_equal(d['raw_scores'][:,0],a['raw_scores'][:,1])
            assert d['contribution_features'].shape==(len(d['ids']),640) and d['contribution_features'].dtype==np.float32
        with np.load(parent/'thresholds.npz') as a,np.load(out/'thresholds.npz') as z:np.testing.assert_array_equal(z['conditional'][0],a['conditional'][1])
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_bad_config_ancestry_and_failed_smoke_cannot_complete(tmp_path,monkeypatch,subtests):
    for defect in ('truth','collision','parent_code','source_identity','smoke'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,roles,b,counts,e,parent):
                if defect=='truth':c['truth']='forbidden'
                elif defect=='collision':c['output_root']=c['feature_root'];c['source_config']['output_root']=c['output_root']
                elif defect in ('parent_code','source_identity'):
                    marker=json.loads((parent/'complete.json').read_text())
                    if defect=='parent_code':marker['code_commit']='c'*40
                    else:marker['model_identity']['epoch']=60
                    write(parent/'complete.json',marker)
                else:mp.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
                with pytest.raises(ValueError):m.run(c,'cpu')
                assert not (Path(c['output_root'])/'complete.json').exists()


def test_new_defaulthead_or_domain_mismatch_never_silently_replaces_frozen_control(tmp_path,monkeypatch,subtests):
    for defect in ('logits','metadata'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,roles,b,counts,e,parent):
                if defect=='logits':
                    original=m._encode
                    def changed(*a):
                        f,v,z=original(*a);z[:,0]+=.0001;return f,v,z
                    mp.setattr(m,'_encode',changed)
                else:roles['V']['metadata']['day_i']=roles['V']['metadata']['day_i'][::-1].copy()
                with pytest.raises(ValueError):m.run(c,'cpu')
                assert 'novel' not in e and not (Path(c['output_root'])/'complete.json').exists()
