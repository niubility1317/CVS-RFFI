"""Hierarchical discovery and reuse of opaque, already frozen U feature inputs."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_hier_register_core')
def runner():return importlib.import_module('experiments.pamf_phase3_open_world.star_hier_register_run')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def inputs():
    from test_pamf_phase3_star_signal_register_core import model
    rng=np.random.default_rng(107);n=96;features=rng.normal(scale=.001,size=(n,160))
    for start,scale,axis in ((0,.3,0),(32,1.,1),(64,3.,2)):
        features[start:start+32]*=scale;features[start:start+32,axis]+=1
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    ids=np.asarray([('z' if i<32 else 'b' if i<64 else 'a')+'-'+str(i).zfill(3) for i in range(n)])
    base=np.ones((n,2));base[:16,1]=-1;base[32:49,1]=-1;base[64:,1]=-1
    return model(),features.astype(np.float32),base,ids


def test_actual_hierarchical_clusters_full_U_majority_order_and_population_shrink():
    m=api();state,x,base,ids=inputs();source=dict(schema=np.asarray(m.SOURCE_SCHEMA),eps=np.asarray(1e-6));before=deepcopy((state,x,base,ids,source))
    anonymous,assignment,diag=m.discover(state,source,x,base,ids)
    assert diag['algorithm']=='sklearn.cluster.HDBSCAN' and diag['source_radius_used_for_clustering'] is False
    assert diag['parameters']==dict(min_cluster_size=16,min_samples=8,metric='euclidean',algorithm='brute',n_jobs=1,alpha=1,
        cluster_selection_epsilon=0.,cluster_selection_method='eom',allow_single_cluster=False,copy=True,max_cluster_size=None)
    assert diag['all_U_rows_used'] and diag['registered_count']==2 and len(diag['clusters'])==3
    assert any(c['size']==32 and c['rejected_fraction']==.5 and not c['qualified'] for c in diag['clusters'])
    np.testing.assert_array_equal(assignment['assignments'][:32],-1)
    np.testing.assert_array_equal(assignment['assignments'][32:64],1);np.testing.assert_array_equal(assignment['assignments'][64:],0)
    assert anonymous['anonymous_ids'].tolist()==['anon-0001','anon-0002']
    white=m.density.unit(x)
    for j,rows in enumerate((np.arange(64,96),np.arange(32,64))):
        center=white[rows].mean(0);radial=np.square(white[rows]-center).sum()/(32*160)
        np.testing.assert_allclose(anonymous['means'][j],center,atol=1e-15)
        np.testing.assert_allclose(anonymous['variances'][j],32/192*radial+160/192*state['pooled_L_variance'],atol=1e-15)
    other=dict(source,eps=np.asarray(1e6));again,_,_=m.discover(state,other,x,base,ids)
    for key,value in anonymous.items():np.testing.assert_array_equal(again[key],value)
    for current,original in zip((state,x,base,ids,source),before):
        if isinstance(current,dict):
            for key in current:np.testing.assert_array_equal(current[key],original[key])
        else:np.testing.assert_array_equal(current,original)


def test_input_permutation_free_K_and_unchanged_shared_numeric_API(subtests):
    m=api();state,x,base,ids=inputs();source=dict(schema=np.asarray(m.SOURCE_SCHEMA),eps=np.asarray(.01))
    anonymous,a,_=m.discover(state,source,x,base,ids);order=np.random.default_rng(123).permutation(len(ids))
    again,b,_=m.discover(state,source,x[order],base[order],ids[order])
    for key in anonymous:np.testing.assert_array_equal(anonymous[key],again[key])
    for key in a:np.testing.assert_array_equal(a[key],b[key][np.argsort(order)])
    empty,rows,diag=m.discover(state,source,x,np.ones_like(base),ids)
    assert empty['means'].shape==(0,160) and diag['registered_count']==0 and np.all(rows['assignments']==-1)
    for name in ('validate_source','validate_anonymous','anonymous_ratios','predict','calibrate','source_radius'):
        with subtests.test(alias=name):assert getattr(m,name) is getattr(m.base,name)


@contextmanager
def cached_fixture(root,mp):
    m=runner();state,x,_,_=inputs();ref=root/m.base.RUN_ID/'predict';ref.mkdir(parents=True)
    p=dict(code_commit=m.base.PARENT_CODE,output_root=str(root/'class_source'),reference_prediction_root=str(root/'raw'),
        train_config=dict(IQ_root=str(root/'IQ'),output_root=str(root/'r03'),native_config=dict(registered_classes=['old'+str(i) for i in range(6)])),
        dura_checkpoint=str(root/'G.pt'),dura_release=str(root/'G_code'),Q_conditions=dict(star=dict(public_root=str(root/'Q'))))
    mp.setattr(m.base.parent,'validate_config',lambda value:value)
    c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=p,output_root=str(root/'hier/predict'),batch_size=128,
        U_public_root=str(root/'G10/discovery/public'),U_capsule_id=m.base.U_reader.U_CAPSULE_ID,U_split_id=m.base.U_reader.U_SPLIT_ID,input_reference_root=str(ref))
    old_config={k:v for k,v in c.items() if k!='input_reference_root'};old_config.update(run_id=m.base.RUN_ID,code_commit=m.INPUT_CODE,output_root=str(ref))
    identity=dict(run_id='r03',parent_F0_identity=dict(seed=2026092701),dura_identity=dict(version='dura_v1'))
    manifest=dict(count=m.U_COUNT,capsule_id=c['U_capsule_id'],split_id=c['U_split_id'],registered_classes=p['train_config']['native_config']['registered_classes'])
    done=dict(schema=m.base.SCHEMA,status=m.base.STATUS,run_id=m.base.RUN_ID,code_commit=m.INPUT_CODE,config=old_config,method_names=list(m.METHODS),primary_method=m.PRIMARY_METHOD,
        model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],known_count=m.COUNTS['V'],novel_count=m.COUNTS['novel'],U_count=m.U_COUNT,U_manifest=manifest,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,original_neural_models_source_only=True,neural_updates=0,private_truth_reads=0,U_truth_reads=0,
        original_density_refitted=False,all_three_conditions_complete=True,all_conditions_through_DURA=True,source_radius_frozen_before_U=True,anonymous_frozen_before_V_and_Q=True,
        reference_prediction_identity=dict(config=p,model_identity=identity))
    source_manifest=dict(role='L',count=6300)
    fit=dict(schema=m.base.SCHEMA,status=m.base.MODELS_STATUS,run_id=m.base.RUN_ID,code_commit=m.INPUT_CODE,model_identity=identity,source_L_count=6300,
        source_L_manifest=source_manifest,source_radius_fitted=True,source_radius_frozen_before_U=True,original_density_refitted=False,source_V_fit=False,target_U_reads=0,
        diagnostics=dict(source_count=6300,self_excluded=True))
    reg=dict(schema=m.base.SCHEMA,status=m.base.REGISTRATION_STATUS,run_id=m.base.RUN_ID,code_commit=m.INPUT_CODE,U_count=m.U_COUNT,U_manifest=manifest,
        anonymous_frozen_before_V_and_Q=True,true_U_labels_used=False,true_unknown_count_used=False,query_fit=False,source_prediction_reads=0,U_bank_deployed=False)
    for file,value in (('complete.json',done),('models/complete.json',fit),('registration/complete.json',reg)):
        path=ref/file;path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    np.savez(ref/'models/density.npz',**state);source=dict(schema=np.asarray(m.core.SOURCE_SCHEMA),eps=np.asarray(.001));np.savez(ref/'models/source_radius.npz',**source)
    n=m.U_COUNT;logits=np.tile(np.arange(6,dtype=np.float32),(n,1));features=np.resize(x,(n,160)).copy()
    values=dict(ids=np.asarray(['opaque-'+str(i) for i in range(n)]),default_logits=logits,closed=logits.argmax(1).astype(np.int64),signal_features=features,
        base_scores=np.zeros((n,2)),assignments=np.full(n,-1,np.int64),dbscan_labels=np.full(n,-1,np.int64),rejected=np.zeros(n,bool))
    np.savez(ref/'registration/U.npz',**values)
    b=dict(model_identity=identity,reader_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'])
    yield m,c,state,b,done,fit,values


def test_cached_source_and_U_exact_public_fields_no_native_or_newradius(tmp_path,monkeypatch):
    with cached_fixture(tmp_path,monkeypatch) as (m,c,state,b,done,fit,values):
        assert m.validate_config(c) is c
        monkeypatch.setattr(m.core,'source_radius',lambda *a:pytest.fail('No new source radius fit'))
        source,diag,L_ids,L_manifest,cost=m._source(c,b,b['model_identity'],'cpu',state)
        assert L_ids.shape==(0,) and L_ids.dtype.kind=='U' and L_manifest==fit['source_L_manifest']
        assert cost==dict(source_L_reads=0,source_signal_rows=0,source_radius_new_fits=0) and not diag['source_radius_used_for_clustering']
        rows,manifest=m._U(c,b,'cpu');assert set(rows)=={'ids','default_logits','closed','signal_features'} and manifest==done['U_manifest']
        for key in rows:np.testing.assert_array_equal(rows[key],values[key])


def test_cache_wrong_ancestor_density_hidden_label_and_input_role_rejected(tmp_path,monkeypatch,subtests):
    with cached_fixture(tmp_path,monkeypatch) as (m,c,state,b,done,fit,values):
        ref=Path(c['input_reference_root']);complete=(ref/'complete.json').read_text(encoding='utf-8')
        for defect in ('code','model','source_only','truth','state','hidden','closed'):
            with subtests.test(defect=defect):
                bad=deepcopy(done);v={k:a.copy() for k,a in values.items()};d={k:a.copy() for k,a in state.items()}
                if defect=='code':bad['code_commit']='d'*40
                elif defect=='model':bad['model_identity']['run_id']='foreign'
                elif defect=='source_only':bad['original_neural_models_source_only']=False
                elif defect=='truth':bad['U_truth_reads']=1
                elif defect=='state':d['origin'][0]=.2
                elif defect=='hidden':v['y']=np.zeros(m.U_COUNT,np.int64)
                else:v['closed'][0]=0
                (ref/'complete.json').write_text(json.dumps(bad),encoding='utf-8');np.savez(ref/'models/density.npz',**d);np.savez(ref/'registration/U.npz',**v)
                with pytest.raises(ValueError):
                    if defect in ('hidden','closed'):m._U(c,b,'cpu')
                    else:m._source(c,b,b['model_identity'],'cpu',state)
                (ref/'complete.json').write_text(complete,encoding='utf-8')
        np.savez(ref/'models/density.npz',**state)
        for update in (dict(truth='forbidden'),dict(U_capsule_id='clean'),dict(output_root=str(ref/'nested')),dict(input_reference_root=str(ref.parent/'score'))):
            with subtests.test(update=update):
                with pytest.raises(ValueError):m.validate_config(dict(c,**update))
