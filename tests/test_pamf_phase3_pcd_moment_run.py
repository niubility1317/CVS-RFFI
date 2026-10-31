"""Synthetic closed-form moments and frozen-cache producer permissions."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_moment_run')
def core():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_moment_core')


def inputs(n=8):
    from test_pamf_phase3_star_class_competition_core import fixture
    state,_,_=fixture();rng=np.random.default_rng(404)
    x=rng.normal(size=(n,160));x=(x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)
    closed=np.asarray(([0,0,0,0,1,1,2,2]*((n+7)//8))[:n],np.int64)
    logits=np.zeros((n,6),np.float32);logits[np.arange(n),closed]=9
    margins=np.ones((n,2),np.float64)
    return state,x,logits,closed,margins


def test_model_implied_within_moments_are_not_identity_or_total_class_covariance():
    m=core();state,*_=inputs();mom=m.source_moments(state);mu=state['known_means'].reshape(6,15,160);means=mu.mean(1)
    residual=(mu-means[:,None]).reshape(90,160)
    expected=state['known_variances'].mean()*np.eye(160)+residual.T@residual/90
    np.testing.assert_allclose(mom['class_means'],means,atol=0,rtol=0)
    np.testing.assert_allclose(mom['covariance'],expected,atol=1e-15,rtol=0)
    assert not np.allclose(expected,np.eye(160)) and np.max(abs(expected-np.diag(np.diag(expected))))>1e-6
    np.testing.assert_allclose(mom['inverse_sqrt']@expected@mom['inverse_sqrt'],np.eye(160),atol=2e-14,rtol=0)
    np.testing.assert_allclose(mom['sqrt']@mom['sqrt'],expected,atol=2e-14,rtol=0)


def test_population_class_balanced_core_shift_full_covariance_recolor_and_no_mutation():
    m=core();state,x,logits,c,s=inputs();s[3,1]=-1;logits[5,c[5]]=2;before=deepcopy(state);xx=x.copy()
    adapters,d=m.fit_adapters(state,x,logits,c,s);mom=m.source_moments(state)
    mask=np.asarray([True,True,True,False,True,False,True,True]);weights=np.asarray([1/9,1/9,1/9,1/3,1/6,1/6])
    white=(m.density.unit(x)-state['origin'])@state['whitener'];v=(white[mask]-mom['class_means'][c[mask]])@mom['inverse_sqrt']
    nu=weights@v;center=v-nu;cov=center.T@(weights[:,None]*center);ev,vec=np.linalg.eigh(.9*cov+.1*np.eye(160))
    response=np.clip(1/np.sqrt(ev),.5,2);B=(vec*response)@vec.T;A=mom['inverse_sqrt']@B@mom['sqrt'];delta=nu@mom['sqrt']
    assert d['status']=='FITTED' and d['fit_succeeded'] and d['selected_count']==6 and d['selected_class_histogram']==[3,1,2,0,0,0]
    assert d['selected_class_mass']==pytest.approx([1/3]*3+[0]*3) and d['effective_sample_size']==pytest.approx(1/np.square(weights).sum())
    np.testing.assert_array_equal(adapters['SHIFT']['matrix'],np.eye(160))
    np.testing.assert_allclose(adapters['SHIFT']['offset'],np.tile(-delta,(6,1)),atol=2e-13,rtol=0)
    np.testing.assert_allclose(adapters['CORAL']['matrix'],A,atol=2e-13,rtol=0)
    np.testing.assert_allclose(adapters['CORAL']['offset'],mom['class_means']-(mom['class_means']+delta)@A,atol=2e-13,rtol=0)
    assert d['covariance_denominator']=='population_class_balanced_weight_sum_one' and np.all(np.asarray(d['clipped_inverse_sqrt_responses'])<=2)
    order=np.asarray([7,1,4,2,0,6,5,3]);again,_=m.fit_adapters(state,x[order],logits[order],c[order],s[order])
    for arm in m.ARMS:
        for key in ('matrix','offset'):np.testing.assert_allclose(again[arm][key],adapters[arm][key],rtol=0,atol=3e-13)
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    np.testing.assert_array_equal(x,xx)


def test_NO_CORE_is_explicit_identity_and_identity_scores_are_bitexact(subtests):
    m=core();state,x,logits,c,s=inputs()
    for selected in (0,1):
        with subtests.test(selected=selected):
            margins=s.copy();margins[:,1]=-1;margins[:selected,1]=0
            adapters,d=m.fit_adapters(state,x,logits,c,margins)
            assert d['status']=='NO_CORE' and not d['fit_succeeded'] and d['selected_count']==selected
            for arm in m.ARMS:
                np.testing.assert_array_equal(adapters[arm]['matrix'],np.eye(160));np.testing.assert_array_equal(adapters[arm]['offset'],np.zeros((6,160)))
                np.testing.assert_array_equal(m.raw_scores(state,x,c,adapters[arm]),m.original.raw_scores(state,x,c)[:,[1,3]])


def test_transformed_density_oracle_row_local_own_class_and_invalid_boundaries(subtests):
    m=core();state,x,logits,c,s=inputs();adapters,_=m.fit_adapters(state,x,logits,c,s);adapter=adapters['CORAL']
    h=((m.density.unit(x)-state['origin'])@state['whitener'])@adapter['matrix']+adapter['offset'][c]
    bg=m.density._log_density(state,h,'background');pooled=m.density._log_density(state,h,'known');expected=[]
    from test_pamf_phase3_star_class_competition_core import logsum
    for i,label in enumerate(c):
        terms=[]
        for mu,v in zip(state['known_means'][label*15:(label+1)*15],state['known_variances'][label*15:(label+1)*15]):
            terms.append(-np.log(15)-80*np.log(2*np.pi*v)-np.square(h[i]-mu).sum()/(2*v)+float(state['log_abs_det_whitener']))
        expected.append([pooled[i]-bg[i],logsum(terms)-bg[i]])
    actual=m.raw_scores(state,x,c,adapter);np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,x[i:i+1],c[i:i+1],adapter) for i in range(len(x))]),actual,atol=2e-12,rtol=0)
    assert np.max(abs(np.linalg.norm(h,axis=1)-1))>.1
    for bad in ('nan','nonunit','class','logits','dtype','extra'):
        with subtests.test(bad=bad),pytest.raises((ValueError,TypeError)):
            if bad=='nan':a=x.copy();a[0,0]=np.nan;m.fit_adapters(state,a,logits,c,s)
            elif bad=='nonunit':m.fit_adapters(state,x*2,logits,c,s)
            elif bad=='class':m.fit_adapters(state,x,logits,c+6,s)
            elif bad=='logits':m.fit_adapters(state,x,logits[:,::-1].copy(),c,s)
            elif bad=='dtype':m.fit_adapters(state,x,logits,c,s.astype(np.float32))
            else:m.fit_adapters(state,x,logits,c,s,true_TX=np.zeros(len(c)))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as old
    with old(root,mp) as (fresh,r,_,b,events,identity):
        m=api();r['code_commit']=m.REFERENCE_CODE;r['train_config']['code_commit']=m.REFERENCE_CODE;identity['code_commit']=m.REFERENCE_CODE
        pm=fresh.run(r,'cpu');pcd,_=fresh._load_model(r,'cpu');events.clear()
        mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT)
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=deepcopy(r),U_reference_root=str(root/'old_registry'),Q_conditions=deepcopy(r['Q_conditions']),output_root=str(root/'moment'),batch_size=128)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(m,'_load_model',lambda *a:(pcd,identity))
        header=dict(schema=m.U_REFERENCE_SCHEMA,status=m.U_REFERENCE_STATUS,run_id=m.U_REFERENCE_RUN,code_commit=m.U_REFERENCE_CODE,
            reference_prediction_identity=pm,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
            U_manifest=dict(data_status='VALIDATED_ONCE',count=576,capsule_id=r['train_config']['U_capsule_id'],split_id=r['train_config']['U_split_id']))
        uroot=Path(c['U_reference_root']);(uroot/'discovery').mkdir(parents=True);m._write(uroot/'complete.json',header)
        _,states,tau,_=m._reference(c);rng=np.random.default_rng(4);x=rng.normal(size=(576,160));x=(x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)
        closed=np.arange(576,dtype=np.int64)%6;logits=np.zeros((576,6),np.float32);logits[np.arange(576),closed]=10
        margins=fresh._raw(states['PCD'],x,closed)-tau[2:4,closed].T
        np.savez(uroot/'discovery/U.npz',ids=np.asarray(['U'+str(i) for i in range(576)]),features=x,default_logits=logits,closed=closed,base_scores=margins,
                 assignment=np.asarray([object()]*576,dtype=object))
        source,target=fresh._source_rows,fresh._target_rows
        def known(*args):
            assert Path(c['output_root'],'adapters/complete.json').is_file();events.append('source_'+args[-1]);return source(*args)
        def future(*args):
            assert Path(c['output_root'],'adapters/complete.json').is_file();events.append(args[-1]);return target(*args)
        mp.setattr(fresh,'_source_rows',known);mp.setattr(fresh,'_target_rows',future)
        mp.setattr(fresh,'_training_metadata',lambda *a:pytest.fail('No reopening original fresh U cache/IDs'))
        yield m,c,pm,b,identity,events,header


def test_actual_frozen_producer_smoke_U5_freeze_then_all_cached_paths_and_control_exact(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,identity,events,header):
        original=np.lib.npyio.NpzFile.__getitem__;members=[]
        def get(z,key):
            if str(z.zip.filename).replace('\\','/').endswith('/discovery/U.npz'):
                assert key in m.U_FIELDS;members.append(key)
            return original(z,key)
        monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',get)
        for key in ('fit_known','fit_background'):monkeypatch.setattr(m.core.density,key,lambda *a,**k:pytest.fail('No source-statistic fitting'))
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert set(members)==m.U_FIELDS and len(members)==5 and done['target_U_reads']==done['U_ID_metadata_reads']==576
        assert events==['source_known','source_novel','clean_DURA','star_DURA']
        assert done['real_F_forward_rows']==done['real_signal_forward_rows']==done['G_forward_rows']==0
        assert done['source_cached_prediction_rows']==m.COUNTS['V']+m.COUNTS['novel'] and done['target_Q_cached_prediction_rows']==2*m.Q_COUNT
        for role in ('known','novel'):
            actual=m._source_rows(out,role);old=m._select(m.fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role))
            np.testing.assert_array_equal(actual['scores'][:,:4],old['scores']);np.testing.assert_array_equal(actual['scores'][:,4:6],actual['scores'][:,2:4]);np.testing.assert_array_equal(actual['scores'][:,6:8],actual['scores'][:,2:4])
        with np.load(out/'thresholds.npz',allow_pickle=False) as z:
            tau=z['conditional'];np.testing.assert_array_equal(tau[4:6],tau[2:4]);np.testing.assert_array_equal(tau[6:8],tau[2:4])
        with pytest.raises(FileExistsError):m.run(c,'cpu')
        for change in (dict(truth='private'),dict(U_reference_root=c['output_root']),dict(output_root=c['U_reference_root']),dict(Q_conditions={})):
            with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))


def test_wrong_U_ancestry_and_failed_smoke_never_load_U_or_source_query(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,identity,events,header):
        for key in ('code_commit','model_identity','U_manifest'):
            broken=deepcopy(header)
            if key=='code_commit':broken[key]='0'*40
            elif key=='model_identity':broken[key]=dict(identity,arm='SOURCE')
            else:broken[key]['capsule_id']='wrong'
            (Path(c['U_reference_root'])/'complete.json').write_text(json.dumps(broken),encoding='utf-8')
            with subtests.test(key=key),pytest.raises(ValueError):m.run(c,'cpu')
            assert not Path(c['output_root']).exists() and not events
        (Path(c['U_reference_root'])/'complete.json').write_text(json.dumps(header),encoding='utf-8')
        monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke failed')))
        with pytest.raises(ValueError,match='smoke failed'):m.run(c,'cpu')
        assert not Path(c['output_root']).exists() and not events
