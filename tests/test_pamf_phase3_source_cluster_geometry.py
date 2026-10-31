"""Source-only label-blind discovery, future-independent predictions and screening."""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.source_cluster_geometry')


def test_source_support_radius_cluster_boundary_and_query_batch_independence(monkeypatch):
    m=api();x=np.zeros((12,160));x[:,0]=np.repeat(np.arange(6)*10,2);x[:,1]=np.tile([-1.,1.],6);y=np.repeat(np.arange(6),2)
    support=m.fit_support(x,y);np.testing.assert_allclose(support['source_radius'],1.)
    early=np.zeros((4,160));early[:,0]=[10,10,30,30];early[:,1]=[-.5,.5,-.25,.25];ids=np.asarray(['z','y','b','a'])
    monkeypatch.setattr(m,'_cluster',lambda a:np.asarray([7,7,2,2],np.int64))
    state,assignment,diag=m.discover(early,ids,support)
    np.testing.assert_array_equal(assignment,[1,1,0,0]);np.testing.assert_allclose(state['radii'],[.25,.5])
    q=early[[2,0]].copy();q[0,1]=.25;q[1,1]=.500001
    result=m.predict(q,state);np.testing.assert_array_equal(result['cluster_prediction'],[0,-1])
    one=[m.predict(row[None],state) for row in q]
    for key in result:np.testing.assert_array_equal(result[key],np.concatenate([a[key] for a in one]))
    assert all('y' not in inspect.signature(f).parameters for f in (m.discover,m.predict))
    before=deepcopy(state);m.predict(q,state)
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    with pytest.raises(ValueError):m.fit_support(x,y*0)


def test_strict_majority_matching_keeps_all_six_denominators_and_missing_NA():
    m=api();y=np.repeat(np.arange(6),4);a=np.repeat(np.arange(6),4)
    # Cluster zero is exactly tied between class 0 and class 1: no eligible mapping.
    a[4:8]=0;future=np.arange(6);raw=np.asarray([0,0,2,3,4,-1]);baseline=future.copy()
    row=m.evaluate_case(y,a,future,raw,baseline,np.asarray([2,2,2,3,3,3]),future.copy())
    assert row['early']['matched_class_count']==4 and row['future']['macro_correct_acceptance']==pytest.approx(3/6)
    assert row['future']['macro_acceptance']==pytest.approx(5/6) and row['future']['unmatched_accepted_count']==2
    assert row['future']['by_day']['2']['macro_correct_acceptance'] is None
    assert not row['future']['by_day']['2']['coverage_complete'] and not row['source_screen_passed']
    assert row['A_N'] is None and row['R_U'] is None and row['H_OW'] is None
    perfect=m.evaluate_case(y,np.repeat(np.arange(6),4),future,future,baseline,np.full(6,2),future)
    assert perfect['source_screen_passed']
    # A failed receiver cannot be hidden by the other four receivers.
    cases={str(r):perfect for r in m.RECEIVERS};cases['8']=row
    assert not m.screen(cases)
    assert not m.screen({})


def test_actual_hdbscan_stable_opaque_ids_empty_clusters_and_nonfinite(monkeypatch,subtests):
    m=api();rng=np.random.default_rng(17);x=np.zeros((96,160));x[:48,0]=-4;x[48:,0]=4;x+=rng.normal(0,.01,x.shape)
    support=m.fit_support(np.eye(160)[:6],np.arange(6));ids=np.asarray([f'opaque-{j:03}' for j in range(len(x))]);state,a,diag=m.discover(x,ids,support)
    assert len(state['centers'])==2 and diag['algorithm']=='sklearn.cluster.HDBSCAN'
    order=rng.permutation(len(x));other,b,_=m.discover(x[order],ids[order],support)
    np.testing.assert_array_equal(a[order],b)
    for key in state:np.testing.assert_array_equal(state[key],other[key])
    empty,assignment,_=m.discover(np.empty((0,160)),np.asarray([],dtype='U'),support)
    result=m.predict(x[:2],empty);np.testing.assert_array_equal(result['cluster_prediction'],[-1,-1]);assert len(assignment)==0
    for bad in (np.full((2,160),np.nan),np.zeros((2,159))):
        with subtests.test(shape=bad.shape),pytest.raises(ValueError):m.predict(bad,state)


class Reader(torch.nn.Module):
    def __init__(self):super().__init__();self.register_buffer('scale',torch.tensor(30.))
    def features(self,x):return x[:,0,:160]
    def classify_features(self,x):return self.scale*torch.nn.functional.normalize(x,dim=1)[:,:6]


class Signal(torch.nn.Module):
    def __init__(self,offset):super().__init__();self.bias=torch.nn.Parameter(torch.full((160,),offset,dtype=torch.float32))
    def forward(self,x):return torch.nn.functional.normalize(x[:,0,:160]+self.bias,dim=1)


def fixture(tmp_path,mp):
    m=api();mp.setattr(m,'CELL_COUNT',4);mp.setattr(m,'SOURCE_COUNT',360)
    ref=dict(code_commit=m.REFERENCE_CODE,output_root=str(tmp_path/'old_predict'),train_config=dict(output_root=str(tmp_path/'old_train'),U_public_root=str(tmp_path/'U')),
        source_control_config=dict(output_root=str(tmp_path/'body_predict'),reference_config=dict(train_config=dict(IQ_root=str(tmp_path/'IQ')))),Q_conditions={})
    mp.setattr(m.reg.fresh,'validate_config',lambda c:c)
    c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=ref,output_root=str(tmp_path/'diagnostic'),batch_size=128)
    identity=dict(source_only=False,target_U_reads=576,model='final_PCD');pm=dict(model_identities={'TARGET_PCD':identity},parent_F0_identity={'model':'F0'},loaded_dura_identity={'model':'G'},dura_identity={'model':'G'},source_control_prediction_identity={'old':'metadata'})
    state=dict(schema=np.asarray(m.reg.core.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160)*2,log_abs_det_whitener=np.asarray(160*np.log(2)),pooled_L_variance=np.asarray(1.),
        known_means=np.zeros((90,160)),known_variances=np.ones(90),known_weights=np.full(90,1/90),background_means=np.zeros((1,160)),background_variances=np.ones(1),background_weights=np.ones(1))
    reader=Reader().eval().requires_grad_(False);original=Signal(0).eval().requires_grad_(False);pcd=Signal(.003).eval().requires_grad_(False);g=torch.nn.Identity().eval()
    b=dict(reader=reader,model=original,dura=(g,lambda x:(x,)),reader_identity=pm['parent_F0_identity'],model_identity={'model':'source_r03'},dura_identity=pm['loaded_dura_identity'])
    mp.setattr(m.reg,'_reference',lambda *a:(pm,state,np.zeros((2,6)),{}));mp.setattr(m.reg,'_backend',lambda *a:b);mp.setattr(m.reg,'_load_model',lambda *a:(pcd,identity))
    grid=np.asarray([(y,r,d) for y in range(6) for r in m.RECEIVERS for d in (1,2,3) for _ in range(4)]);y,rx,day=grid.T;rng=np.random.default_rng(2)
    iq=rng.normal(0,.02,(360,2,256)).astype(np.float32);iq[np.arange(360),0,y]+=2
    role=dict(ids=np.asarray([f'source-{i:04}' for i in range(360)]),restored=iq,metadata=dict(y=y,rx_i=rx,day_i=day,domain=np.asarray([m.RECEIVERS.index(r)*3+d-1 for r,d in zip(rx,day)])),manifest=dict(role='L',count=360,restored_input_path='P1'))
    events=[]
    def opened(config,name,header):
        assert name=='L' and Path(c['output_root'],'smoke.json').exists();events.append('L');return role
    mp.setattr(m.reg.fresh,'_iq_role',opened);mp.setattr(m.reg.fresh.source_eval,'_close_role',lambda *a:None)
    mp.setattr(m,'_cluster',lambda x:np.zeros(len(x),np.int64))
    return m,c,role,events


def test_actual_samepass_encoding_all20_files_before_scoring_and_source_only(tmp_path,monkeypatch):
    m,c,role,events=fixture(tmp_path,monkeypatch);evaluate=m.evaluate_case;calls=[]
    def after_all(*args):
        assert len(list(Path(c['output_root'],'cases').glob('*/RX*/predictions.npz')))==20
        calls.append(1);return evaluate(*args)
    monkeypatch.setattr(m,'evaluate_case',after_all)
    done=m.run(c,'cpu');out=Path(c['output_root']);summary=m._read(out/'summary.json')
    assert events==['L'] and len(calls)==20 and done['case_count']==done['cluster_fits']==20
    assert done['real_F_forward_rows']==360 and done['real_signal_forward_rows']==720 and done['G_forward_rows']==0
    assert not done['model_source_only'] and done['source_only_inputs'] and done['all_predictions_frozen_before_evaluation']
    assert all(done[k]==0 for k in ('source_V_reads','source_NV_reads','source_OE_reads','target_U_reads','target_Q_reads','private_truth_reads','optimizer_updates','new_GMM_fits','new_threshold_fits'))
    with np.load(out/'source_features.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['F0_closed'],role['metadata']['y'])
        np.testing.assert_allclose(saved['PCD_white'],2*saved['PCD_unit'].astype(np.float64),atol=1e-6)
    assert set(summary['representations'])==set(m.REPRESENTATIONS) and all(a['A_N'] is None for a in summary['representations'].values())
    with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_config_roles_and_failed_discovery_never_complete_or_evaluate(tmp_path,monkeypatch,subtests):
    m,c,role,events=fixture(tmp_path,monkeypatch)
    for delta in (dict(truth='forbidden'),dict(batch_size=True),dict(reference_config=dict(c['reference_config'],code_commit='b'*40)),dict(output_root=c['reference_config']['train_config']['output_root'])):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
    monkeypatch.setattr(m,'evaluate_case',lambda *a:pytest.fail('No partial evaluation'))
    monkeypatch.setattr(m,'_cluster',lambda *a:(_ for _ in ()).throw(ValueError('clustering failed')))
    with pytest.raises(ValueError,match='clustering failed'):m.run(c,'cpu')
    assert not Path(c['output_root'],'complete.json').exists() and not Path(c['output_root'],'summary.json').exists()
