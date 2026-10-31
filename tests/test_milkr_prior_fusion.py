import numpy as np
import pytest
from tests.test_milkr_teacher_kernel import examples
from experiments.pamf_phase2_practical.milkr import prior_fusion as model


def rule():return dict(config=['signature_historical','all',0.,'single'],ridge=.1,ridge_reference_classes=6,prior_weight=.5,fusion_policy='rf_distance_margin_half_v1')


def test_prior_residual_fit_matches_augmented_normal_equations():
    x=examples(31,8);y=np.arange(8);state=model.fit(x,y,list(map(str,y)),fixed=rule())
    sim=model.sig.similarity(state.features,state.features,state.config[0]);K=model.base.make_kernel(sim,state.tau,state.config[2],state.config[3])
    target=np.eye(8)[y]-model.prior(x['teacher_probs'],8)
    A=np.block([[K+state.ridge*np.eye(8),np.ones((8,1))],[np.ones((1,8)),np.zeros((1,1))]])
    solution=np.linalg.solve(A,np.concatenate((target,np.zeros((1,8)))))
    assert np.allclose(solution[:-1],state.alpha) and np.allclose(solution[-1],state.intercept)
    query=examples(32,5);scores,energy=state.evaluate(query)
    one=[state.evaluate({key:value[i:i+1] for key,value in query.items()}) for i in range(5)]
    assert np.allclose(scores,np.concatenate([r[0] for r in one])) and np.allclose(energy,np.concatenate([r[1] for r in one]))
    assert np.allclose(scores.sum(1),1)


def test_fusion_uses_G_only_when_distance_and_margin_improve_without_labels():
    raw=np.array([[.7,.3],[.6,.4],[.8,.2],[.6,.4]])
    restored=np.array([[.1,.9],[.1,.9],[.4,.6],[.9,.1]])
    d0=np.array([.4,.2,.4,.4]);dg=np.array([.2,.3,.2,.2])
    scores,weight=model.fuse((raw,d0),(restored,dg))
    assert np.array_equal(weight,[.5,0.,0.,.5])
    assert np.allclose(scores[0],(raw[0]+restored[0])/2) and np.array_equal(scores[1:3],raw[1:3])
    single=[model.fuse((raw[i:i+1],d0[i:i+1]),(restored[i:i+1],dg[i:i+1])) for i in range(4)]
    assert np.array_equal(scores,np.concatenate([r[0] for r in single]))
    assert np.array_equal(weight,np.concatenate([r[1] for r in single]))


def test_prior_head_order_is_mapped_by_legal_class_tables():
    assert model.head_columns(['a','b','c'],['c','a','b'])==[2,0,1]
    with pytest.raises(ValueError,match='class'):model.head_columns(['a','b','c'],['c','a','new'])


def test_prior_three_fold_selection_excludes_held_physical_rows(monkeypatch):
    x=examples(33,24);y=np.repeat(np.arange(8),3);seen=[];original=model.context
    def capture(features,labels,config):seen.append(len(labels));return original(features,labels,config)
    monkeypatch.setattr(model,'context',capture);state=model.fit(x,y,list(map(str,range(24))),fixed=rule())
    assert set(seen)=={16,24} and seen.count(24)==1 and state.selection['physical_OOF']


@pytest.mark.parametrize('method',['prior_fusion','residual_skip','time_orbit','episodic_metric','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble','corr_orbit_ensemble','crossrx_residual_ensemble','adaptive_residual_ensemble'])
def test_worker_emits_aligned_raw_G_and_fusion_rows_with_strict_support(tmp_path,monkeypatch,method):
    import json,types,torch
    from experiments.pamf_phase2_practical import milkr_prior_fusion as worker
    cap=tmp_path/'capsule';(cap/'splits').mkdir(parents=True);(cap/'stage_rows').mkdir()
    names=list('abcdefgh');bank=np.zeros((16,2,256),np.float32);bank[:,0,0]=np.repeat(np.arange(8),2)
    np.save(cap/'received.npy',bank);np.save(cap/'stage_rows/N20.npy',np.arange(16));(cap/'manifest.json').write_text(json.dumps(dict(receiver='rx',capsule_id='testcap')))
    for n,count in ((0,6),(2,8)):
        (cap/'splits'/f'N{n}.json').write_text(json.dumps(dict(k=1,new_count=n,split_id=str(n),registered_classes=names[:count])))
    run=tmp_path/'run';run.mkdir();(run/'source_selection.json').write_text(json.dumps(dict(K1_rule=rule())))
    contract=tmp_path/'source.json';contract.write_text(json.dumps(dict(classes=list('fedcba'))))
    reader=torch.nn.Linear(1,1).eval();gmodel=torch.nn.Linear(1,1).eval();engine=types.SimpleNamespace(pipeline=types.SimpleNamespace(model=gmodel));support=[];mapped=[]
    monkeypatch.setattr(worker,'initialize',lambda rt:(reader,engine))
    monkeypatch.setattr(worker,'load_task',lambda cap,split_id,stage:dict(support_x=bank[np.arange(0,(6 if split_id=='0' else 8)*2,2)],support_y=np.arange(6 if split_id=='0' else 8),support_ids=list(map(str,range(6 if split_id=='0' else 8)))))
    monkeypatch.setattr(worker,'split_rows',lambda cap,s:np.arange(1,len(s['registered_classes'])*2,2))
    def extract(reader,raw,device,engine=None,batch=128):return dict(code=np.asarray(raw)[:,0,0].astype(int),restored=np.full(len(raw),engine is not None)),None
    monkeypatch.setattr(worker,'extract',extract)
    from experiments.pamf_phase2_practical import teacher_features
    monkeypatch.setattr(teacher_features,'extract_time',extract)
    monkeypatch.setattr(teacher_features,'extract_phase8',extract)
    monkeypatch.setattr(teacher_features,'extract_orbit_ensemble',extract)
    if method in ('time_orbit','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble','corr_orbit_ensemble','crossrx_residual_ensemble','adaptive_residual_ensemble'):
        def wrong_extract(*args,**kwargs):raise AssertionError('Time worker used ordinary C4 extraction')
        monkeypatch.setattr(worker,'extract',wrong_extract)
    class Bridge:
        def transform(self,features,*,restored,old_columns):mapped.append(old_columns);return features
    if method in ('episodic_metric','residual_skip','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble','corr_orbit_ensemble','crossrx_residual_ensemble','adaptive_residual_ensemble'):
        monkeypatch.setattr(worker.bridge_type(method),'load',lambda path:Bridge())
        def wrong_bridge(path):raise AssertionError('Metric worker used ordinary CCA encoder')
        monkeypatch.setattr(worker.TeacherBridge,'load',wrong_bridge)
    else:monkeypatch.setattr(worker.TeacherBridge,'load',lambda path:Bridge())
    class State:
        def __init__(self,features,labels):
            support.append(features['restored'].tolist());self.count=len(labels);self.alpha=np.zeros((self.count,self.count));self.config=('signature','all',0.,'single');self.tau=1.;self.ridge=.1;self.selection={}
        def evaluate(self,features):
            code=features['code'];scores=np.zeros((len(code),self.count));rows=np.arange(len(code));restored=bool(features['restored'][0])
            scores[rows,code]=.9 if restored else .4
            if not restored:scores[rows,(code+1)%self.count]=.6
            return scores,np.full(len(code),.1 if restored else .5)
    monkeypatch.setattr(worker,'fit',lambda f,y,ids,fixed:State(f,y))
    if method in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble','corr_orbit_ensemble','crossrx_residual_ensemble'):
        from experiments.pamf_phase2_practical.milkr import orbit_ensemble
        monkeypatch.setattr(orbit_ensemble,'fit',lambda f,y,ids,fixed:State(f,y))
        def wrong_fit(*args,**kwargs):raise AssertionError('Ensemble worker used single-orbit fitter')
        monkeypatch.setattr(worker,'fit',wrong_fit)
    if method=='adaptive_residual_ensemble':
        from experiments.pamf_phase2_practical.milkr import adaptive_residual
        monkeypatch.setattr(adaptive_residual,'fit',lambda f,y,ids,fixed:State(f,y))
        def wrong_fit(*args,**kwargs):raise AssertionError('Adaptive worker used the ordinary fitter')
        monkeypatch.setattr(worker,'fit',wrong_fit)
    rt=dict(run_root=str(run),data_root=str(tmp_path),source_contract=str(contract),K=[1],N=[2],arm='test',model_seeds=[1],decision_batch=3,kernel_method=method)
    worker.worker(rt,'capsule');root=run/'predictions/rx';payload=json.loads((root/'predictions.json').read_text())
    assert len(payload['rows'])==12 and support==[[False]*6,[True]*6,[False]*8,[True]*8]
    assert all(value==[5,4,3,2,1,0] for value in mapped)
    for row in payload['rows']:
        with np.load(root/row['prediction']) as z:
            expected=np.arange(len(row['registered_classes']))
            assert np.array_equal(z['query_indices'],2*expected+1)
            assert np.array_equal(z['predicted_indices'],expected if not row['input_path'].endswith('Q0') else (expected+1)%len(expected))
    complete=json.loads((root/'complete.json').read_text());assert complete['truth_read'] is False and complete['raw_score_cache_final_rows']==0
