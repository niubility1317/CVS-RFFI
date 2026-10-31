import copy
import numpy as np
import pytest
from experiments.pamf_phase3_restoration import cache as m


@pytest.fixture
def bundle(monkeypatch):
    manifest=dict(schema=m.SCHEMA,status='VALIDATED_ONCE',capsule_id=m.CAPSULE,split_id=m.SPLIT,
        count=m.N,counts={'L_s':6300,'U_s':56700},shape=[m.N,2,256],dtype='float32',
        order='originalL_s_then_originalU_s',episode=m.EPISODE,
        TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0,
        F0_identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,
            model_seed=2026092701,source_run='20261027-phase1-pamf-repl-shuffled-views-m16-r02',
            source_row='main_split-repl-shuffled_time-3view-s2026092701',
            source_code_commit='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'))
    arrays=dict(clean=np.broadcast_to(np.zeros((1,2,256),np.float32),(m.N,2,256)),
        received=np.broadcast_to(np.zeros((1,2,256),np.float32),(m.N,2,256)),
        ids=np.asarray(['source:'+str(i) for i in range(m.N)]),
        roles=np.asarray(['L_s']*6300+['U_s']*56700),
        rx=np.zeros(m.N,np.int64),day=np.zeros(m.N,np.int64),
        F0_features=np.broadcast_to(np.zeros((1,160),np.float32),(m.N,160)))
    opened=[]
    monkeypatch.setattr(m,'read',lambda p:copy.deepcopy(manifest))
    def load(path,**kwargs):
        opened.append(path.stem)
        assert kwargs==dict(mmap_mode='r',allow_pickle=False)
        return arrays[path.stem]
    monkeypatch.setattr(m.np,'load',load)
    return manifest,arrays,opened


def test_pure_optimization_reader_never_opens_or_returns_F0_array(bundle):
    _,arrays,opened=bundle
    del arrays['F0_features']
    actual=m.load_cache('source',guided=False)
    assert opened==['clean','received','ids','roles','rx','day']
    assert 'F0_features' not in actual


@pytest.mark.parametrize('key,value',[
    ('OE_reads',1),('target_inputs',1),('TX_labels_read',1),('optimizer_updates',1),
    ('capsule_id','different'),('split_id','different'),('status','PARTIAL')])
def test_bad_source_handle_rejected_before_any_IQ_open(bundle,key,value):
    manifest,_,opened=bundle;manifest[key]=value
    with pytest.raises(ValueError,match='Completed original'):
        m.load_cache('source',guided=True)
    assert opened==[]


def test_guided_reader_requires_original_F0_not_other_same_dimension_teacher(bundle):
    manifest,_,_=bundle;manifest['F0_identity']['source_code_commit']='wrong'
    with pytest.raises(ValueError,match='guidance identity'):
        m.load_cache('source',guided=True)


@pytest.mark.parametrize('defect',['shape','role_order','IDs','guidance_shape'])
def test_misaligned_source_arrays_fail_instead_of_silent_truncation(bundle,defect):
    _,arrays,_=bundle
    if defect=='shape':arrays['received']=arrays['received'][:-1]
    elif defect=='role_order':arrays['roles']=arrays['roles'][::-1]
    elif defect=='IDs':arrays['ids'][1]=arrays['ids'][0]
    else:arrays['F0_features']=arrays['F0_features'][:,:159]
    with pytest.raises(ValueError):
        m.load_cache('source',guided=True)


def test_valid_guided_reader_exposes_aligned_targets_only_when_enabled(bundle):
    _,arrays,opened=bundle
    actual=m.load_cache('source',guided=True)
    assert opened[-1]=='F0_features' and actual['F0_features'] is arrays['F0_features']
