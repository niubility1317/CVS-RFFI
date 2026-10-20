import copy
import pytest
from experiments.pamf_t1_r17_eval import runner as r

def metadata(segment):
    parent='/runs/'+r.T1_RUN+'/'+r.T1_ROW+'/stage20/final_ssdg.pth'
    sources=[] if segment==20 else [parent]
    c=dict(run_id=r.T1_RUN,row_id=r.T1_ROW,segment=segment,cumulative_epochs=segment,
           checkpoint_selection='final_only',separate_head_training=False,target_access=False,
           checkpoint_sources=sources,temperature=dict(T=1.,As_KL=0.,ordinary_KL=1.,raw_KL=1.,deployment_temperature=1.))
    init=dict(scratch_only=segment==20,checkpoint_sources=sources,ancestors=[],source_roles='EXACT_MATCH',target_contact=False,seed=2026092701)
    done=dict(status='SOURCE_TRAINED',epoch=20 if segment==20 else 10,optimizer_steps=4440 if segment==20 else 2220,
              target_access=False,target_evaluated=False,head_fit_updates=0,config=c,checkpoint='final_ssdg.pth')
    frozen=dict(status='SOURCE_FROZEN',selection='final_only',config=c,target_access=False,checkpoint='final_ssdg.pth')
    return init,done,frozen

@pytest.mark.parametrize('segment',[20,30])
def test_original_lineage(segment):
    assert r.validate_source_metadata(*metadata(segment),segment)['row_id']==r.T1_ROW

@pytest.mark.parametrize('field,value',[('run_id','F50'),('row_id','T6-s2026092701'),('cumulative_epochs',50),
                                      ('target_access',True),('checkpoint_selection','best_target'),('separate_head_training',True)])
def test_wrong_reader_or_selection_rejected(field,value):
    i,d,f=copy.deepcopy(metadata(30));d['config'][field]=value
    with pytest.raises(ValueError):r.validate_source_metadata(i,d,f,30)

@pytest.mark.parametrize('field,value',[('T',6.),('As_KL',1.),('ordinary_KL',0.),('raw_KL',0.),('deployment_temperature',6.)])
def test_temperature_and_kl_contract(field,value):
    i,d,f=copy.deepcopy(metadata(30));d['config']['temperature'][field]=value
    with pytest.raises(ValueError):r.validate_source_metadata(i,d,f,30)

def test_cross_parent_rejected():
    i,d,f=metadata(30);i['checkpoint_sources']=['/other/stage20/final_ssdg.pth']
    with pytest.raises(ValueError):r.validate_source_metadata(i,d,f,30)

def original_prediction(original):
    if original:return dict(status='PREDICTIONS_COMPLETE',row_id=r.T1_ROW,count=r.COUNT,views=list(r.VIEWS),truth_read=False,
                            query_fit=False,all_parameters_frozen=True,head_fit_updates=0)
    return dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=r.COUNT,views=list(r.VIEWS),
                readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0)

@pytest.mark.parametrize('original',[False,True])
def test_all_views_required_before_truth(original):
    value=original_prediction(original);r.validate_prediction_metadata(value,original=original)
    value['views']=value['views'][:-1]
    with pytest.raises(ValueError):r.validate_prediction_metadata(value,original=original)

@pytest.mark.parametrize('original',[False,True])
def test_no_fitted_head(original):
    value=original_prediction(original);value['head_fit_updates']=1
    with pytest.raises(ValueError):r.validate_prediction_metadata(value,original=original)

def test_score_does_not_connect_truth_after_bad_predictions(monkeypatch,tmp_path):
    import os
    from types import SimpleNamespace
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def incomplete(args):raise ValueError('missing seventh view')
    monkeypatch.setattr(r,'completed_predictions',incomplete)
    monkeypatch.setattr(r,'read',lambda *a:pytest.fail('truth must remain closed'))
    with pytest.raises(ValueError,match='seventh'):r.score(SimpleNamespace(output=tmp_path,base=tmp_path))
