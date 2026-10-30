import pytest
from experiments.pamf_phase2_practical.external_accel_takeover import resolve_jobs,_check_pending


def _job(root,origin,kind,original,dependencies=()):
    output=root/original
    j=dict(id=origin+'-'+original,origin=origin,kind=kind,original_id=original,
        profile='issl' if 'issl' in original else 'pamf_tri_sig_time_ssl_milkr',
        dependencies=list(dependencies),output=str(output),completion=str(output/'complete.json'),
        runtime_file='old-runtime',release_root='old-release',execution_version='original')
    return j


@pytest.mark.parametrize('state,expected',[('done','original'),('active','original'),('pending','external_cuda_v2')])
def test_prediction_and_score_routing_preserves_active_and_completed(tmp_path,state,expected):
    old=_job(tmp_path/'old','legacy','predict','predict-issl')
    score=_job(tmp_path/'old','legacy','score','score-predict-issl',[old['id']]);score['prediction']=old['output']
    new=_job(tmp_path/'new','legacy','predict','predict-issl')
    new_score=_job(tmp_path/'new','legacy','score','score-predict-issl');new_score['prediction']=new['output']
    pamf=_job(tmp_path/'pamf','replacement','predict','predict-pamf');pamf['execution_version']='cuda'
    done={old['id']} if state=='done' else set();active={old['id']:{}} if state=='active' else {}
    result=resolve_jobs([old,score,pamf],done,active,
        {new['original_id']:new,new_score['original_id']:new_score},
        {'runtime_file':'new-runtime','release_root':'new-release','method_profiles':['issl','lora_triplet_knn']})
    assert result[0]['execution_version']==expected and result[1]['execution_version']==expected
    assert result[1]['prediction']==result[0]['output']
    assert result[2]==pamf


def test_other_external_profiles_keep_existing_release(tmp_path):
    job=_job(tmp_path/'old','legacy','predict','predict-dadda_sda');job['profile']='dadda_sda'
    result=resolve_jobs([job],set(),{}, {},
        {'runtime_file':'new','release_root':'new','method_profiles':['issl','lora_triplet_knn']})
    assert result[0]==job


@pytest.mark.parametrize('artifact',['log','output'])
def test_unowned_partial_never_restarted(tmp_path,artifact):
    j=_job(tmp_path/'output','legacy','predict','predict-issl')
    cfg={'old-runtime':{'run_root':str(tmp_path/'run')}}
    _check_pending([j],cfg)
    from pathlib import Path
    if artifact=='log':
        p=Path(cfg['old-runtime']['run_root'])/'logs'/'predict-issl.log';p.parent.mkdir(parents=True);p.touch()
    else:Path(j['output']).mkdir(parents=True)
    with pytest.raises(ValueError,match='Unowned partial'):_check_pending([j],cfg)
