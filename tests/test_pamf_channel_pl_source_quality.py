from copy import deepcopy
import pytest
from experiments.pamf_channel_pl.source_quality import validate_prefix_headers,scientific_rule


def headers():
    contract={'classes':[0,1,2,3,4,5],'source_rxs':[1],'source_days':[1,2,3]}
    cfg={'model_seed':2026092701,'num_classes':6,'ablation':'F21','run_id':'parent','row_id':'PL-s1',
        'parent_source':None,'own_EMA':True}
    done={'status':'SOURCE_FROZEN','epoch':20,'updates':4440,'target_access':False,'U_truth_read':False,
        'head_fit_updates':0,'initialization':'scratch','ancestors':[],'config':cfg}
    payload={'run_id':'parent','candidate_id':'PL-s1','epoch':20,'a1_ema_successful_updates':4440,
        'args':{'baseline_ckpt':'','teacher_ckpt':'','use_unlabeled':False,'use_ema_teacher':False}}
    case={'model_seed':2026092701,'source_config':deepcopy(cfg)}
    return case,done,payload,contract,deepcopy(contract)


def test_actual_F21_payload_overrides_legacy_declarative_EMA_flag():
    validate_prefix_headers(*headers())


@pytest.mark.parametrize('bad',['target','Utruth','head','epoch','seed','classes','teacher','EMA','native_U','inherited','contract'])
def test_rejects_illegal_or_mismatched_checkpoint(bad):
    c,d,p,a,b=headers()
    if bad=='target':d['target_access']=True
    elif bad=='Utruth':d['U_truth_read']=True
    elif bad=='head':d['head_fit_updates']=1
    elif bad=='epoch':p['epoch']=30
    elif bad=='seed':d['config']['model_seed']=7
    elif bad=='classes':d['config']['num_classes']=10
    elif bad=='teacher':p['args']['teacher_ckpt']='foreign'
    elif bad=='EMA':p['args']['use_ema_teacher']=True
    elif bad=='native_U':p['args']['use_unlabeled']=True
    elif bad=='inherited':p['args']['baseline_ckpt']='foreign'
    else:b['source_rxs']=[2]
    with pytest.raises(ValueError):validate_prefix_headers(c,d,p,a,b)


def test_scientific_rule_is_precision_and_coverage_not_validation_accuracy_ranking():
    values=dict(candidate=1000,candidate_correct=999,transfer=200,transfer_correct=200)
    assert scientific_rule(values)['pass']
    for change in ({'transfer_correct':197},{'transfer':50,'transfer_correct':50},{'candidate':0,'candidate_correct':0,'transfer':0,'transfer_correct':0}):
        changed=dict(values,**change)
        assert not scientific_rule(changed)['pass']
