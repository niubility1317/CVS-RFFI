import numpy as np
import pytest
from experiments.pamf_phase3_restoration.source_inverse import align_source,panel,summarize,METHODS,SCENES


def source():
    ids=np.asarray(['id'+str(i) for i in range(63000)])
    return ids,np.asarray(['L_s']*6300+['U_s']*56700)


def test_realignment_preserves_roles_not_array_order():
    ids,roles=source();perm=np.arange(62999,-1,-1)
    order=align_source(ids,roles,ids[perm],roles[perm])
    assert np.array_equal(ids[perm][order],ids)
    selected=panel(roles)
    assert len(set(selected.tolist()))==512
    assert (roles[selected]=='L_s').sum()==128
    assert (roles[selected]=='U_s').sum()==384


def test_reject_changed_source_role_and_duplicated_ID():
    ids,roles=source();changed=roles.copy();changed[[0,6300]]=changed[[6300,0]]
    with pytest.raises(ValueError,match='roles'):align_source(ids,roles,ids,changed)
    bad=ids.copy();bad[0]=bad[1]
    with pytest.raises(ValueError,match='physical ID'):align_source(ids,roles,bad,roles)


def test_partial_diagnostic_cannot_be_summarized():
    with pytest.raises(ValueError,match='Full fixed'):summarize({'scene':np.asarray(SCENES)})


def test_summary_reports_each_scene_and_uses_all_cases():
    scene=np.resize(np.asarray(SCENES),10240);records={'scene':scene}
    for method in METHODS:
        for metric in ('nmse','feature_cosine_distance','clean_decision_agreement'):
            records[method+'__'+metric]=(scene=='practical_low_urban').astype(float)
    result=summarize(records)
    assert sum(result[s]['count'] for s in SCENES)==10240
    assert result['practical_low_urban']['methods']['DURA_v1']['nmse']==1
    assert result['clean']['methods']['DURA_v1']['nmse']==0
