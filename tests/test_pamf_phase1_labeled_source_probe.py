import numpy as np
import pytest

from experiments.pamf_phase1_flow.labeled_source_probe import align_labeled_positions, source_counts


def test_counts_distinguish_clean_teacher_and_dura_errors():
    truth=np.array([0,1,1,2,0])
    clean=np.array([0,1,0,2,1])
    dura=np.array([0,0,1,1,1])
    mask=np.array([True,True,True,True,False])
    assert source_counts(truth,clean,dura,mask)==dict(count=4,clean_correct=3,dura_correct=2,
        clean_dura_disagree=3,disagree_clean_only_correct=2,disagree_dura_only_correct=1,disagree_both_wrong=0)


def test_rejects_bad_or_empty_mask():
    with pytest.raises(ValueError):
        source_counts(np.array([0]),np.array([0]),np.array([0]),np.array([False]))


def test_labels_align_by_physical_id_not_storage_order():
    roles=np.array(['L_s','U_s','L_s','L_s'])
    physical=np.array(['a','u','b','c'])
    assert align_labeled_positions(roles,physical,np.array(['c','a','b'])).tolist()==[3,0,2]
    with pytest.raises(ValueError):
        align_labeled_positions(roles,physical,np.array(['c','a','u']))
