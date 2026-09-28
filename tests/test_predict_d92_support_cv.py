import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from predict_d92_support_cv import validate_split


def fixture():
    classes=[str(i) for i in range(8)]
    split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='synthetic',
        registered_classes=classes,support_indices=list(range(8)),query_indices=[8,9],support_labels=list(range(8)),k=1)
    return split,dict(capsule_id='synthetic'),np.asarray([str(i) for i in range(10)]),classes[:6]


def test_accepts_support_only_split():
    args=fixture();s,q,y=validate_split(*args)
    assert len(s)==8 and len(q)==2 and set(y)==set(range(8))


@pytest.mark.parametrize('change', [
    {'query_labels':[1,2]}, {'query_roles':['old','new']},
    {'query_indices':[0,8]}, {'support_indices':[0]*8},
    {'support_labels':[0]*8}, {'support_labels':[float(i) for i in range(8)]},
    {'support_indices':[False]+list(range(1,8))}, {'capsule_id':'different'},
    {'k':2}, {'query_indices':[8,999]},
])
def test_rejects_truth_overlap_or_wrong_binding(change):
    split,manifest,ids,old=fixture();split.update(copy.deepcopy(change))
    with pytest.raises(ValueError):validate_split(split,manifest,ids,old)
