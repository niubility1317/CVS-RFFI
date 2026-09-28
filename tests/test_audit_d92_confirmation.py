import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_d92_confirmation import audit


def fixture():
    cm=[[2 if i==j else 0 for j in range(8)] for i in range(8)];cm[0][6]=1
    row=dict(split_id='test',classes=[str(i) for i in range(8)],class_count=8,new_count=2,
        confusion=cm,accuracy=16/17,old_accuracy=12/13,new_accuracy=1,harmonic_mean=24/25,
        macro_f1=.95,old_macro_f1=5.8/6,new_macro_f1=.9,old_floor=2/3,new_floor=1,
        class_accuracy=[2/3]+[1]*7,query_count=17)
    return dict(status='SCORED',results=[row])


def test_independent_counts_reproduce_unequal_class_metrics():
    assert audit(fixture(),1)['status']=='VERIFIED'


@pytest.mark.parametrize('key', ['accuracy','old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_floor','query_count'])
def test_rejects_tampered_reported_metric(key):
    data=fixture();data['results'][0][key]+=0.1
    with pytest.raises(ValueError):audit(data,1)


def test_requires_complete_nonnegative_integer_counts():
    data=fixture()
    with pytest.raises(ValueError):audit(data,2)
    broken=copy.deepcopy(data);broken['results'][0]['confusion'][0][0]=2.0
    with pytest.raises(ValueError):audit(broken,1)
