"""Independent source-shaped CPU geometry checks; no data or learned model."""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .relations import measure


def _fixture():
    labels = torch.arange(6).repeat_interleave(1050)
    receivers = torch.tensor([1,3,4,6,8]).repeat_interleave(210).repeat(6)
    days = torch.tensor([1,2,3]).repeat_interleave(70).repeat(30)
    identifiers = [f"synthetic-{index}" for index in range(6300)]
    valid = torch.ones(6300, dtype=torch.bool)
    return labels,receivers,days,identifiers,valid


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected fixed-source metadata/role rejection")


def run_checks():
    python_rng = random.getstate(); numpy_rng = np.random.get_state(); torch_rng = torch.get_rng_state().clone()
    labels,receivers,days,identifiers,valid = _fixture()
    tx_vectors = F.one_hot(labels,24).to(torch.float64).requires_grad_()
    values_before = tx_vectors.detach().clone()
    labels_before,receivers_before,days_before,valid_before = [value.clone() for value in (labels,receivers,days,valid)]
    ids_before = tuple(identifiers)
    tx = measure(tx_vectors,labels,receivers,days,identifiers,valid)
    assert tx["summary"]["overall"]["relation_count"] == 126000
    assert tx["summary"]["overall"]["valid_count"] == 126000
    assert tx["summary"]["overall"]["gap"]["mean"] == 1.0
    assert tx["summary"]["overall"]["ordering_le_zero_count"] == 0
    assert tx["summary"]["cross_RX_same_TX_pairs"]["cosine"]["mean"] == 1.0
    assert tx["summary"]["same_RX_other_TX_pairs"]["cosine"]["mean"] == 0.0
    assert tx_vectors.grad is None and not tx["tensors"]["gap"].requires_grad
    indices = tx["tensors"]["relations"]
    a,p,n = indices.unbind(dim=1)
    assert torch.equal(a,torch.arange(6300).repeat_interleave(20))
    assert bool((labels[a] == labels[p]).all()) and bool((labels[a] != labels[n]).all())
    assert bool((receivers[a] != receivers[p]).all()) and bool((receivers[p] == receivers[n]).all())
    assert bool((days[a] == days[p]).all()) and bool((days[p] == days[n]).all())
    assert bool((a != p).all()) and bool((a != n).all()) and bool((p != n).all())
    # Independent original-order cell ordinal census, without sorting IDs.
    counters = Counter(); ordinals=[]
    for label,receiver,day in zip(labels.tolist(),receivers.tolist(),days.tolist()):
        key=(label,receiver,day);ordinals.append(counters[key]);counters[key]+=1
    ordinals=torch.tensor(ordinals)
    assert torch.equal(tx["tensors"]["cell_ordinals"],ordinals)
    assert bool((ordinals[a] == ordinals[p]).all()) and bool((ordinals[p] == ordinals[n]).all())
    assert all(count == 70 for count in counters.values())
    # Interleave cells and rotate their members: original input order, not ID
    # order or the former fixture ordinal, must define the returned plan.
    permutation=torch.arange(6300).reshape(90,70).roll(13,dims=1).T.reshape(-1)
    shuffled=measure(tx_vectors[permutation],labels[permutation],receivers[permutation],days[permutation],
                     [identifiers[index] for index in permutation.tolist()],valid[permutation])
    shuffled_ordinals=torch.arange(70).repeat_interleave(90)
    assert torch.equal(shuffled["tensors"]["cell_ordinals"],shuffled_ordinals)
    shuffled_a,shuffled_p,shuffled_n=shuffled["tensors"]["relations"].unbind(dim=1)
    assert bool((shuffled_ordinals[shuffled_a] == shuffled_ordinals[shuffled_p]).all())
    assert bool((shuffled_ordinals[shuffled_p] == shuffled_ordinals[shuffled_n]).all())
    assert sum(row["relation_count"] for row in tx["summary"]["RX"].values()) == 126000
    assert sum(row["relation_count"] for row in tx["summary"]["day"].values()) == 126000
    assert sum(row["relation_count"] for row in tx["summary"]["TX"].values()) == 126000
    assert tx["summary"]["overall"]["gap_mean"] == 1.0
    for name in ("RX", "day", "TX"):
        assert all(row["gap_mean"] == 1.0 for row in tx["summary"][name].values())
    python_lists=measure(tx_vectors,labels.tolist(),receivers.tolist(),days.tolist(),identifiers,valid)
    assert python_lists["summary"] == tx["summary"]
    assert torch.equal(python_lists["tensors"]["relations"], indices)
    rx_index=torch.tensor([{1:0,3:1,4:2,6:3,8:4}[int(receiver)] for receiver in receivers])
    rx_vectors=F.one_hot(rx_index,24).to(torch.float64)
    rx=measure(rx_vectors,labels,receivers,days,identifiers,valid)
    assert rx["summary"]["overall"]["gap"]["mean"] == 0.0
    assert rx["summary"]["overall"]["ordering_le_zero_count"] == 126000
    assert rx["summary"]["same_RX_other_TX_pairs"]["cosine"]["mean"] == 1.0
    assert rx["summary"]["cross_RX_same_TX_pairs"]["cosine"]["mean"] == 0.0
    invalid_one=valid.clone();invalid_one[0]=False
    excluded=measure(tx_vectors,labels,receivers,days,identifiers,invalid_one)
    assert excluded["summary"]["overall"]["valid_count"] == 125940
    assert excluded["summary"]["overall"]["invalid_count"] == 60
    assert excluded["summary"]["overall"]["gap"]["mean"] == 1.0
    assert excluded["summary"]["cross_RX_same_TX_pairs"]["invalid_count"] == 8
    assert excluded["summary"]["same_RX_other_TX_pairs"]["invalid_count"] == 10
    empty=measure(tx_vectors,labels,receivers,days,identifiers,torch.zeros_like(valid))
    assert empty["summary"]["overall"]["valid_count"] == 0
    assert empty["summary"]["overall"]["invalid_count"] == 126000
    assert empty["summary"]["overall"]["ordering_le_zero_rate"] is None
    for group in (empty["summary"]["overall"], *empty["summary"]["RX"].values(), *empty["summary"]["day"].values(), *empty["summary"]["TX"].values()):
        for key in ("cos_positive","cos_negative","gap"):
            assert group[key]["count"] == 0 and group[key]["mean"] is None and group[key]["std"] is None
    # Zero residuals are disclosed and retained whenever the visible flag is true.
    zero_vectors=torch.zeros_like(tx_vectors)
    zero=measure(zero_vectors,labels,receivers,days,identifiers,valid)
    assert zero["summary"]["zero_vector_count"] == zero["summary"]["valid_zero_vector_count"] == 6300
    assert zero["summary"]["overall"]["valid_count"] == 126000
    assert zero["summary"]["overall"]["gap"]["mean"] == 0.0
    unequal=labels.clone();unequal[0]=1
    _raises(lambda:measure(tx_vectors,unequal,receivers,days,identifiers,valid))
    duplicates=list(identifiers);duplicates[1]=duplicates[0]
    _raises(lambda:measure(tx_vectors,labels,receivers,days,duplicates,valid))
    for role in ("U_s","V","query"):
        _raises(lambda role=role:measure(tx_vectors,labels,receivers,days,identifiers,valid,role=role))
    _raises(lambda:measure(tx_vectors[:6299],labels[:6299],receivers[:6299],days[:6299],identifiers[:6299],valid[:6299]))
    _raises(lambda:measure(tx_vectors[:,:23],labels,receivers,days,identifiers,valid))
    _raises(lambda:measure(tx_vectors,labels.to(torch.float64),receivers,days,identifiers,valid))
    _raises(lambda:measure(tx_vectors,labels,receivers,days,identifiers,valid.to(torch.int64)))
    hidden=labels.clone();hidden[0]=-1
    _raises(lambda:measure(tx_vectors,hidden,receivers,days,identifiers,valid))
    bad_rx=receivers.clone();bad_rx[0]=0
    _raises(lambda:measure(tx_vectors,labels,bad_rx,days,identifiers,valid))
    nonfinite=tx_vectors.detach().clone();nonfinite[0,0]=float('nan')
    _raises(lambda:measure(nonfinite,labels,receivers,days,identifiers,valid))
    _raises(lambda:measure(tx_vectors,labels.tolist()[:-1],receivers,days,identifiers,valid))
    bad_day=days.clone();bad_day[0]=4
    _raises(lambda:measure(tx_vectors,labels,receivers,bad_day,identifiers,valid))
    assert torch.equal(tx_vectors.detach(),values_before)
    for left,right in zip((labels,receivers,days,valid),(labels_before,receivers_before,days_before,valid_before)):
        assert torch.equal(left,right)
    assert tuple(identifiers)==ids_before and tx_vectors.grad is None
    existing_gradient=torch.full_like(tx_vectors,.375)
    tx_vectors.grad=existing_gradient.clone()
    retained=measure(tx_vectors,labels,receivers,days,torch.arange(6300),valid)
    assert torch.equal(tx_vectors.grad,existing_gradient)
    assert retained["summary"] == tx["summary"]
    assert random.getstate()==python_rng and torch.equal(torch.get_rng_state(),torch_rng)
    current_numpy=np.random.get_state()
    assert current_numpy[0]==numpy_rng[0] and np.array_equal(current_numpy[1],numpy_rng[1]) and current_numpy[2:]==numpy_rng[2:]
    for result in (tx,rx,excluded,empty,zero,shuffled,retained):json.dumps(result["summary"],allow_nan=False)
    return {"status":"VERIFIED","environment":{"torch":torch.__version__,"device":"cpu","dtype":"float64"},
            "checks":{"TX_onehot_gap1":True,"RX_onehot_gap0_same_RX_other_TX_cos1":True,
                      "single_invalid_excludes_exactly60":True,"allinvalid_null_statistics":True,
                      "original_cell_ordinal_preserved":True,"zero_vectors_disclosed_not_filtered":True,
                      "root_summary_keys_and_Python_list_metadata":True,
                      "invalid_source_cells_IDs_roles_rejected":True,"inputs_gradients_RNG_unchanged":True},
            "relation_count":126000,"single_invalid_remaining":125940,
            "boundary":"visible_mean physical residual source geometry only; no model-performance, DG or new-TX claim"}


if __name__=="__main__":
    print(json.dumps(run_checks(),ensure_ascii=False,allow_nan=False,indent=2))
