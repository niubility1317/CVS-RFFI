"""Synthetic arithmetic checks for paired source evidence, not model accuracy."""
import copy

import pytest

from experiments.cvs_spectral_relation_identity.analyze import (
    RELATION_SCALARS, aggregate, summarize_cells, summarize_curves, summarize_records,
)
from experiments.cvs_spectral_relation_identity.dispatch import CANDIDATES, SEEDS
from experiments.cvs_spectral_relation_identity.model import VARIANTS


def records():
    rows = []
    for i,variant in enumerate(CANDIDATES):
        for j,seed in enumerate(sorted(SEEDS)):
            rows.append(dict(variant=variant,seed=seed,accuracy=.8+.01*j+.005*i,
                             worst_rx=.7+.02*j+.005*i,parameters=247731,macs=1e6))
    return rows


def curves():
    rows = []
    for variant in VARIANTS:
        for seed in sorted(SEEDS):
            epochs = [dict(epoch=e,clean_ce=1-e/1000,source_val_ce=1.2-e/2000,
                source_val_accuracy=.8+e/2000,source_val_worst_rx=.7+e/2000,
                spectral_relation_gradient_norm=e/200,
                spectral_relation_diagnostics={'spectral_relation':{'records':[
                    dict(block='frequency.spectral_relation',packets=28,relative_output_change_mean=.1)]}})
                for e in range(1,201)]
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed),epochs=epochs))
    return rows


def cell_rows():
    rows = []
    for variant in VARIANTS:
        for seed in sorted(SEEDS):
            groups = []
            for tx in range(6):
                for receiver in (1,3,4,6,8):
                    for day in (1,2,3):
                        group = dict(tx=tx,receiver=receiver,day=day,count=300)
                        for metric in RELATION_SCALARS:
                            group.update({metric+'_mean':receiver/10,metric+'_min':0.,metric+'_max':1.})
                        groups.append(group)
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed),
                             source_diagnostics=dict(spectral_relation_groups=groups)))
    return rows


def test_paired_seeds_are_joined_by_identity_not_input_order():
    data = records()
    result = summarize_records(data[::-1])
    comparison = next(r for r in result['source_paired_summary']
                      if r['left']==VARIANTS[1] and r['right']==VARIANTS[0] and r['metric']=='score')
    assert comparison['mean_delta_pp']==pytest.approx(.5)
    assert comparison['paired_SD_pp']==pytest.approx(0,abs=1e-12)
    assert comparison['positive_seeds']==4
    assert result['selection']['selected_variant']==VARIANTS[1]


@pytest.mark.parametrize('mutation',['missing','duplicate','nan','bool'])
def test_bad_source_records_rejected(mutation):
    data = records()
    if mutation=='missing': data.pop()
    elif mutation=='duplicate': data[-1]=copy.deepcopy(data[-2])
    elif mutation=='nan': data[0]['accuracy']=float('nan')
    else: data[0]['accuracy']=True
    with pytest.raises(ValueError): summarize_records(data)


def test_full_curve_windows_and_ce_gap_recomputed():
    result = summarize_curves(curves())
    assert len(result['source_curves'])==1600
    assert len(result['source_curve_summary'])==400
    assert len(result['spectral_relation_epoch_outputs'])==1600
    row = result['source_learning_by_seed'][0]
    assert row['train_CE_late_delta']==pytest.approx(-.025)
    assert row['V_CE_late_delta']==pytest.approx(-.0125)
    assert row['CE_gap_late_delta']==pytest.approx(.0125)
    assert row['CE_gap_E200']==pytest.approx(.3)
    assert result['source_gradient_by_seed'][0]['mean']==pytest.approx(.5025)


@pytest.mark.parametrize('mutation',['missing_epoch','out_of_order','duplicate_seed','nonfinite_gradient'])
def test_incomplete_or_invalid_curves_rejected(mutation):
    data = curves()
    if mutation=='missing_epoch': data[0]['epochs'].pop()
    elif mutation=='out_of_order': data[0]['epochs'].reverse()
    elif mutation=='duplicate_seed': data[-1]=copy.deepcopy(data[-2])
    else: data[0]['epochs'][99]['spectral_relation_gradient_norm']=float('inf')
    with pytest.raises(ValueError): summarize_curves(data)


def test_source_rx_aggregation_uses_all_cells_and_separate_model_seeds():
    result = summarize_cells(cell_rows())
    assert len(result['source_relation_cells'])==720
    assert len(result['source_relation_by_seed'])==48
    assert len(result['source_relation_summary'])==12
    row = result['source_relation_by_seed'][0]
    assert row['receiver']=='ALL' and row['count']==27000
    assert row['branch_output_norm_mean']==pytest.approx(.44)
    rx = next(r for r in result['source_relation_summary'] if r['receiver']==8)
    assert rx['floor_fraction_mean_mean']==pytest.approx(.8)
    assert rx['floor_fraction_mean_SD']==0


@pytest.mark.parametrize('mutation',['missing_cell','duplicate_cell','wrong_count','missing_seed'])
def test_incomplete_source_cell_aggregation_rejected(mutation):
    data = cell_rows()
    groups = data[0]['source_diagnostics']['spectral_relation_groups']
    if mutation=='missing_cell': groups.pop()
    elif mutation=='duplicate_cell': groups[-1]=copy.deepcopy(groups[-2])
    elif mutation=='wrong_count': groups[0]['count']=299
    else: data.pop()
    with pytest.raises(ValueError): summarize_cells(data)
