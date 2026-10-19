import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.pamf_native_formal.runner import (
    KIND, PARENT, PARENT_COMMIT, RUN, VIEWS, configuration,
    qualify_source, parent_contract, prepare_source, assess_goal,
)
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN


def summary(kind, nmse, accuracy):
    name = 'r03_reference' if kind == 'reference_controls' else 'restored'
    views = {view: {name: dict(waveform=dict(iq_nmse=nmse))} for view in VIEWS}
    views['clean'][name]['waveform']['iq_nmse'] = .001
    records = [dict(view='practical_low_urban', readout=name, dimension='overall', value='all', accuracy=accuracy),
               dict(view='clean', readout=name, dimension='overall', value='all', accuracy=98.5)]
    records += [dict(view='practical_low_urban', readout=name, dimension='TX', value=str(i), accuracy=accuracy) for i in range(6)]
    return dict(status='ANALYZED', kind=kind, role='source', count=1800, views=views, records=records,
        F_run=REFERENCE_RUN, F_seed=2026092701, original_default_head=True, all_predictions_before_truth=True,
        training_TX_labels_read=0, training_F_access=False, head_fit_updates=0, target_access=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def parent(tmp_path):
    root = tmp_path/PARENT
    write(root/'launch.json', dict(commit=PARENT_COMMIT))
    write(root/'complete.json', dict(status='ANALYZED', source_only=True, target_rows=[], training_TX_F_access=False))
    write(root/KIND/'training_complete.json', dict(status='SOURCE_TRAINED', kind=KIND, epochs=20, updates=4940,
        L_exposures=126000, U_exposures=1134000, TX_labels_read=0, F_outputs_read=0, target_reads=0, frozen_context_updates=0))
    write(root/KIND/'complete.json', dict(status='ANALYZED', kind=KIND, role='source', all_predictions_before_truth=True, training_TX_F_access=False))
    write(root/KIND/'summary.json', summary(KIND, .0276, 82.16))
    write(root/'reference_controls/summary.json', summary('reference_controls', .02836, 79.89))
    (root/KIND/'final.pt').write_bytes(b'source checkpoint placeholder')
    return root


def test_native_source_is_separately_frozen_without_rewriting_R10_decision():
    decision = qualify_source(summary(KIND, .0276, 82.16), summary('reference_controls', .02836, 79.89))
    assert decision['qualified'] and decision['target_candidates'] == [KIND]
    assert decision['selection_stage'] == RUN and decision['previous_R10_primary_changed'] is False
    assert decision['training_TX_F_access'] is False and decision['target_reads_before_decision'] == 0


def test_any_waveform_clean_or_worst_TX_failure_stops_promotion():
    ref = summary('reference_controls', .02836, 79.89)
    assert not qualify_source(summary(KIND, .03, 83), ref)['qualified']
    bad = summary(KIND, .027, 82); bad['records'][-1]['accuracy'] = 78
    assert not qualify_source(bad, ref)['qualified']
    bad = summary(KIND, .027, 82); bad['records'][1]['accuracy'] = 98.2
    assert not qualify_source(bad, ref)['qualified']


@pytest.mark.parametrize('change', [dict(role='target'), dict(kind='uncertain_map_prior'), dict(training_F_access=True),
    dict(target_access=True), dict(all_predictions_before_truth=False), dict(F_seed=2026092702)])
def test_foreign_or_contaminated_source_evidence_is_rejected(change):
    with pytest.raises(ValueError):
        qualify_source(dict(summary(KIND, .0276, 82.16), **change), summary('reference_controls', .02836, 79.89))


def test_fixed_parent_training_budget_and_target_nonuse_are_checked(tmp_path):
    root = parent(tmp_path)
    assert parent_contract(root)['qualified']
    value = json.loads((root/KIND/'training_complete.json').read_text())
    for change in (dict(updates=4939), dict(TX_labels_read=1), dict(F_outputs_read=1), dict(target_reads=1), dict(frozen_context_updates=1)):
        write(root/KIND/'training_complete.json', dict(value, **change))
        with pytest.raises(ValueError):
            parent_contract(root)


def test_source_preparation_preserves_parent_and_rejects_existing_output_links(tmp_path):
    root = parent(tmp_path); output = tmp_path/RUN; output.mkdir()
    # A fake source suffices: no checkpoint deserialization or target IQ is needed.
    args = SimpleNamespace(parent_run=root, output=output)
    original = (root/'complete.json').read_bytes()
    prepare_source(args)
    assert (output/KIND/'final.pt').read_bytes() == (root/KIND/'final.pt').read_bytes()
    assert (root/'complete.json').read_bytes() == original
    assert json.loads((output/'source_decision.json').read_text())['target_candidates'] == [KIND]
    with pytest.raises(FileExistsError):
        prepare_source(args)


def test_unqualified_source_cannot_create_target_authorization(tmp_path):
    root = parent(tmp_path); output = tmp_path/RUN; output.mkdir()
    bad = summary(KIND, .027, 82); bad['records'][-1]['accuracy'] = 78
    write(root/KIND/'summary.json', bad)
    with pytest.raises(ValueError):
        prepare_source(SimpleNamespace(parent_run=root, output=output))
    assert not (output/'source_decision.json').exists()
    assert not (output/KIND).exists()


def test_formal_plan_keeps_one_frozen_candidate_and_all_seven_views():
    plan = configuration()
    assert plan['run'] == RUN and plan['candidate'] == KIND and plan['count'] == 168000
    assert plan['views'] == list(VIEWS) and plan['new_training_updates'] == plan['head_fit_updates'] == plan['channel_generation_calls'] == 0
    assert plan['capacity_lock'] == 'receiver_residual_capacity4.lock' and plan['max_own_GPU_workers'] == 1


def test_user_targets_use_original_clean_and_all_five_other_scenes():
    result = summary(KIND, .027, 82)
    result['role'] = 'target'; result['count'] = 168000
    result['records'] = [dict(view=view, readout='restored', dimension='overall', value='all',
        accuracy=70. if view == 'practical_low_urban' else 85., baseline_accuracy=90.) for view in VIEWS]
    assert assess_goal(result)['performance_targets_met']
    bad = copy.deepcopy(result); bad['records'][-1]['accuracy'] = 69.99
    assert not assess_goal(bad)['performance_targets_met']
    bad = copy.deepcopy(result); bad['records'][2]['accuracy'] = 84.99
    assert not assess_goal(bad)['performance_targets_met']
    # Degrading restored clean must not relax the reference used by other scenes.
    bad['records'][0]['accuracy'] = 80.
    assert not assess_goal(bad)['performance_targets_met']
