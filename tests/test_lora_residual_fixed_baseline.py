from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.pamf_lora_main import residual_dura_gate_run as gate
from experiments.pamf_lora_main import residual_dura_score as scorer


def test_fixed_baseline_uses_the_original_restorer_and_sibling_registered_reader(tmp_path, monkeypatch):
    chosen = gate.ADDITIONAL_READERS[0]
    parent = tmp_path / 'runs' / gate.PARENT_RUN
    fixed = tmp_path / 'runs' / gate.FIXED_BASELINE
    fixed.mkdir(parents=True)
    rt = dict(run_root=str(fixed), parent_root=str(parent))
    def smoke(runtime, *, restoration_seed):
        assert restoration_seed == gate.SEED
        assert runtime['model_seed'] == chosen
        assert Path(runtime['run_root']).parent == fixed
        assert 'target_views' not in runtime
        gate.reader_checkpoint(runtime, {'test': True}, chosen)
    complete = Mock(return_value=tmp_path / 'verified')
    monkeypatch.setattr(gate, 'complete_f', complete)
    monkeypatch.setattr(gate, 'smoke', smoke)
    gate.fixed_baseline_worker(rt, 'smoke', str(chosen))
    complete.assert_called_once_with(parent.parent / gate.SET_RUN, {'test': True},
                                     gate.case(chosen, gate.SET_METHOD))


def test_fixed_scorer_waits_for_all_rows_before_any_transfer_or_truth(tmp_path, monkeypatch):
    remote = '/runs/' + gate.FIXED_BASELINE
    submit = tmp_path / 'submit.json'
    submit.write_text(f'{{"run_root":"{remote}","archive_transfer":"VERIFIED"}}')
    wait = Mock(side_effect=RuntimeError('prediction matrix still incomplete'))
    transfer = Mock()
    monkeypatch.setattr(scorer, 'wait_for_predictions', wait)
    monkeypatch.setattr(scorer.subprocess, 'check_output', transfer)
    with pytest.raises(RuntimeError, match='still incomplete'):
        scorer.collect(submit, tmp_path / 'ssh', tmp_path / 'truth', tmp_path / 'new-output',
                       tmp_path / 'full-control', disturbance_gate=True,
                       fixed_reader_seed=gate.ADDITIONAL_READERS[1])
    assert wait.call_args.args[1] == remote
    transfer.assert_not_called()
    assert not (tmp_path / 'new-output').exists()
