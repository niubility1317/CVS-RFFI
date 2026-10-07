from types import SimpleNamespace
import pytest
import torch
from experiments.cvs_phase1_repair import design as d
from experiments.cvs_phase1_overlay.model import native_modules


def test_frozen_matrix_and_schedule():
    assert len(d.rows()) == 32
    for row in d.rows():
        c = d.config(row); d.validate(c)
        assert c['output_root'].startswith('/home/') and '\\' not in c['output_root']
        a = d.make_args(c, 'cpu')
        assert not a.pseudo_domain_gate and not a.baseline_ckpt and not a.teacher_ckpt
        assert not a.use_muse_ssdg and a.epochs == 200
        assert a.pseudo_temporal_mode == c['temporal_mode']
        with pytest.raises(ValueError): d.validate(dict(c, lr_min=2e-6))
    values = [d.cosine_lr(2e-4, e) for e in range(1, 201)]
    assert values[0] == pytest.approx(2e-4) and values[-1] == pytest.approx(1e-6)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_temporal_bank_survives_shuffle_but_not_prediction_change_or_missing_epoch():
    n = native_modules(); bank = {}; a = SimpleNamespace(pseudo_temporal_min_conf=.8, pseudo_temporal_bank_min_streak=2)
    def run(epoch, ids, predictions, confidence=None, rx=1):
        meta = {k: torch.tensor(ids) for k in ('sig_i', 'base_index')}
        meta.update({k: torch.full((len(ids),), v) for k, v in [('rx_i', rx), ('day_i', 1), ('eq_i', 1)]})
        return n._temporal_bank_mask_tensor(torch.tensor(predictions), torch.tensor(confidence or [.99]*len(ids)),
                    [None, meta], a, 'cpu', epoch=epoch, bank=bank).tolist()
    assert run(131, [1, 99], [2, 3]) == [False, False]
    assert run(131, [1], [2]) == [False]  # same epoch cannot manufacture a streak
    assert run(132, [99, 1], [3, 2]) == [True, True]
    assert run(133, [1], [4]) == [False]
    assert run(134, [99], [3]) == [False]  # epoch gap resets
    assert run(133, [99], [3], rx=3) == [False]  # receiver is part of identity
    assert run(134, [1], [4], [.4]) == [False]
    assert run(135, [1], [4]) == [False]


def test_snapshot_requires_all_fixed_rows_before_target(tmp_path, monkeypatch):
    from experiments.cvs_phase1_repair import evaluate as e
    monkeypatch.setattr(e,'BASE',tmp_path)
    value=dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=[d.config(r) for r in d.rows()],
               target_access=False,selection='ALL_PREREGISTERED_FIXED_CONTROLS')
    d.write(tmp_path/'source_matrix_frozen.json',value)
    assert len(e.snapshot(d)) == 32
    value['rows'].pop();d.write(tmp_path/'source_matrix_frozen.json',value)
    with pytest.raises(ValueError):e.snapshot(d)


def test_paired_results_keep_four_seed_alignment():
    from experiments.cvs_phase1_repair import evaluate as e
    data=[dict(arm=a,model_seed=s,view=v,dimension='overall',accuracy=i/100+j/1000,macro_f1=i/100+j/1000)
          for i,a in enumerate(d.ARMS) for j,s in enumerate(d.SEEDS) for v in e.VIEWS]
    results=e.paired_results(data)
    assert len(results)==8*7*2
    assert results[0]['mean_difference']==pytest.approx(.01)
    assert results[0]['sd_difference']==pytest.approx(0)
    assert results[0]['positive_seeds']==4
