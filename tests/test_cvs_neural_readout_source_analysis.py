"""Synthetic arithmetic tests; no training artifacts or target metrics are read."""
import copy
import json
import math
import statistics as st

import pytest

from experiments.cvs_neural_readout_identity import analyze as module


@pytest.fixture
def rows():
    result = []
    for v, variant in enumerate(module.CANDIDATES):
        for s, seed in enumerate(sorted(module.SEEDS)):
            epochs = []
            for epoch in range(1, 201):
                item = dict(epoch=epoch, clean_ce=.8-.002*epoch+.01*s+.002*v,
                    source_val_ce=.9-.001*epoch+.03*s+.005*v,
                    source_val_accuracy=.6+.001*epoch+.005*v+.001*s,
                    source_val_worst_rx=.5+.001*epoch+.003*v+.002*s)
                if variant == 'readout_complex_attention':
                    item['source_val_accuracy'] += .004*s-.008
                    item['source_val_worst_rx'] -= .002*s
                if variant in module.VARIANTS:
                    item['readout_gradient_norm'] = 0 if epoch == 1 else .01*epoch+.1*s
                    item['readout_diagnostics'] = {'learned_readout': {'records': [
                        dict(block=block, packets=28,
                            relative_output_change_mean=.001*epoch+.01*s+.1*b,
                            attention_effective_tokens_mean=10+epoch*.1+s+b,
                            attention_entropy_mean=math.log(10+epoch*.1+s+b),
                            projection_norm=1+.002*epoch+.2*s+.3*b)
                        for b, block in enumerate(module.READOUT_BLOCKS)]}}
                epochs.append(item)
            result.append(dict(resolved=dict(variant=variant, model_seed=seed), epochs=epochs,
                               log_audit={'steps': 10000}))
    return result


def test_full_four_seed_ce_gap_and_descriptive_windows(rows):
    result = module.summarize_source_telemetry(rows)
    assert len(result['source_learning_by_seed']) == 16
    assert len(result['source_curve_summary']) == 800
    for record in result['source_learning_summary']:
        source = [r for r in rows if r['resolved']['variant'] == record['variant']]
        gaps = [r['epochs'][-1]['source_val_ce']-r['epochs'][-1]['clean_ce'] for r in source]
        assert record['CE_gap_mean'] == pytest.approx(st.mean(gaps))
        assert record['CE_gap_SD'] == pytest.approx(st.stdev(gaps))
        assert record['train_CE_late_delta_mean'] == pytest.approx(-.05)
        assert record['V_CE_late_delta_mean'] == pytest.approx(-.025)
        assert record['CE_gap_late_delta_mean'] == pytest.approx(.025)
    # The gap SD is computed from paired gaps, not by subtracting unrelated SDs.
    rows[1]['epochs'][-1]['source_val_ce'] += .2
    changed = module.summarize_source_telemetry(rows)['source_learning_summary'][0]
    assert changed['CE_gap_SD'] != pytest.approx(changed['V_CE_SD']-changed['train_CE_SD'])
    assert result['scope']['epoch_reselection'] is False
    assert result['scope']['target_results_read'] is False


def test_pairing_uses_registered_seeds_and_fixed_e200_without_mutation(rows):
    for row in rows:
        row['epochs'][0]['source_val_accuracy'] = .999
        row['epochs'][0]['source_val_worst_rx'] = .999
    rows.reverse()
    original = copy.deepcopy(rows)
    result = module.summarize_source_telemetry(rows)
    assert rows == original
    assert [r['model_seed'] for r in result['readout_pairwise_by_seed']] == sorted(module.SEEDS)
    for s, record in enumerate(result['readout_pairwise_by_seed']):
        assert record['epoch'] == 200
        assert record['V_delta_pp'] == pytest.approx(-.3+.4*s)
        assert record['worst_RX_delta_pp'] == pytest.approx(.3-.2*s)
        assert record['score_delta_pp'] == pytest.approx(.1*s)
    val = next(r for r in result['readout_pairwise_summary'] if r['metric'] == 'V')
    assert val['mean_delta_pp'] == pytest.approx(.3)
    assert val['paired_SD_pp'] == pytest.approx(st.stdev([-.3, .1, .5, .9]))
    assert val['positive_seeds'] == 3


def test_every_gradient_epoch_and_both_last_batch_branches_are_used(rows):
    result = module.summarize_source_telemetry(rows)
    assert len(result['readout_gradient_epochs']) == 1600
    assert len(result['readout_gradient_by_seed']) == 8
    assert len(result['readout_last_epoch']) == 16
    assert len(result['readout_last_epoch_summary']) == 4
    first = result['readout_gradient_by_seed'][0]
    assert first['epochs'] == 200 and first['audited_steps'] == 10000
    assert first['zero_epoch_means'] == 1
    assert first['full_run_epoch_mean'] == pytest.approx(st.mean([0]+[.01*x for x in range(2, 201)]))
    assert first['epoch_mean_max'] == 2
    assert first['E151_200_epoch_mean'] == pytest.approx(1.755)
    final = result['readout_last_epoch_summary'][0]
    assert final['relative_output_change_mean_mean'] == pytest.approx(.215)
    assert final['attention_effective_tokens_mean_mean'] == pytest.approx(31.5)
    assert final['projection_norm_mean'] == pytest.approx(1.7)
    assert all(r['packets'] == 28 and r['epoch'] == 200 for r in result['readout_last_epoch'])
    # Change an early epoch far outside the descriptive tail: full-run mean must respond.
    new = next(r for r in rows if r['resolved']['variant'] in module.VARIANTS)
    new['epochs'][7]['readout_gradient_norm'] += 10
    updated = module.summarize_source_telemetry(rows)['readout_gradient_by_seed'][0]
    assert updated['full_run_epoch_mean']-first['full_run_epoch_mean'] == pytest.approx(.05)
    assert updated['E151_200_epoch_mean'] == first['E151_200_epoch_mean']


@pytest.mark.parametrize('mutation', ['missing_seed', 'duplicate_seed', 'missing_epoch', 'duplicate_epoch',
                                     'nan_ce', 'inf_gradient', 'missing_branch', 'wrong_batch', 'bad_tokens'])
def test_incomplete_or_nonfinite_input_cannot_be_summarized_as_full_e200(rows, mutation):
    new = next(r for r in rows if r['resolved']['variant'] in module.VARIANTS)
    if mutation == 'missing_seed':
        rows.pop()
    elif mutation == 'duplicate_seed':
        rows[-1] = copy.deepcopy(rows[-2])
    elif mutation == 'missing_epoch':
        rows[0]['epochs'].pop(30)
    elif mutation == 'duplicate_epoch':
        rows[0]['epochs'][30]['epoch'] = 30
    elif mutation == 'nan_ce':
        rows[0]['epochs'][30]['clean_ce'] = float('nan')
    elif mutation == 'inf_gradient':
        new['epochs'][7]['readout_gradient_norm'] = float('inf')
    elif mutation == 'missing_branch':
        new['epochs'][0]['readout_diagnostics']['learned_readout']['records'].pop()
    elif mutation == 'wrong_batch':
        new['epochs'][-1]['readout_diagnostics']['learned_readout']['records'][0]['packets'] = 128
    elif mutation == 'bad_tokens':
        new['epochs'][-1]['readout_diagnostics']['learned_readout']['records'][0]['attention_effective_tokens_mean'] = 65
    with pytest.raises(ValueError):
        module.summarize_source_telemetry(rows)


def test_report_explains_measurement_limits_and_uses_no_step_extrema(rows):
    result = module.summarize_source_telemetry(rows)
    text = module.telemetry_report(result)
    assert '不是误差率差' in text and '50个batch均值的等权平均' in text
    assert '27000个验证样本' in text and '不重新选择epoch' in text
    assert '28个样本不能代表全源分布' in text and '因果结论' in text
    assert '不证明head之间的多样性' in text and '未测量head相似度或信道解耦' in text
    assert '未计算step极值或单分支梯度' in text
    assert 'complex_attention−attention' in text
    assert not any('step_max' in k or 'step_min' in k for r in result['readout_gradient_by_seed'] for k in r)
    # JSON output must not conceal missing/NaN arithmetic as a successful report.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('tokens,entropy', [(64+9e-4, math.log(64)+9e-6), (1-9e-6, -9e-7)])
def test_analysis_preserves_collector_fp32_attention_tolerances(rows, tokens, entropy):
    for row in rows:
        if row['resolved']['variant'] in module.VARIANTS:
            for record in row['epochs'][-1]['readout_diagnostics']['learned_readout']['records']:
                record['attention_effective_tokens_mean'] = tokens
                record['attention_entropy_mean'] = entropy
    result = module.summarize_source_telemetry(rows)
    for summary in result['readout_last_epoch_summary']:
        assert summary['attention_effective_tokens_mean_mean'] == tokens
        assert summary['attention_entropy_mean_mean'] == entropy


def test_analyze_preserves_completion_gate_before_writing(tmp_path, monkeypatch):
    evidence = tmp_path/'automation_reports/CV-SincNet'/module.RUN/'evidence'
    evidence.mkdir(parents=True)
    source = evidence/'source_research_complete.json'
    source.write_text('{}', encoding='utf-8')
    def reject(_):
        raise ValueError('synthetic incomplete source evidence')
    monkeypatch.setattr(module, 'validate_completed', reject)
    with pytest.raises(ValueError, match='incomplete source'):
        module.analyze(tmp_path)
    assert list(evidence.iterdir()) == [source]
    assert not (evidence.parent/'report.md').exists()
