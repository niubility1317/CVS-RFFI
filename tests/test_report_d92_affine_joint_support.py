"""Protect A/B/C report scope, complete cells and parent-first H/gap values."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import report_d92_affine_joint_support as report


def fixture():
    statistics = {name: [] for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    mean_by_path = {
        'R0': dict(zip(report.METRICS, (None, .70, .60, .80, .63, None, 0., .10, .25, .69, .09))),
        'R_AFFINE_seq': dict(zip(report.METRICS, (None, .75, .61, .78, .65, None, .05, .14, .20, .71, .10)))}
    def rows(values, count, **dimensions):
        return [dict(dimensions, metric=key, mean=value, measured_parent_count=0 if value is None else count)
            for key, value in values.items()]
    for path in report.PATHS:
        statistics['overall'].extend(rows(mean_by_path[path], 96, diagnostic='oof', population='new_present', path=path))
        for diagnostic in ('oof', 'proxy'):
            for k in (1, 5, 10, 20):
                for new_count in (0, 2, 5, 10, 20):
                    values = deepcopy(mean_by_path[path])
                    if new_count == 0:
                        for key in ('C_new_accuracy', 'C_h', 'C_abs_new_old_gap'):
                            values[key] = None
                        values['C_old_accuracy'] = values['C_old_columns_accuracy'] = values['B_old_accuracy']
                        values['total_old_accuracy_drop'] = values['new_competition_loss'] = 0.
                    if k == 1:
                        values = dict.fromkeys(report.METRICS)
                    statistics['by_k_new_count'].extend(rows(values, 0 if k == 1 else 8,
                        diagnostic=diagnostic, path=path, k=k, new_count=new_count))
        for k in (5, 10, 20):
            for new_count in (2, 5, 10, 20):
                for cohort in ('rx1', 'rx3'):
                    for scene in ('practical_high', 'practical_low_urban'):
                        statistics['by_receiver_scene'].extend(rows(mean_by_path[path], 2,
                            diagnostic='oof', path=path, cohort=cohort, receiver=cohort,
                            scenario=scene, k=k, new_count=new_count))
                    for model in (2026092701, 2026092702):
                        statistics['by_model_cohort'].extend(rows(mean_by_path[path], 2,
                            diagnostic='oof', path=path, cohort=cohort, model_seed=model, k=k, new_count=new_count))
    counters = ('episodes', 'sequence_paths', 'ajlr_preparation_count', 'ajlr_stage_count', 'trained_ajlr_stage_count',
        'optimizer_steps', 'rejected_trial_count', 'inner_objective_evaluation_count', 'head_fit_count', 'factorization_count',
        'baseline_head_triangular_solve_count', 'baseline_effective_df_triangular_solve_count',
        'head_triangular_solve_count', 'prior_triangular_solve_count', 'ce_adjoint_solve_count',
        'derivative_triangular_solve_count', 'latent_svd_count',
        'head_triangular_rhs_count', 'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
        'prior_triangular_rhs_count', 'prior_triangular_rhs_element_count', 'prior_triangular_dense_work_unit_count',
        'derivative_triangular_rhs_count', 'derivative_triangular_rhs_element_count', 'derivative_triangular_dense_work_unit_count',
        'intercept_fit_count', 'prior_intercept_fit_count', 'intercept_addition_count', 'prior_intercept_addition_count')
    coverage = dict.fromkeys(counters, 0); coverage.update(episodes=160, sequence_paths=1800)
    summary = dict(status='COMPLETE_AFFINE_JOINT_PROBE_VERIFIED',
        scope='SUPPORT_ONLY_AFFINE_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION',
        run_id='synthetic-report-not-an-experiment', release_commit='a'*40, coverage=coverage,
        query_rows_used=0, source_rows_used=0, actual_A=None, adaptation_gain_B_minus_A=None,
        old_class_count=6, algorithm=dict(schema='d92_affine_joint_local_ridge_v1', free_intercept=True),
        resources=dict(run_wall_seconds=10., additional_ground_statistics_bytes=0), statistics=statistics)
    execution = dict(status='VERIFIED', runtime_commit='a'*40, commit='b'*40)
    return summary, execution


def test_full_80_cells_and_H_absolute_gap_use_verified_parent_statistics():
    summary, execution = fixture()
    text, data = report.assemble(summary, execution, source='/synthetic/summary.json')
    assert len(data['matrix']) == 80 and data['automatic_promotion'] is False
    assert data['overall']['R_AFFINE_seq']['C_h'] == .65
    assert data['overall']['R_AFFINE_seq']['C_abs_new_old_gap'] == .20
    assert data['delta_vs_R0']['C_h'] == pytest.approx(.02)
    assert data['overall']['R_AFFINE_seq']['C_h'] != pytest.approx(2*.61*.78/(.61+.78))
    assert data['overall']['R_AFFINE_seq']['C_abs_new_old_gap'] != pytest.approx(abs(.61-.78))
    assert 'A 与 B−A 均为 N/A' in text and '5888' in text and '实际保留数值状态' in text
    assert '自动晋级门槛' in text and '新增独立数据验证' in text


@pytest.mark.parametrize('change', ['partial', 'query', 'source', 'A', 'execution', 'missing_cell', 'duplicate_cell', 'K1', 'missing_stratum'])
def test_partial_unsafe_or_fabricated_results_cannot_be_rendered(change):
    summary, execution = fixture()
    if change == 'partial': summary['coverage']['episodes'] = 159
    if change == 'query': summary['query_rows_used'] = 1
    if change == 'source': summary['source_rows_used'] = 1
    if change == 'A': summary['actual_A'] = .50
    if change == 'execution': execution['runtime_commit'] = 'c'*40
    if change == 'missing_cell': summary['statistics']['by_k_new_count'].pop()
    if change == 'duplicate_cell': summary['statistics']['by_k_new_count'].append(deepcopy(summary['statistics']['by_k_new_count'][0]))
    if change == 'K1':
        row = next(r for r in summary['statistics']['by_k_new_count'] if r['k'] == 1 and r['metric'] == 'B_old_accuracy')
        row.update(mean=.50, measured_parent_count=8)
    if change == 'missing_stratum':
        summary['statistics']['by_receiver_scene'] = [r for r in summary['statistics']['by_receiver_scene'] if r['cohort'] == 'rx3']
    with pytest.raises(ValueError):
        report.assemble(summary, execution, source='/synthetic/summary.json')


def test_affine_formula_and_analytic_cost_are_not_reported_as_no_intercept_or_SGD():
    summary, execution = fixture()
    text, _ = report.assemble(summary, execution, source='/synthetic/summary.json')
    assert 'α=F−zb' in text and 'g_b=sum_rows(G)' in text
    assert 'C+1 列 RHS' in text and '不能当作实测 FLOP' in text
    assert '不计入adapter梯度坐标' in text and '没有自由截距' not in text
    summary['algorithm']['free_intercept'] = False
    with pytest.raises(ValueError):
        report.assemble(summary, execution, source='/synthetic/summary.json')
