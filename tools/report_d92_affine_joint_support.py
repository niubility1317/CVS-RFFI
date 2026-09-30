"""Render the complete independently verified AffineJoint support results, without fitting."""
import argparse
import json
import math
from pathlib import Path

PATHS = ('R0', 'R_AFFINE_seq')
METRICS = ('A_old_accuracy', 'B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h',
    'adaptation_gain_B_minus_A', 'support_adaptation_B_minus_B0', 'total_old_accuracy_drop',
    'C_abs_new_old_gap', 'C_old_columns_accuracy', 'new_competition_loss')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def values(rows, **filters):
    selected = [row for row in rows if all(row.get(key) == value for key, value in filters.items())
        and row['metric'] in METRICS]
    result = {row['metric']: row for row in selected}
    require(len(selected) == len(result) == len(METRICS), 'Missing or duplicate report metric: '+repr(filters))
    for row in selected:
        require(type(row['measured_parent_count']) is int and row['measured_parent_count'] >= 0, 'Invalid measured parent count')
        require(row['mean'] is None or type(row['mean']) in (int, float) and math.isfinite(row['mean']), 'Invalid report mean')
    return result


def percent(value):
    return 'N/A' if value is None else f'{100*value:.4f}'


def number(value):
    return 'N/A' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


def assemble(summary, execution, *, source):
    require(summary['status'] == 'COMPLETE_AFFINE_JOINT_PROBE_VERIFIED'
        and summary['scope'] == 'SUPPORT_ONLY_AFFINE_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION', 'Unverified result scope')
    require(summary['coverage']['episodes'] == 160 and summary['coverage']['sequence_paths'] == 1800, 'Incomplete support result')
    require(summary['query_rows_used'] == summary['source_rows_used'] == 0, 'Forbidden input use')
    require(summary['actual_A'] is summary['adaptation_gain_B_minus_A'] is None, 'Unavailable ground A was fabricated')
    require(summary['old_class_count'] == 6 and summary['algorithm']['schema'] == 'd92_affine_joint_local_ridge_v1' and summary['algorithm']['free_intercept'] is True, 'Method or old-class mismatch')
    require(execution['status'] == 'VERIFIED' and execution['runtime_commit'] == summary['release_commit'], 'Independent analysis binding mismatch')
    stats = summary['statistics']
    overall = {path: values(stats['overall'], diagnostic='oof', population='new_present', path=path) for path in PATHS}
    require(all(overall[path]['B_old_accuracy']['measured_parent_count'] == 96 for path in PATHS), 'Incomplete measured new-present population')
    means = {path: {metric: overall[path][metric]['mean'] for metric in METRICS} for path in PATHS}
    deltas = {metric: None if means['R0'][metric] is None or means['R_AFFINE_seq'][metric] is None
        else means['R_AFFINE_seq'][metric]-means['R0'][metric] for metric in METRICS}
    matrix = []
    for diagnostic in ('oof', 'proxy'):
        for path in PATHS:
            for k in (1, 5, 10, 20):
                for new_count in (0, 2, 5, 10, 20):
                    cell = values(stats['by_k_new_count'], diagnostic=diagnostic, path=path, k=k, new_count=new_count)
                    base = values(stats['by_k_new_count'], diagnostic=diagnostic, path='R0', k=k, new_count=new_count)
                    require(cell['A_old_accuracy']['mean'] is cell['adaptation_gain_B_minus_A']['mean'] is None, 'Cell fabricated A')
                    require(cell['B_old_accuracy']['measured_parent_count'] == (0 if k == 1 else 8), 'Missing matrix parents')
                    if k == 1:
                        require(all(cell[metric]['mean'] is None for metric in METRICS), 'K1 holdout was fabricated')
                    if new_count == 0:
                        require(all(cell[metric]['mean'] is None for metric in ('C_new_accuracy', 'C_h', 'C_abs_new_old_gap')), 'Invented new classes')
                    matrix.append(dict(diagnostic=diagnostic, path=path, k=k, old_count=6,
                        new_count=new_count, registered_count=6+new_count,
                        measured_parents=cell['B_old_accuracy']['measured_parent_count'],
                        means={metric: cell[metric]['mean'] for metric in METRICS},
                        deltas={metric: None if cell[metric]['mean'] is None or base[metric]['mean'] is None
                            else cell[metric]['mean']-base[metric]['mean'] for metric in METRICS}))
    lines = ['# AffineJoint 与 BranchLocalRidge：完整 support 优化结果', '',
        '**完整 160 个配置已独立核验。以下是 support 持出诊断，不能写成 query 准确率或新增独立数据验证。**', '',
        f'run：`{summary["run_id"]}`。训练版本：`{summary["release_commit"]}`；独立分析版本：`{execution["commit"]}`。', '',
        '只有预登记的 R0 和 R_AFFINE_seq 两条路径。本报告描述完整结果，不自动挑选局部最优配置或晋级方法。', '',
        '旧类为 6 个，新增 0/2/5/10/20 个，总注册 6/8/11/16/26 个；K=1/5/10/20 表示每类原始 support 数。两模型 seed × 两 cohort × 两 receiver/scene 组合，共 160 个配置。固定 practical residual/post_sync/noeq、fs=25MHz，地面 source-only Phase1 不变。', '',
        'A 是未适应地面模型的旧类准确率，本 support 试验没有测量，因此 A 与 B−A 均为 N/A。B0/R0 已经是 support 分类器，不能代替 A。B 只用旧 support，C 继承本配置实际 B 函数和 U_B，再使用新旧 support 注册；全部已注册类统一竞争。', '',
        'oof 按同一配置、同一旧 support 与物理旧 held 样本配对 B/C。内层 held 是合法训练监督，外层 held 只评估冻结状态。总体统计采用 K5/10/20、新增2/5/10/20 的 96 个可测配置。H 和绝对新旧差都先逐配置计算再平均；总体新旧均值之差不能代替平均绝对差。', '',
        'true K1 仍执行完整 B/C 闭式头，但无独立持出准确率。proxy 使用 K>1 support 的各单样本 anchor；trainK1 不训练 adapter，仍拟合闭式头。这是同一 support 库的诊断，不是新的独立 K1 证据。重复旧 support 与 B 结果不是独立观测，未新增独立置信区间。', '',
        '| 方法 | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册旧类下降 | 平均绝对新旧差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for path in PATHS:
        lines.append('| '+' | '.join([path]+[percent(means[path][metric]) for metric in METRICS[:9]])+' |')
    lines += ['', '准确率与 H 用%；差值用百分点。', '',
        '| AffineJoint 相对 R0 | B旧 | C旧 | C新 | H | 注册下降变化 | 绝对新旧差变化 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| 差值 | '+' | '.join(percent(deltas[m]) for m in ('B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy',
            'C_h', 'total_old_accuracy_drop', 'C_abs_new_old_gap'))+' |', '',
        '注册下降改善应同时检查 B 与 C：B 本身下降也会缩小 B−C，不能把全部变化解释为遗忘减少。10/1/3 个百分点是用户的理想方向，未作为自动晋级门槛；本轮无法检验 B−A≥10。', '',
        f'AffineJoint 的 C 旧类列内诊断为 {percent(means["R_AFFINE_seq"]["C_old_columns_accuracy"])}%，全类竞争为 {percent(means["R_AFFINE_seq"]["C_old_accuracy"])}%，新类竞争损失 {percent(means["R_AFFINE_seq"]["new_competition_loss"])} 个百分点。列内诊断只解释错误，不作为部署路由。', '',
        '完整 K×新增类数表如下。所有 A/B−A 为 N/A；外层持出后的实际训练 K 见训练分层，不能将 K20 写成训练使用全部 20 样本。', '']
    for diagnostic in ('oof', 'proxy'):
        lines += [('外层 oof' if diagnostic == 'oof' else '单样本 anchor proxy'), '',
            '| 方法 | K | 旧类 | 新增 | 总类 | 可测配置 | A旧 | B旧 | C旧 | C新 | H | B−B0 | 注册下降 | 绝对新旧差 | ΔH/R0 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cell in matrix:
            if cell['diagnostic'] != diagnostic:
                continue
            lines.append('| '+' | '.join([cell['path'], str(cell['k']), '6', str(cell['new_count']),
                str(cell['registered_count']), str(cell['measured_parents'])]+[percent(cell['means'][metric]) for metric in
                    ('A_old_accuracy', 'B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h',
                     'support_adaptation_B_minus_B0', 'total_old_accuracy_drop', 'C_abs_new_old_gap')]
                +[percent(cell['deltas']['C_h'])])+' |')
        lines.append('')
    selected_metrics = ('B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    for table, dimensions, label in (('by_receiver_scene', ('cohort', 'receiver', 'scenario'), '接收机与场景'),
        ('by_model_cohort', ('model_seed', 'cohort'), '模型 seed 与 cohort')):
        groups = {}
        for row in stats[table]:
            if row['diagnostic'] != 'oof' or row['new_count'] == 0 or row['mean'] is None or row['metric'] not in selected_metrics:
                continue
            key = tuple(row[k] for k in dimensions)+(row['path'], row['metric'])
            total, count = groups.get(key, (0., 0)); n = row['measured_parent_count']
            groups[key] = total+n*row['mean'], count+n
        keys = sorted({key[:-2] for key in groups})
        require(len(keys) == 4, 'Incomplete report strata: '+table)
        lines += [label+'（合并可测 K 与新增类数）', '', '| 分层 | 方法 | B旧 | C旧 | C新 | H | 注册下降 | 绝对新旧差 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for key in keys:
            for path in PATHS:
                value = []
                for metric in selected_metrics:
                    total, count = groups[key+(path, metric)]
                    require(count == 24, 'Incomplete measured stratum')
                    value.append(total/count)
                lines.append('| '+' | '.join(['/'.join(map(str, key)), path]+[percent(v) for v in value])+' |')
        lines.append('')
    r, c = summary['resources'], summary['coverage']
    lines += ['数学结构与实际成本', '',
        'AffineJoint 保留实际 B 先验、旧 τ/γ 和当前 U 重映射；每个闭式头以 E=Y−M_B 合并求解截距：A=K+I，F=A⁻¹E，z=A⁻¹1，s=1ᵀz，b=1ᵀF/s，α=F−zb，f=m_B+kα+b。实际计算使用一次 Cholesky、两次 triangular solve，不显式求逆。完整截距伴随包含 g_b=sum_rows(G)，pooled class-RMS CE 与 0.5||Z||² 通过最终 LocalRidge 头联合微调。详见[截距推导](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)和[独立核心](D92_AFFINE_JOINT_CORE_20261001.md)。', '',
        '固定宽度8的字典，实际保留秩 r≤8，最多736r≤5888个 adapter 梯度训练坐标；闭式拟合的分类头另计。U/V0合计94208 B。函数近端控制当前 support 上投影前的均方增量，不提供 query 准确率保持保证。每个头另解析拟合 nC 个系数和 C 个截距数组值，截距独立contrast为 C−1；它们不计入adapter梯度坐标。C 不继承后验协方差且重用旧数据，不能称为精确顺序 Bayes。自由截距解除强制旧参考均值限制，不保证准确率提升；在同一 raw kernel/尺度/U/support/prior 下 q 只是等价表示，旧尺度估计仍有统计作用。b 的变化依赖该表示，不能单独归因为准确率变化。', '',
        '| 实际资源项 | 数值 |', '|---|---:|']
    resource_items = (
        ('完整运行墙钟 / s', 'run_wall_seconds'), ('最大 lane 峰值 RSS / B', 'maximum_lane_peak_rss_bytes'),
        ('R0 拟合累计 / s', 'base_fit_fit_seconds_sum'), ('R0 评分累计 / s', 'base_fit_score_seconds_sum'),
        ('AffineJoint 准备累计 / s', 'ajlr_preparation_prepare_seconds_sum'), ('AffineJoint 拟合累计 / s', 'candidate_fit_fit_seconds_sum'),
        ('AffineJoint 评分累计 / s', 'candidate_fit_score_seconds_sum'),
        ('最大实际保留数值状态 / B', 'candidate_fit_maximum_persistent_state_bytes'),
        ('最大推理必需数值缓冲区 / B', 'candidate_fit_maximum_deployment_numeric_state_bytes'),
        ('最大准备数值状态 / B', 'ajlr_preparation_maximum_prepared_numeric_state_bytes'),
        ('额外地面统计 / B', 'additional_ground_statistics_bytes'),
        ('完整部署包 / B', 'deployment_package_bytes'), ('新增传输 / B', 'incremental_transmission_bytes'),
        ('GPU 峰值 / B', 'peak_gpu_memory_bytes'))
    for label, key in resource_items:
        lines.append('| '+label+' | '+number(r.get(key))+' |')
    lines += ['', '运行在 N607 CPU 两 lane、每 lane 两 BLAS 线程、float64。硬件原值与分层成本见 summary.json/CSV。持续保留状态包含训练坐标和缓存；推理数值缓冲区是单独口径，均不含 Python 开销、冻结 Phase1 或完整部署包。累计阶段工作和嵌套计时不能直接相加作为墙钟或单次星载时延。两路径 AffineJoint 与三路径 FCR 工作不同，不能仅用总时间证明 adapter 更快。真实星载训练/推理、完整模型传输未测为 N/A。', '',
        '| 实际执行计数 | 次数 |', '|---|---:|']
    for key in ('ajlr_preparation_count', 'ajlr_stage_count', 'trained_ajlr_stage_count', 'optimizer_steps',
        'rejected_trial_count', 'inner_objective_evaluation_count', 'head_fit_count', 'factorization_count',
        'baseline_head_triangular_solve_count', 'baseline_effective_df_triangular_solve_count',
        'head_triangular_solve_count', 'prior_triangular_solve_count', 'ce_adjoint_solve_count',
        'derivative_triangular_solve_count', 'latent_svd_count',
        'head_triangular_rhs_count', 'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
        'prior_triangular_rhs_count', 'prior_triangular_rhs_element_count', 'prior_triangular_dense_work_unit_count',
        'derivative_triangular_rhs_count', 'derivative_triangular_rhs_element_count', 'derivative_triangular_dense_work_unit_count',
        'intercept_fit_count', 'prior_intercept_fit_count', 'intercept_addition_count', 'prior_intercept_addition_count'):
        lines.append('| '+key+' | '+number(c[key])+' |')
    lines += ['', '前向每次 triangular 使用 C+1 列 RHS，伴随使用 C 列；n²·RHS 是尺寸工作代理，不能当作实测 FLOP 或星载耗时。所有拒绝试探计费，停止不等于收敛。reference distance 是 raw distance 工作的解释子集，不重复相加。完整初末状态、梯度、接受/拒绝、旧 prior 与核/α 均保留；只读训练诊断随后解释实际执行，不从内部拟合准确率推断独立性能。', '',
        '复用既有 p2_min_v1/VALIDATED_ONCE support 身份，没有改变 received IQ、物理 ID、信道或角色划分。未加载历史目标 adapter/head，仅当前 B→C 继承；源样本、源逐记录特征、query 拟合与反馈均不使用。地面统计输入为0 B，不把软件发布包字节当作卫星方法数据传输。', '',
        f'完整独立分析来源：`{source}`。逐配置与必要分层保留在同目录 CSV；NPZ 保留原路径。', '']
    interpretation = dict(status='COMPLETE_SUPPORT_REPORT_RENDERED_DECISION_PENDING', run_id=summary['run_id'],
        scope=summary['scope'], declared_main_path='R_AFFINE_seq', automatic_promotion=False,
        actual_A=None, overall=means, delta_vs_R0=deltas, matrix=matrix, resources=r, source=source,
        runtime_commit=execution['runtime_commit'], analysis_commit=execution['commit'])
    return '\n'.join(lines), interpretation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--interpretation', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists() and not args.interpretation.exists(), 'Report output exists; preserve and reconcile')
    report, interpretation = assemble(json.loads(args.summary.read_text(encoding='utf-8')),
        json.loads(args.execution.read_text(encoding='utf-8')), source=str(args.summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.interpretation.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        stream.write(report)
    with args.interpretation.open('x', encoding='utf-8') as stream:
        json.dump(interpretation, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(report=str(args.report), matrix_rows=len(interpretation['matrix']), status=interpretation['status'])))


if __name__ == '__main__':
    main()
