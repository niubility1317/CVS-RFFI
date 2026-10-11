"""Interpret closed source trajectories without opening target scores or predictions."""
import argparse
import json
from pathlib import Path


def build(value):
    if (not value.get('source_only') or value.get('target_prediction_or_scores_read') is not False
            or value.get('errors') or not value.get('full_closed_source_step_epoch_CSV_warmup_and_stdout_read')):
        raise ValueError('Complete closed source evidence required; partial matrix is allowed')
    style, sensitivity = [], []
    for row in value['rows']:
        if row['status'] != 'COMPLETE' or row['steps'] != 4440:
            raise ValueError('A source row is incomplete')
        search = row['style_search_summary']
        if search:
            identity = row['style_identity']
            delta = search['eval_CE_delta']
            n = delta['observations']
            if n != 3996 or sum(delta[k] for k in ('positive', 'zero', 'negative')) != n:
                raise ValueError('Style observation count differs')
            style.append(dict(row_id=row['row_id'], arm=row['arm'], seed=row['initialization']['seed'],
                active_steps=n, eval_clean_CE=search['eval_clean_CE']['mean'],
                eval_selected_CE=search['eval_selected_CE']['mean'], eval_CE_delta=delta['mean'],
                harder_fraction=delta['positive']/n, easier_fraction=delta['negative']/n,
                train_prediction_change_pp=100*identity['style_prediction_change']['mean'],
                train_accuracy_delta_pp=100*identity['style_accuracy_delta']['mean'],
                train_margin_delta=identity['style_margin_delta']['mean'],
                extra_search_forwards=row['resources']['style_search_forwards'],
                extra_search_packets=row['resources']['style_search_L']))
        combined = row['combined_fishr_gradient_ratios']
        for level, stats in row['layer_summary'].items():
            if 'sensitivity_A/low_norm_rate' not in stats:
                continue  # The learned-probe control uses a different statistic.
            raw = stats['raw_fishr']
            gradient = combined[level+'/gradient_norm']
            ratio = combined[level+'/ratio_to_main_ce']
            if raw['observations'] != 999 or gradient['observations'] != 999:
                raise ValueError('Sensitivity computation count differs')
            sensitivity.append(dict(row_id=row['row_id'], arm=row['arm'], seed=row['initialization']['seed'], level=level,
                calls=raw['observations'], raw_term_mean=raw['mean'], raw_negative_fraction=raw['negative']/raw['observations'],
                low_norm_A_mean=stats['sensitivity_A/low_norm_rate']['mean'],
                low_norm_B_mean=stats['sensitivity_B/low_norm_rate']['mean'],
                weighted_gradient_mean=gradient['mean'], weighted_gradient_min=gradient['min'],
                gradient_ratio_median=ratio['median'], gradient_ratio_q90=ratio['q90'],
                fixed_coefficient=row['calibration']['fixed_coefficients'][level]))
    summary=dict(source_run=value['run_id'], source_read_at=value['read_at'],
        matrix_status=value['status'], source_complete_rows=len(value['rows']), source_steps=value['total_steps'],
        source_only=True, target_scores_read=False, active_training_changed=False,
        style=style, sensitivity=sensitivity)
    lines=['# 源域机制诊断：实际约束强度与风格难度', '',
        f"仅使用已闭合源训练记录：{len(value['rows'])} 行、{value['total_steps']:,} 次主更新；矩阵状态 {value['status']}。未读取测试评分、预测或目标 IQ，未调整在训任务。各行完整训练轨迹均被读取，梯度审计和每步机制计数的采样频率按原日志定义。", '',
        '## 风格是否形成判别困难', '',
        '|行|eval clean CE|eval 所选 CE|CE 增量|更难批次|更易批次|同掩码训练改判率|训练准确率差（百分点）|训练 margin 差|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in style:
        lines.append(f"|{r['row_id']}|{r['eval_clean_CE']:.5f}|{r['eval_selected_CE']:.5f}|{r['eval_CE_delta']:+.5f}|{100*r['harder_fraction']:.2f}%|{100*r['easier_fraction']:.2f}%|{r['train_prediction_change_pp']:.3f}%|{r['train_accuracy_delta_pp']:+.3f}|{r['train_margin_delta']:+.4f}|")
    lines += ['',
        '上述统计覆盖每行全部 3,996 个活跃批次。困难选择在四个风格候选中最大化整批 CE，候选集合没有 clean 恒等视图，因此“候选最难”并不保证比 clean 更难。这是现有候选定义的性质，不是未执行搜索。eval 中选择候选、train 中计算额外 CE，二者状态不同；训练改判与准确率差使用相同 dropout 掩码的配对前向。', '',
        '平均 CE 增加也不要求平均 margin 下降：CE 由完整 logits 分布计算，风格日志中的 margin 是正确 logit 与最大错误 logit 之差，两者口径不同，且整批均值会受少量困难样本影响。不能仅凭特征统计量变化或平均 CE 增量证明覆盖了未见接收机变化。每个搜索行额外执行 19,980 次 eval 前向、2,557,440 次 L 样本读出，另有 3,996 次带梯度的风格前向；这些开销须与实际困难程度一起解释。', '',
        '## Fishr 式主身份敏感度是否产生梯度', '',
        '|行|层|固定系数|有权重梯度范数均值|最小范数|相对同组 CE 中位数|相对 CE P90|低范数比例 A/B|原始项负值比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sensitivity:
        lines.append(f"|{r['row_id']}|{r['level']}|{r['fixed_coefficient']:.5g}|{r['weighted_gradient_mean']:.5g}|{r['weighted_gradient_min']:.5g}|{r['gradient_ratio_median']:.3f}|{r['gradient_ratio_q90']:.3f}|{r['low_norm_A_mean']:.3g}/{r['low_norm_B_mean']:.3g}|{100*r['raw_negative_fraction']:.2f}%|")
    lines += ['',
        '每个敏感度行含 999 次实际辅助步骤；原始项是独立 A/B 方差向量的有符号内积，有限批次允许负值。低范数比例只对应当前实现阈值，不是物理可辨识性的充分证明。表中梯度是联合 Fishr 项在相应参数组上的实际梯度，浅层约束也能影响后续层；不能把参数组梯度当成某一独立层损失的贡献。相对 CE 的比例受分母大小影响，必须同时看绝对范数。', '',
        '## 后续待检验问题', '',
        '本证据能检验约束是否执行、梯度是否非零及源 L 候选的判别难度，不能证明域泛化收益。现有两批固定实验继续完成，不因本诊断改变训练或重跑。后续若研究候选搜索，应先在合法源 L 的固定检查上比较“包含恒等候选的困难选择”和当前四候选规则，检验是否能避免选择更容易的视图；同时保留相同前向成本与 dropout 对照。这只是由源域证据提出的待检验问题，本轮未发布该变体，也未据此选择 checkpoint。', '']
    return summary, '\n'.join(lines)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,text=build(json.loads(a.source.read_text(encoding='utf-8')))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'source_mechanism_findings.md').write_text(text,encoding='utf-8')
    (a.output/'evidence/source_mechanism_findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',source_rows=result['source_complete_rows'],style_rows=len(result['style']),sensitivity_groups=len(result['sensitivity']),target_scores_read=False)))
