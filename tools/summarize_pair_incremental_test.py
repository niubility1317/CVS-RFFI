"""Validate and render the completed-eight diagnostic evidence without new inference."""
import argparse
import csv
import json
from pathlib import Path
import statistics

SCENES = ['leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pooled(row, dimension, group):
    cells = [row['metrics'][s]['by_' + dimension][str(group)] for s in SCENES]
    return 100 * sum(x['correct'] for x in cells) / sum(x['total'] for x in cells)


def main(root):
    data = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    evidence = json.loads((root / 'execution_evidence.json').read_text(encoding='utf-8'))
    training = {x['row_id']: x for x in json.loads((root / 'training_evidence.json').read_text(encoding='utf-8'))}
    runtime = {x['row_id']: x for x in evidence['predictions']}
    meta = data['metadata']
    assert len(data['rows']) == len(json.loads((root / 'manifest.json').read_text())['rows']) and evidence['all_sealed_before_scoring']
    summary, breakdown, classes = [], [], []
    lookup = {}
    for row in data['rows']:
        row_id = row['row_id']
        lookup[row_id] = row
        assert runtime[row_id]['n'] == meta['n'] == 168000
        assert runtime[row_id]['query_state_unchanged'] and runtime[row_id]['truth_access'] is False
        assert data['scored_at_unix'] > runtime[row_id]['sealed_at_unix']
        for scenario, metric in row['metrics'].items():
            cm = metric['confusion_matrix']
            assert sum(map(sum, cm)) == metric['total']
            assert sum(cm[i][i] for i in range(6)) == metric['correct']
            assert abs(metric['accuracy'] - 100 * metric['correct'] / metric['total']) < 1e-9
            for dimension in ['rx', 'day', 'rx_day']:
                groups = metric['by_' + dimension]
                assert sum(v['total'] for v in groups.values()) == metric['total']
                assert sum(v['correct'] for v in groups.values()) == metric['correct']
                for group, m in groups.items():
                    breakdown.append({'row_id': row_id, 'scenario': scenario, 'dimension': dimension, 'group': group,
                                      'correct': m['correct'], 'total': m['total'], 'accuracy_percent': m['accuracy']})
            for c in metric['per_class']:
                classes.append({'row_id': row_id, 'scenario': scenario, 'class_id': c['class_id'],
                                'tx_label': meta['class_id_to_tx'][c['class_id']], 'correct': c['correct'],
                                'total': c['total'], 'accuracy_percent': c['accuracy']})
        vals = {s: row['metrics'][s]['accuracy'] for s in SCENES}
        rx_values = {rx: pooled(row, 'rx', rx) for rx in row['metrics']['clean']['by_rx']}
        worst_rx = min(rx_values, key=rx_values.get)
        resource = training[row_id]['resources']
        summary.append({'row_id': row_id, 'clean': row['metrics']['clean']['accuracy'], **vals,
                        'leo_equal_scene_mean': row['leo_equal_scene_mean'], 'leo_scene_floor': row['leo_scene_floor'],
                        'worst_leo_receiver': worst_rx, 'worst_leo_receiver_acc': rx_values[worst_rx],
                        'train_hours': resource['wall_time_seconds'] / 3600,
                        'train_peak_allocated_gib': resource['peak_cuda_memory_allocated_bytes'] / 2**30,
                        'prediction_seconds_for_both_views': runtime[row_id]['seconds'],
                        'source_v_clean': training[row_id]['metrics_last']['val_tx_acc'],
                        'source_v_leo_mean': training[row_id]['metrics_last']['stage_source_val_sat_mean_tx']})
    deltas = []
    for row in data['rows']:
        if row['row_id'].startswith('MATCHED_ZERO_'):
            continue
        control_id = 'MATCHED_ZERO_S' + row['row_id'].rsplit('_S', 1)[1]
        control = lookup[control_id]
        deltas.append({'row_id': row['row_id'], 'control': control_id,
                       'clean_delta_pp': row['metrics']['clean']['accuracy'] - control['metrics']['clean']['accuracy'],
                       **{s + '_delta_pp': row['metrics'][s]['accuracy'] - control['metrics'][s]['accuracy'] for s in SCENES},
                       'leo_mean_delta_pp': row['leo_equal_scene_mean'] - control['leo_equal_scene_mean']})
    csv_file(root / 'summary.csv', summary)
    csv_file(root / 'paired_deltas.csv', deltas)
    csv_file(root / 'receiver_day_breakdown.csv', breakdown)
    csv_file(root / 'per_class.csv', classes)
    families = ['MATCHED_ZERO', 'A_POINT', 'ASYMMETRIC', 'POINT_MEMORY', 'TANGENT', 'ROUTE', 'TANGENT_ROUTE', 'B_SAFE']
    groups = []
    for family in families:
        items = [r for r in summary if r['row_id'].rsplit('_S', 1)[0] == family]
        paired = [r for r in deltas if r['row_id'].rsplit('_S', 1)[0] == family]
        if not items:
            continue
        values = [r['leo_equal_scene_mean'] for r in items]
        groups.append({'method': family, 'completed_seeds': len(items),
                       'clean_mean': statistics.mean(r['clean'] for r in items),
                       'leo_mean': statistics.mean(values),
                       'leo_seed_sample_std': statistics.stdev(values) if len(values) > 1 else None,
                       'paired_leo_mean_delta_pp': statistics.mean(r['leo_mean_delta_pp'] for r in paired) if paired else 0.0,
                       'paired_clean_mean_delta_pp': statistics.mean(r['clean_delta_pp'] for r in paired) if paired else 0.0})
    csv_file(root / 'method_summary.csv', groups)
    checked = json.loads((root / 'current_training_status.json').read_text(encoding='utf-8'))
    lines = ['# 已完成19行目标域诊断测试结果', '',
             '训练状态复核时间（UTC）：' + checked['checked_at_utc'] + '。', '',
             '状态：VERIFIED，19/19行已完成预测和独立评分。初始训练状态为19行完成、4行运行、1行技术失败；本报告只测试开始核实时已完成的19行，不自动扩大矩阵。', '',
             '本次新增11行测试，复用原8行预测，合并19行结果。此前约98.7%/96%的数字是source验证结果，不能替代本次目标接收机测试。', '',
             '## 测试口径', '',
             '- 固定E200最终学生checkpoint，FP32，关闭TF32；所有模型权重键严格匹配，禁止适配及query状态更新。',
             '- ManySig六个已知TX：' + '、'.join(meta['class_id_to_tx']) + '；equalized=1，中心裁剪、长度256。',
             '- source接收机索引1/3/4/6/8，日期1/2/3；target接收机索引0/2/5/7/9/10/11，日期0/1/2/3。每目标接收机24,000个样本，每天42,000个；日期0同时是未见日期。',
             '- 全量168,000个物理样本；clean对照168,000个。一个physical仅对应一次LEO接收观测，19行共享完全相同的received IQ。三场景按opaque ID固定分配、互斥，数量分别为56,164/56,095/55,741；seed=2027。',
             '- 场景随机分配独立于类别和模型。每个场景准确率在其自身子集计算；LEO均值是三种场景准确率等权平均，不把同一physical重复成三份。',
             '- 19行prediction全部固定后，独立scorer才按opaque ID连接truth。168,000个ID无重复、无缺失，模型持久状态不变；scorer时间晚于全部prediction固定时间。',
             '- 用户明确要求提前测试完成行；本次作为诊断，不回流选择、调参或重训，不作默认晋升。没有新类注册、unknown或适配，因此这些指标不适用。', '',
             '## 逐行实测准确率', '',
             '|行|Clean|晴空弱信道|低仰角弱信道|雨衰弱信道|LEO等权均值|', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["row_id"]}|{r["clean"]:.4f}%|{r[SCENES[0]]:.4f}%|{r[SCENES[1]]:.4f}%|{r[SCENES[2]]:.4f}%|{r["leo_equal_scene_mean"]:.4f}%|')
    lines += ['', '## 方法组汇总', '', '均值仅覆盖本轮已完成seed；标准差是seed间样本标准差，单seed不计算。配对差值按每行对应同seed的MATCHED_ZERO计算，不混用全体对照均值。B_SAFE另有EMA版本混杂，禁止作为纯机制收益。', '',
              '|方法|已完成seed数|Clean均值|LEO均值|LEO seed标准差|配对LEO差值/百分点|', '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        sd = '—' if g['leo_seed_sample_std'] is None else f'{g["leo_seed_sample_std"]:.4f}'
        lines.append(f'|{g["method"]}|{g["completed_seeds"]}|{g["clean_mean"]:.4f}%|{g["leo_mean"]:.4f}%|{sd}|{g["paired_leo_mean_delta_pp"]:+.4f}|')
    lines += ['', '## 同seed相对MATCHED_ZERO的差值', '', '|行|Clean差值/百分点|LEO均值差值/百分点|', '|---|---:|---:|']
    for r in deltas:
        lines.append(f'|{r["row_id"]}|{r["clean_delta_pp"]:+.4f}|{r["leo_mean_delta_pp"]:+.4f}|')
    lines += ['', '当前所有比较均为诊断证据，不选择冠军，不将结果用于该确认集后续调参或重跑。B_SAFE采用r3修复版EMA，其与旧版行之间的差异混合了代码修复和机制效果，不能作为严格消融。', '',
              '## 每接收机LEO准确率', '',
              '以下为每接收机三场景合并的正确数/样本数；它与上表的三场景等权均值定义不同。接收机数字均为数据集索引。', '',
              '|行|RX0|RX2|RX5|RX7|RX9|RX10|RX11|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['rows']:
        lines.append('|' + row['row_id'] + '|' + '|'.join(f'{pooled(row,"rx",g):.3f}%' for g in [0,2,5,7,9,10,11]) + '|')
    lines += ['', '## 每天LEO准确率', '', '|行|Day0（未见日期）|Day1|Day2|Day3|', '|---|---:|---:|---:|---:|']
    for row in data['rows']:
        lines.append('|' + row['row_id'] + '|' + '|'.join(f'{pooled(row,"day",g):.3f}%' for g in range(4)) + '|')
    lines += ['', '## 每类LEO准确率', '', '|行|' + '|'.join(meta['class_id_to_tx']) + '|', '|---|' + '---:|' * 6]
    for row in data['rows']:
        values = []
        for c in range(6):
            slots = [row['metrics'][s]['per_class'][c] for s in SCENES]
            values.append(100 * sum(x['correct'] for x in slots) / sum(x['total'] for x in slots))
        lines.append('|' + row['row_id'] + '|' + '|'.join(f'{x:.3f}%' for x in values) + '|')
    lines += ['', '## 资源', '', '训练耗时受同GPU并发与任务完成释放资源影响，不是独占速度比较。预测耗时包含clean和received共336,000次样本前向及输出整理，不含一次性数据builder和模型加载；不能直接当作单样本在线延迟。', '',
              '|行|训练小时|训练峰值已分配GiB|预测秒（旧行复用原记录）|', '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["row_id"]}|{r["train_hours"]:.2f}|{r["train_peak_allocated_gib"]:.2f}|{r["prediction_seconds_for_both_views"]:.2f}|')
    lines += ['', '## 执行与验证', '',
              '实际执行代码提交ce883451f6a4feecfacce4555173039bf9bf022d，Git远端OID已读回一致。release/runs名称phase1_adv3b02_completed19_test_20260907_v1，普通账户N607，GPU1，队列PID3472453；最终done.json与results.json独立读回确认SCORED19。',
              '7项本地协议/评分负测通过，新增reuse_root逻辑的独立P0/P1审查通过。新增11个checkpoint真实无query smoke均通过；原8行引用先前已验证的prediction。沿用首次8行测试已验证的完整FP32和NumPy兼容接口，本次未修改训练、checkpoint或测试观测。',
              '后处理独立核对每行样本覆盖、混淆矩阵总数/对角线、逐接收机/日期分组与总体计数一致，以及评分晚于全部预测固定。19行共76组clean/LEO指标验证通过；复用8行prediction，新增11行全部prediction固定后评分。',
              'B_SAFE392005仍为E109连续两轮无优化器更新的技术失败；没有最终checkpoint，不补造测试数据。本次启动时另4行保持运行。原r3共享EMA缓存修复与旧21行的混合版本仍为NO_PROMOTION。', '',
              '## 数据文件', '',
              '- summary.csv：19行总表。', '- method_summary.csv：方法组均值、seed标准差与配对差值。', '- paired_deltas.csv：16个同seed对照差值。',
              '- receiver_day_breakdown.csv：逐场景、接收机、日期及接收机×日期，共2,964行。',
              '- per_class.csv：逐场景逐类，共456行。',
              '- results.json：完整正确数/总数、全部混淆矩阵和分组结果。',
              '- execution_evidence.json：prediction固定时间、兼容性和query状态审计。',
              '- training_evidence.json：19行全部E1—E200的已提取source验证/资源字段；不作为完整loss根因分析。', '']
    (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'validated_rows':len(summary), 'breakdown_rows':len(breakdown), 'class_rows':len(classes), 'paired_deltas':deltas,
                      'worst_receivers':[{k:r[k] for k in ['row_id','worst_leo_receiver','worst_leo_receiver_acc']} for r in summary]}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
