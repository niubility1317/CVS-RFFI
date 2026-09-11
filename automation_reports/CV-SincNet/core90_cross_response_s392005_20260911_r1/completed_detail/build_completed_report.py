"""Build descriptive completed-row tables from fully parsed local evidence."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ('U0', 'U2', 'Ux')
SCENES = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')
TITLES = dict(zip(SCENES, ('clean', 'LEO晴空弱信道', 'LEO低仰角弱信道', 'LEO雨衰弱信道')))
GROUPS = {'test_unseen_day_seen_rx':'未知日期、已见RX', 'test_seen_day_unseen_rx':'已见日期、未知RX',
          'test_unseen_day_unseen_rx':'未知日期、未知RX'}
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
audit = read(ROOT/'prediction_audit.json')
data = {}
flat = []
for v in VARIANTS:
    p = ROOT/v
    score = read(p/'independent_scores.json')
    activation = read(p/'cross_response_activation.json')
    resource = read(p/'phase1_resource_summary.json')
    terminal = read(p/'phase1_terminal_status.json')
    receipt = read(p/'phase1_training_completion_receipt.json')
    epochs = [json.loads(line) for line in (p/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    log = (p/f'{v}.log').read_text(encoding='utf-8', errors='replace')
    assert len(epochs) == 200 and [m['epoch'] for m in epochs] == list(range(1,201))
    assert [m['phase'] for m in epochs] == ['label']*130 + ['pseudo']*70
    assert score['status'] == terminal['status'] == 'COMPLETE' and audit[v]['status'] == 'VERIFIED'
    assert receipt['phase1_training_complete'] and terminal['exit_code'] == 0
    assert score['truth_last'] and score['selection_source'] == 'final_only'
    assert activation['status'] == 'ACTIVE_VERIFIED' and not activation['missing']
    assert all(m.get('train_pseudo_truth_available',0) == 0 for m in epochs)
    errors = [line for line in log.splitlines() if any(t in line for t in ('Traceback', 'Error:', 'CUDA out of memory'))]
    assert not errors
    groups = {}
    derived = {}
    for scene in SCENES:
        aggregate = score['test'] if scene == 'clean' else score['sat_test_named'][scene]['aggregate']
        named = score['named_test'] if scene == 'clean' else score['sat_test_named'][scene]['named']
        groups[scene] = dict(aggregate=aggregate, named=named)
        matrix = audit[v]['confusion_matrices'][scene]
        f1 = []
        for i in range(6):
            tp = matrix[i][i]
            denom = sum(matrix[i]) + sum(row[i] for row in matrix)
            f1.append(200*tp/denom if denom else 0.)
        derived[scene] = dict(macro_f1=sum(f1)/6,
            worst_strict_receiver=min((named[f'test_unseen_day_rx_{rx}']['tx_acc'],rx) for rx in (0,2,5,7,9,10,11)),
            worst_seen_receiver=min((named[f'test_rx_{rx}']['tx_acc'],rx) for rx in (0,2,5,7,9,10,11)))
        for category, rows in [('aggregate',{'all':aggregate}),('named',named),('receiver_all_days',score['per_receiver'][scene]),('class',score['per_class'][scene])]:
            for group, metrics in rows.items():
                flat.append(dict(variant=v,scene=scene,category=category,group=group,
                    correct=metrics['tx_correct'],total=metrics['tx_total'],accuracy_percent=metrics['tx_acc']))
    phase_stats = {}
    for phase in ('label','pseudo'):
        subset = [m for m in epochs if m['phase']==phase]
        phase_stats[phase] = dict(epochs=len(subset),epoch_seconds=sum(m['epoch_time_s'] for m in subset),
             minimum_update_fraction=min(m['train_optimizer_step_applied'] for m in subset))
    data[v] = dict(score=score,groups=groups,derived=derived,resource=resource,activation=activation,
        final_train_accuracy=epochs[-1].get('train_tx_acc'),final_source_val_accuracy=epochs[-1].get('val_tx_acc'),
        first_joint_epoch=next((m['epoch'] for m in epochs if m.get('train_cross_response_response_joint_open',0)>0),None),
        phase_stats=phase_stats,checkpoint_sha256=terminal['selected_checkpoint_sha256'],
        source_split=terminal['source_split_receipt'], parsed_epochs=200, parsed_stdout_lines=len(log.splitlines()),
        errors=errors,warning_lines=[x for x in log.splitlines() if 'Warning' in x or '[WARN' in x],
        nonfinite_loss_epoch_rate_sum=sum(m.get('train_skipped_nonfinite_loss',0) for m in epochs),
        total_epoch_seconds=sum(m['epoch_time_s'] for m in epochs))
assert all(data[v]['source_split']==data['U0']['source_split'] for v in VARIANTS)

with (ROOT/'scores_long.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
(ROOT/'completed_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

lines=['# CORE90交叉响应：已完成三项详细测试数据', '',
    '数据快照：2026-09-11上午，本报告固定分析已完成并独立核验的U0、U2、Ux，其他变体不混入最终比较。seed392005，scratch-only，E200（label130+pseudo70），统一final-only checkpoint。', '',
    'U0为无交叉响应基线；U2为响应任务；Ux为身份交互约束。U0与U2/Ux也存在完整块采样差异，U1尚未纳入本次已完成快照，不能把差值全部归因于单一损失。U4联合变体尚未完成，不能据此判断双任务协同。', '',
    '## 证据与口径', '',
    '三项各完整解析200轮metrics及全部stdout；每项198000个独立物理测试记录、6类，每场景198000条预测，四场景合计792000条。已只读扫描全部2376000条预测，核对6类有限logits、唯一ID、四场景覆盖及完全相同的物理记录/分组/场景seed，再读取已固定的truth复算。每项所有总体、命名子集、接收机和类别计数均与独立scorer一致。', '',
    '测试组成：未知日期day0、已见RX为30000条；已见日期day1/2/3、未知RX为126000条；未知日期day0、未知RX为42000条。三个主组互斥，总计198000；test_day_0等别名不能再次累加。源数据L_s/U_s/V=6300/56700/27000。准确率单位%，差值为百分点；LEO均值是三个等样本量场景的算术均值。', '',
    '## 四场景总体', '', '|场景|样本数/项|U0|U2|U2−U0|Ux|Ux−U0|','|---|---:|---:|---:|---:|---:|---:|']
for s in SCENES:
    vals=[data[v]['groups'][s]['aggregate']['tx_acc'] for v in VARIANTS]
    lines.append(f'|{TITLES[s]}|198000|{vals[0]:.4f}|{vals[1]:.4f}|{vals[1]-vals[0]:+.4f}|{vals[2]:.4f}|{vals[2]-vals[0]:+.4f}|')
means=[sum(data[v]['groups'][s]['aggregate']['tx_acc'] for s in SCENES[1:])/3 for v in VARIANTS]
lines.append(f'|三LEO均值|594000场景记录|{means[0]:.4f}|{means[1]:.4f}|{means[1]-means[0]:+.4f}|{means[2]:.4f}|{means[2]-means[0]:+.4f}|')
lines += ['', '### 正确数与错误数', '', '|场景|U0正确/错误|U2正确/错误|Ux正确/错误|','|---|---:|---:|---:|']
for s in SCENES:
    lines.append('|'+TITLES[s]+'|'+'|'.join(f"{data[v]['groups'][s]['aggregate']['tx_correct']}/{198000-data[v]['groups'][s]['aggregate']['tx_correct']}" for v in VARIANTS)+'|')
lines += ['', '### Macro-F1（由完整混淆矩阵计算，%）', '', '|场景|U0|U2|Ux|','|---|---:|---:|---:|']
for s in SCENES:
    lines.append('|'+TITLES[s]+'|'+'|'.join(f"{data[v]['derived'][s]['macro_f1']:.4f}" for v in VARIANTS)+'|')
lines += ['', '## 日期/RX泛化分组', '', '|场景|分组|样本数|U0|U2|Ux|','|---|---|---:|---:|---:|---:|']
for s in SCENES:
    for key,title in GROUPS.items():
        lines.append(f"|{TITLES[s]}|{title}|{data['U0']['groups'][s]['named'][key]['tx_total']}|"+'|'.join(f"{data[v]['groups'][s]['named'][key]['tx_acc']:.4f}" for v in VARIANTS)+'|')
lines += ['', '## 每个未知RX的详细准确率', '', '每个未知RX的已见日期子集18000条，未知日期子集6000条；分别列示，避免混合不同难度。']
for s in SCENES:
    lines += ['',f'### {TITLES[s]}','','|RX|U0已见日期|U2已见日期|Ux已见日期|U0未知日期|U2未知日期|Ux未知日期|','|---:|---:|---:|---:|---:|---:|---:|']
    for rx in (0,2,5,7,9,10,11):
        vals=[data[v]['groups'][s]['named'][f'{prefix}{rx}']['tx_acc'] for prefix in ('test_rx_','test_unseen_day_rx_') for v in VARIANTS]
        lines.append(f'|{rx}|'+'|'.join(f'{x:.4f}' for x in vals)+'|')
lines += ['', '## 各类别准确率', '', '类别为数据集内部TX索引0–5，每类每场景33000条；不把索引解释为外部设备名称。']
for s in SCENES:
    lines += ['',f'### {TITLES[s]}','','|TX索引|U0|U2|Ux|','|---:|---:|---:|---:|']
    for cls in range(6):
        lines.append(f'|{cls}|'+'|'.join(f"{data[v]['score']['per_class'][s][str(cls)]['tx_acc']:.4f}" for v in VARIANTS)+'|')
lines += ['', '## 成本与训练健康', '', '同机存在其他并发实验；wall time是本次端到端观察值，包含训练/评估开销，不是隔离吞吐基准，不能据此声称算法更快。显存为进程PyTorch峰值allocated/reserved，不是整卡总量。', '',
    '|指标|U0|U2|Ux|','|---|---:|---:|---:|']
cost_rows=[('总墙钟时间（小时）',[data[v]['resource']['wall_time_seconds']/3600 for v in VARIANTS]),
    ('epoch时间和（小时）',[data[v]['total_epoch_seconds']/3600 for v in VARIANTS]),
    ('峰值allocated（GiB）',[data[v]['resource']['peak_cuda_memory_allocated_bytes']/2**30 for v in VARIANTS]),
    ('峰值reserved（GiB）',[data[v]['resource']['peak_cuda_memory_reserved_bytes']/2**30 for v in VARIANTS]),
    ('最终源验证准确率',[data[v]['final_source_val_accuracy'] for v in VARIANTS]),
    ('最终增强训练准确率',[data[v]['final_train_accuracy'] for v in VARIANTS])]
for name,vals in cost_rows:
    lines.append('|'+name+'|'+'|'.join(f'{x:.4f}' for x in vals)+'|')
for title, key in [('计划batch数','batches'),('成功更新数','successful_steps')]:
    lines.append('|'+title+'|'+'|'.join(str(data[v]['activation']['counts'][key]) for v in VARIANTS)+'|')
lines += ['|未执行更新数|'+'|'.join(str(data[v]['activation']['counts']['batches']-data[v]['activation']['counts']['successful_steps']) for v in VARIANTS)+'|', '',
    '三项均无非有限loss跳步、无Traceback/OOM；少量梯度非有限导致跳步，最终均达到E200并闭合评分。增强训练准确率与无增强源验证准确率不是同一分布，不直接相减解释泛化差距。', '',
    '## 机制实际激活', '', '|指标|U0|U2|Ux|','|---|---:|---:|---:|']
for title,key in [('有效完整块','effective_blocks'),('响应头梯度步数','head_gradient_steps'),('响应域梯度步数','domain_gradient_steps'),('响应身份联合梯度步数','joint_gradient_steps'),('决策梯度步数','decision_gradient_steps'),('身份交互梯度步数','cross_gradient_steps')]:
    lines.append('|'+title+'|'+'|'.join(str(data[v]['activation']['counts'][key]) for v in VARIANTS)+'|')
lines += ['', 'U2从E7记录联合梯度，正式门槛16块、误差比≤0.95、连续3次未放宽；最后源验证误差比'+f"{data['U2']['activation']['gate']['last_ratio']:.6f}"+'。共享前端/身份前部的响应梯度最大值均0。Ux只执行身份交互约束；U0关闭交叉响应，不能把对照中的零值当成失活。U2额外辅助参数5416，主模型1049827参数；通用resource summary未包含这组外置辅助参数。', '',
    '## 解读与限制', '',
    f'U2相对U0的三LEO平均提高{means[1]-means[0]:.4f}个百分点，Ux提高{means[2]-means[0]:.4f}个百分点。clean改善更明显，但U2的雨衰总体略低于U0；不能写成所有场景一致改善。应结合未知日期/未知RX组及弱RX表判断是否用局部退化换取总体收益。', '',
    '这是单seed、三项已完成结果的描述性比较，没有跨seed置信区间或显著性结论，不作晋级。尚缺U1、U3、两个U4、U5、head_only、permanent_detach的本次最终比较，不能估计纯双任务协同。全部测试结果只用于报告，不反馈训练、门槛、选模或重跑。', '',
    '通用旧终态文件中的promotion_ready、p0/p1以及endpoint字段混有历史方法诊断，不作为本实验科学晋级依据；本报告依据完整预测复算、E200日志和交叉响应专用激活证据。', '',
    '## 可复核文件', '',
    '- `scores_long.csv`：全部总体、命名子集、12个接收机及6类的四场景正确数/总数/准确率。',
    '- `prediction_audit.json`：全量预测复核、完整6×6混淆矩阵，行是真值、列是预测。',
    '- `completed_summary.json`：评分、资源、激活、全日志统计与checkpoint摘要。',
    '- 各变体目录保存原始独立评分、资源、激活、完整metrics和stdout；逐样本预测原件保留在N607原run目录。', '']
(ROOT/'completed_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(report=str(ROOT/'completed_results.md'),csv_rows=len(flat),variants=list(VARIANTS),leo_means=means),ensure_ascii=False))
