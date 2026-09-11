"""Complete ten-row report; reuse the validated full-log parser, preserve old snapshot."""
from pathlib import Path
source = (Path(__file__).parent/'build_completed_report.py').read_text(encoding='utf-8').split("lines=['#")[0]
source = source.replace("VARIANTS = ('U0', 'U2', 'Ux')", "VARIANTS = ('U0','U1','U2','U3','U4_additive','U4_bilinear','U5','Ux','head_only','permanent_detach')")
source = source.replace("audit = read(ROOT/'prediction_audit.json')", "audit = {**read(ROOT/'prediction_audit.json'), **read(ROOT/'prediction_audit_new7.json')}")
source = source.replace("'scores_long.csv'", "'all_scores_long.csv'").replace("'completed_summary.json'", "'all_completed_summary.json'")
exec(compile(source, 'validated_full_log_parser', 'exec'))

means = {v:sum(data[v]['groups'][s]['aggregate']['tx_acc'] for s in SCENES[1:])/3 for v in VARIANTS}
rows=['# CORE90交叉响应：全部10项最终测试数据', '',
      '快照2026-09-11 14:44（UTC+8）。新增完成U1、U3、U4_additive、U4_bilinear、U5、head_only、permanent_detach；至此10项均完成E200、四场景预测固定及独立评分。seed392005、scratch-only、label130+pseudo70、final-only，沿用原ManySig.pkl。', '',
      '所有10项各完整解析200轮metrics和全部stdout，验证2000轮训练记录；旧3项和新增7项的全部7920000条预测均逐条核对ID、场景、6类有限logits及完整覆盖，再读取固定truth，复算每项148组、总计1480组评分计数，均完全一致。所有变体使用相同物理记录、数据角色、场景seed和源划分。', '',
      '每项每场景198000个独立物理测试记录、6类；四场景792000条。未知日期/已见RX30000条、已见日期/未知RX126000条、未知日期/未知RX42000条，三主组互斥；不重复累计别名。下表准确率与Macro-F1单位%，差值为百分点。三LEO均值为三个等样本量场景算术平均。', '',
      '## 总体准确率', '', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|LEO均值|clean−U0|LEO均值−U0|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for v in VARIANTS:
    vals=[data[v]['groups'][s]['aggregate']['tx_acc'] for s in SCENES]
    rows.append('|'+v+'|'+'|'.join(f'{x:.4f}' for x in vals)+f"|{means[v]:.4f}|{vals[0]-data['U0']['groups']['clean']['aggregate']['tx_acc']:+.4f}|{means[v]-means['U0']:+.4f}|")
rows += ['', 'U0：原基线；U1：完整块采样/交叉前向对照、不加附加任务；U2：响应；U3：决策；U4_additive/U4_bilinear：加性/双线性响应＋决策；U5：双线性联合＋反馈调度；Ux：身份交互；head_only：只训练响应辅助头；permanent_detach：响应对身份永久detach、保留决策。', '',
         '## 三种泛化子集', '']
for group,title in GROUPS.items():
    rows += [f'### {title}', '', '|变体|样本数/场景|clean|晴空LEO|低仰角LEO|雨衰LEO|', '|---|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        rows.append(f"|{v}|{data[v]['groups']['clean']['named'][group]['tx_total']}|"+'|'.join(f"{data[v]['groups'][s]['named'][group]['tx_acc']:.4f}" for s in SCENES)+'|')
    rows.append('')

contrasts = {
    'U1−U0：采样/前向对照': {'U1':1,'U0':-1},
    'U2−U1：增加响应任务': {'U2':1,'U1':-1},
    'U3−U1：增加决策任务': {'U3':1,'U1':-1},
    'U4_additive−U2−U3+U1：匹配加性双因素': {'U4_additive':1,'U2':-1,'U3':-1,'U1':1},
    'U4_bilinear−U4_additive：改为双线性': {'U4_bilinear':1,'U4_additive':-1},
    'U5−U4_bilinear：增加反馈调度': {'U5':1,'U4_bilinear':-1},
    'U4_bilinear−permanent_detach：允许响应进入身份后部': {'U4_bilinear':1,'permanent_detach':-1},
    'Ux−U1：增加身份交互': {'Ux':1,'U1':-1},
}
rows += ['## 预登记对照差值', '', '|对比|clean|晴空LEO|低仰角LEO|雨衰LEO|LEO均值|','|---|---:|---:|---:|---:|---:|']
contrast_results={}
for name,weights in contrasts.items():
    vals=[sum(weight*data[v]['groups'][s]['aggregate']['tx_acc'] for v,weight in weights.items()) for s in SCENES]
    vals.append(sum(vals[1:])/3)
    contrast_results[name]=dict(zip((*SCENES,'leo_mean'),vals))
    rows.append('|'+name+'|'+'|'.join(f'{x:+.4f}' for x in vals)+'|')
rows += ['', '上述均为同seed描述性差值，无跨seed置信区间。仅U4_additive−U2−U3+U1是匹配容量的双因素对照；U4_bilinear不能用同一公式声称纯任务协同。即使一个对照差值为正，也不意味着该变体超过最强单任务或采样对照。', '',
         '## Macro-F1', '', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|','|---|---:|---:|---:|---:|']
for v in VARIANTS:
    rows.append('|'+v+'|'+'|'.join(f"{data[v]['derived'][s]['macro_f1']:.4f}" for s in SCENES)+'|')
rows += ['', '## 弱接收机下限', '', '仅在未知RX集合0/2/5/7/9/10/11内计算；每格为最低准确率及对应RX。已见日期子集每RX18000条，未知日期子集每RX6000条。']
for kind,title in [('worst_seen_receiver','已见日期'),('worst_strict_receiver','未知日期')]:
    rows += ['',f'### {title}', '', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|','|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        rows.append('|'+v+'|'+'|'.join(f"{data[v]['derived'][s][kind][0]:.4f}（RX{data[v]['derived'][s][kind][1]}）" for s in SCENES)+'|')
rows += ['', '## 全部未知RX明细', '']
for s in SCENES:
    for prefix,title in [('test_rx_','已见日期'),('test_unseen_day_rx_','未知日期')]:
        rows += [f'### {TITLES[s]}，{title}', '', '|变体|RX0|RX2|RX5|RX7|RX9|RX10|RX11|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for v in VARIANTS:
            rows.append('|'+v+'|'+'|'.join(f"{data[v]['groups'][s]['named'][prefix+str(rx)]['tx_acc']:.4f}" for rx in (0,2,5,7,9,10,11))+'|')
        rows.append('')
rows += ['## 全部类别明细', '', '每类每场景33000条，TX0–5为内部索引。']
for s in SCENES:
    rows += ['',f'### {TITLES[s]}', '', '|变体|TX0|TX1|TX2|TX3|TX4|TX5|','|---|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        rows.append('|'+v+'|'+'|'.join(f"{data[v]['score']['per_class'][s][str(c)]['tx_acc']:.4f}" for c in range(6))+'|')
rows += ['', '## 成本、全程稳定性与激活', '',
         '|变体|墙钟小时|峰值allocated GiB|成功更新/9800|未执行更新|响应头梯度步|域响应梯度步|身份响应梯度步|决策梯度步|身份交互梯度步|',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for v in VARIANTS:
    d=data[v]; c=d['activation']['counts']; r=d['resource']
    rows.append(f"|{v}|{r['wall_time_seconds']/3600:.4f}|{r['peak_cuda_memory_allocated_bytes']/2**30:.4f}|{c['successful_steps']}|{9800-c['successful_steps']}|{c['head_gradient_steps']}|{c['domain_gradient_steps']}|{c['joint_gradient_steps']}|{c['decision_gradient_steps']}|{c['cross_gradient_steps']}|")
rows += ['', '墙钟时间受同机并发和评估影响，是本次调度环境观察，不是隔离速度基准；峰值是进程PyTorch allocated，不是整卡总量。全部2000轮、完整stdout均已解析：无Traceback/OOM/非有限loss跳步，少量非有限梯度跳步后仍完成E200。所有交叉响应专用激活报告为ACTIVE_VERIFIED；共享前端及身份前部响应梯度最大值均0。U2/U4_additive/U5从E7、U4_bilinear从E6记录身份响应联合梯度；head_only和permanent_detach的身份响应梯度步数保持0。', '',
         '## 科学解释', '',
         '总体clean最高为U3；三LEO均值最高为U1。U1已经超过U0，说明原先仅比较U2/Ux对U0不足以证明附加任务收益。U2相对U1在四个总体场景均下降；Ux相对U1同样四场景均下降。U3相对U1改善clean，但三个LEO总体均下降。', '',
         'U4_additive的匹配双因素差值在clean为正，但实际clean低于U3与U1；LEO均值的双因素差值为负。U4_bilinear低于U4_additive，且比permanent_detach的clean低；本次结果不支持双线性响应向身份传播带来稳定收益。', '',
         'U5较U4_bilinear恢复了clean和LEO表现，说明反馈调度在此对照下有描述性改善；但仍未超过U1的总体clean/LEO均值，不能据此称完整方案胜出。弱RX和类别仍存在明显不均衡，详见上述全表。', '',
         '全部结果是单seed描述，无统计显著性、干净开放集识别或科学晋级主张。旧通用终态promotion_ready等字段不作晋级依据。所有预测先固定、评分独立，结果不反馈调参、选模或选择性重跑；没有因性能较低停止或重发任何行。', '',
         '## 可复核文件', '',
         '- `all_scores_long.csv`：1480行原始分组正确数、总数和准确率；包含命名别名，汇总时不能直接把各行相加。',
         '- `all_completed_summary.json`：10项评分、资源、激活及全日志统计。',
         '- `prediction_audit.json`及`prediction_audit_new7.json`：全量预测复算和全部6×6混淆矩阵。',
         '- `all_contrasts.json`：预登记对照差值；没有跨seed置信区间。',
         '- 各变体目录保存原始评分、资源、终态、激活、200轮metrics和完整stdout；逐样本预测保留在N607原run。', '']
(ROOT/'all_results.md').write_text('\n'.join(rows),encoding='utf-8')
(ROOT/'all_contrasts.json').write_text(json.dumps(contrast_results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(ROOT/'all_results.md'),leo_means=means,contrasts=contrast_results),ensure_ascii=False,indent=2))
