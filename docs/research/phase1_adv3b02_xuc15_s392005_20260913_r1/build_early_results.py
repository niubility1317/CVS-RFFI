import csv
import json
from pathlib import Path

p = Path(__file__).resolve().parent
data = json.loads((p/'early_eval_scores_latest.json').read_text(encoding='utf-8'))
assert data['verification'] == 'VERIFIED_COMPLETE_13_ROWS_4_SCENARIOS_TRUTH_LAST'
scores = data['scores']
scenes = ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
names = ['ADV3B02 CORE90基线','grid基座','grid+X','grid+U','grid+原C2','grid+X+U，被动审计','grid+X+原C2','grid+U+原C2','grid+X+U+原C2','原生A1','原生A1+X','普通CORE90+原C2','grid+X+U+C*+暴露课程','grid+X+U+C*，固定课程','grid+X+U，仅暴露课程']
lines = ['# XUC13已完成模型四场景测试结果', '', '本报告为用户要求的提前测试：13个E200最终模型、seed392005；M09/M10仍由原流程训练，未纳入此表。所有13组预测先固定，再由独立scorer连接truth，覆盖8736000条预测。模型、数据划分、信道配置与已发布矩阵保持一致。', '', '每组每场景168000个目标样本，共6个TX。表中为准确率百分比；三LEO均值是三个正式弱信道场景的等权均值，Δ为相对本次M00的百分点差。该单seed确认集结果不用于选模、改参或选择性重跑。', '', '|实验|配置|clean%|clear%|low_elev%|rain%|三LEO均值%|Δclean(pp)|ΔLEO(pp)|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
base = [scores['M00']['metrics'][s]['accuracy']*100 for s in scenes]
flat = []
for rid in sorted(scores):
    values = [scores[rid]['metrics'][s]['accuracy']*100 for s in scenes]
    avg = sum(values[1:])/3
    lines.append('|'+rid+'|'+names[int(rid[1:])]+'|'+'|'.join(f'{v:.4f}' for v in values+[avg,values[0]-base[0],avg-sum(base[1:])/3])+'|')
    for scene in scenes:
        m = scores[rid]['metrics'][scene]
        flat.append(dict(row=rid,method=names[int(rid[1:])],scenario=scene,total=m['total'],correct=m['correct'],accuracy_pct=100*m['accuracy'],**{'tx'+str(k)+'_accuracy_pct':100*m['per_class_accuracy'][str(k)] for k in range(6)}))
lines += ['', '## 预定比较的描述性差值', '', '以下差值只描述同一次固定测试，不能替代多seed显著性，也不用于改变正在运行的M09/M10。', '', '|比较|含义|Δclean(pp)|Δclear(pp)|Δlow_elev(pp)|Δrain(pp)|','|---|---|---:|---:|---:|---:|']
for hi,lo,label in [('M01','M00','grid基座变化'),('M02','M01','grid体系加入X'),('M03','M01','grid体系加入U'),('M04','M01','grid体系加入原C2'),('M08','M05','X+U体系加入原C2（含其课程差异）'),('M11','M00','普通CORE90加入原C2'),('M12','M14','相同暴露课程下C*配置差异'),('M13','M05','固定课程下主动C*与被动审计差异')]:
    delta = [100*(scores[hi]['metrics'][s]['accuracy']-scores[lo]['metrics'][s]['accuracy']) for s in scenes]
    lines.append('|'+hi+'−'+lo+'|'+label+'|'+'|'.join(f'{v:+.4f}' for v in delta)+'|')
lines += ['', 'M12/M13整程没有CATCHUP/CORRECT动作。因此相关差值不能归因于实际主动纠正，也不能证明三机制协同。M05仅被动审计、M14禁止控制动作；原C2的CORRECT在M04/M06/M07/M08/M11分别为366/301/212/290/280次。', '', '## 每个TX的场景准确率', '', '|实验|场景|正确/总数|TX0%|TX1%|TX2%|TX3%|TX4%|TX5%|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for row in flat:
    lines.append('|'+row['row']+'|'+row['scenario']+'|'+str(row['correct'])+'/'+str(row['total'])+'|'+'|'.join(f"{row['tx'+str(k)+'_accuracy_pct']:.4f}" for k in range(6))+'|')
lines += ['', '## 证据与限制', '', '独立评分器逐场景验证opaque ID覆盖完整且无重复。本次读回验证13个score、52个场景、每场景168000条、总计8736000条；所有prediction修改时间均早于predictions_complete记录，全部评分晚于该记录。', '', '结果来自提前测试目录phase1_adv3b02_xuc13_early_eval_s392005_20260913_r1；原15行dispatcher仍保留后续完整测试流程，两次产物分开保存。本表没有M09/M10结果；没有利用source验证准确率冒充目标测试准确率。当前scorer提供准确率及每类召回率（per_class_accuracy），未提供F1、混淆矩阵或逐RX指标。', '', '原始评分和核验：[early_eval_scores_latest.json](early_eval_scores_latest.json)。完整52行表：[early_eval_metrics.csv](early_eval_metrics.csv)。', '']
(p/'early_eval_results.md').write_text('\n'.join(lines),encoding='utf-8')
with (p/'early_eval_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(flat[0]))
    writer.writeheader(); writer.writerows(flat)
print('\n'.join(lines[:22]))
