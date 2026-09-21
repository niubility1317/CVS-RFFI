"""Recompute the conversation's 18 completed CORE90 results from saved scores."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/CORE90_CONVERSATION_COMPARISON_20260913'
BASE = ROOT / 'automation_reports/CV-SincNet'
P1 = BASE / 'core90_cross_response_s392005_20260911_r1/completed_detail'
P2 = BASE / 'core90_cross_response_v2_s392005_20260911_r1/detailed_refresh_20260912_2347'
A = json.loads((P1 / 'all_completed_summary.json').read_text(encoding='utf-8'))
B = json.loads((P2 / 'summary.json').read_text(encoding='utf-8'))['variants']
SCENES = ['clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']
NAMED = ['test_unseen_day_seen_rx', 'test_seen_day_unseen_rx', 'test_unseen_day_unseen_rx']
NAMES = ['已见RX/未知日期', '未知RX/已见日期', '未知RX/未知日期']
DESCRIPTIONS = {
 'U0': '历史ADV3B02_CORE90基线，本轮从零训练，关闭新增交叉机制',
 'U1': '交叉块组织及角色隔离，不加响应/决策辅助损失',
 'U1_mask_off': 'U1关闭角色mask的机制对照',
 'U2': 'U1加响应预测',
 'U3': 'U1加决策保护/校准',
 'U4_additive': '加性响应预测与决策保护联合',
 'U4_bilinear': '双线性响应预测与决策保护联合',
 'U5': 'U4_bilinear加反馈调度',
 'Ux': '原始身份特征交互压零',
 'Ux_normalized': '先逐条单位范数归一化，再聚合并约束身份交互',
 'head_only': '仅训练响应辅助头的机制对照',
 'permanent_detach': '永久切断响应到身份路径，保留决策损失的机制对照',
}

all_groups = []
for version, path in [('V1', P1/'all_scores_long.csv'), ('V2', P2/'scores_long.csv')]:
    for r in csv.DictReader(path.open(encoding='utf-8-sig', newline='')):
        r = dict(r)
        r['version'] = version
        for k in ['correct', 'total']:
            r[k] = int(r[k])
        r['incorrect'] = r['total'] - r['correct']
        r['accuracy_percent'] = float(r['accuracy_percent'])
        assert abs(r['accuracy_percent'] - 100*r['correct']/r['total']) < 1e-9
        all_groups.append(r)
assert len(all_groups) == 2664
rows = []
for version, source in [('V1', A), ('V2', B)]:
    for variant, d in source.items():
        r = {'id': version+'/'+variant, 'version': version, 'variant': variant,
             'description': DESCRIPTIONS[variant]}
        for scene in SCENES:
            g = d['groups'][scene]
            r[scene] = g['aggregate']['tx_acc']
            assert g['aggregate']['tx_total'] == 198000
            aggregate = [x for x in all_groups if x['version']==version and x['variant']==variant and x['scene']==scene and x['category']=='aggregate']
            assert len(aggregate)==1 and abs(aggregate[0]['accuracy_percent']-r[scene])<1e-9
            assert sum(g['named'][n]['tx_total'] for n in NAMED)==198000
            assert sum(g['named'][n]['tx_correct'] for n in NAMED)==g['aggregate']['tx_correct']
            for n in NAMED:
                r[n+'__'+scene] = g['named'][n]['tx_acc']
            r['macro_f1__'+scene] = d['derived'][scene]['macro_f1'] if version=='V1' else d['macro_f1'][scene]
            floor = d['derived'][scene]['worst_strict_receiver'] if version=='V1' else d['worst_strict_rx'][scene]
            r['worst_strict_rx__'+scene] = floor[0]
            r['worst_strict_rx_id__'+scene] = floor[1]
            classes = [x for x in all_groups if x['version']==version and x['variant']==variant and x['scene']==scene and x['category']=='class']
            assert len(classes)==6
            worst = min(classes, key=lambda x:x['accuracy_percent'])
            r['worst_class__'+scene] = worst['accuracy_percent']
            r['worst_class_id__'+scene] = worst['group']
        r['leo_mean'] = sum(r[s] for s in SCENES[1:])/3
        for prefix in NAMED+['macro_f1', 'worst_strict_rx', 'worst_class']:
            r[prefix+'__leo_mean'] = sum(r[prefix+'__'+s] for s in SCENES[1:])/3
        r['wall_hours'] = d['resource']['wall_time_seconds']/3600
        r['allocated_gib'] = d['resource']['peak_cuda_memory_allocated_bytes']/2**30
        rows.append(r)
assert len(rows)==18
assert A['U0']['source_split']==B['U0']['source_split']
rows.sort(key=lambda r:r['leo_mean'], reverse=True)
lookup = {r['id']:r for r in rows}
assert rows[0]['id']=='V2/U0'
assert max((r for r in rows if r['variant']!='U0'), key=lambda r:r['leo_mean'])['id']=='V2/Ux_normalized'
assert max(rows,key=lambda r:r['clean'])['id']=='V2/Ux_normalized'
OUT.mkdir(exist_ok=True)
for filename, values in [('ranking.csv', rows), ('all_scores_long.csv', all_groups)]:
    with (OUT/filename).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader(); w.writerows(values)
(OUT/'summary.json').write_text(json.dumps({'rows':rows,'score_groups':len(all_groups),'prediction_rows':18*4*198000,'source_split_equal':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

lines = ['# 本对话CORE90优化实现测试集全量比较', '',
 '范围：V1的10组与V2的8组已完成实验，共18组。只比较本对话的交叉响应实现，不纳入其他历史方法。数据取自两批已保存的独立评分结果，本次未重新训练或改变预测。', '',
 '**结论：新增优化中V2/Ux_normalized最好；包括基线时，V2/U0在三个LEO总体场景及跨接收机LEO均值上最好。不存在同时超过它们clean与LEO均值的第三组。**', '',
 'V2/Ux_normalized的clean为80.3697%，LEO均值64.0231%；V2/U0分别为78.8904%和64.8497%。所有增减均用百分点，LEO均值是三个LEO场景的等权算术平均，不把clean混入。', '',
 '## 口径与证据', '',
 '两批模型/数据seed均为392005，从零训练，200epochs，使用最终模型。ManySig、6类；L_s/U_s/V为6300/56700/27000，源RX1、3、4、6、8，目标RX0、2、5、7、9、10、11。两批保存的source_split对象完全一致。每组每场景198000条测试预测；三项互斥分组分别为30000、126000、42000条。四场景使用相同物理样本的不同场景版本，不能当成四倍独立样本。', '',
 '18组共3600epochs、14256000条场景预测、2664项评分分组。本次重新核对全部2664项correct/total与准确率，以及每行三个互斥子组与总体计数相加一致；未重复远端训练或全量预测审计。Macro-F1沿用已有完整混淆矩阵计算结果。', '',
 '## 全部测试集结果（按LEO均值排序）', '']

def table(headers, records):
    lines.append('|'+'|'.join(headers)+'|')
    lines.append('|'+'|'.join(['---']*len(headers))+'|')
    for record in records:
        lines.append('|'+'|'.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in record)+'|')
    lines.append('')

table(['组','clean','晴空','低仰角','雨衰','LEO均值'], [[r['id']]+[r[s] for s in SCENES]+[r['leo_mean']] for r in rows])
lines += ['## 各实现的含义', '']
table(['标识','含义'], list(DESCRIPTIONS.items()))
lines += ['U0是按当前source-only契约重新训练的历史方法语义基线，不是加载历史checkpoint，也不是裸分类器。V1与V2的同名项属于不同实现版本，不合并平均。V1联合行使用旧门控；V2已完成8行不包含身份响应联合候选的完整测试。', '',
 '## 同版本对照', '']
contrasts = [('V1/U1','V1/U0'),('V1/U2','V1/U1'),('V1/U3','V1/U1'),('V1/U4_additive','V1/U1'),('V1/U4_bilinear','V1/U1'),('V1/U5','V1/U4_bilinear'),('V1/U5','V1/U1'),('V2/U1','V2/U0'),('V2/U1','V2/U1_mask_off'),('V2/U3','V2/U1'),('V2/Ux','V2/U1'),('V2/Ux_normalized','V2/Ux'),('V2/Ux_normalized','V2/U1'),('V2/Ux_normalized','V2/U0')]
table(['差值（前者−后者）','clean差值','LEO均值差值'], [[x+' − '+y, lookup[x]['clean']-lookup[y]['clean'],lookup[x]['leo_mean']-lookup[y]['leo_mean']] for x,y in contrasts])
lines += ['归一化Ux对原始Ux的四场景改善一致。U3未显示相对U1的LEO收益；V1/U5优于U4_bilinear，但仍低于V1/U1。V1加性联合交互量U4_additive−U2−U3+U1仅为clean+0.4848、LEO+0.1394个百分点，单seed不能声称显著协同；双线性版本改变容量，不能作为纯双因素协同估计。', '', '## 跨接收机与日期分组', '']
for n,name in zip(NAMED,NAMES):
    lines += ['### '+name, '']
    table(['组','clean','晴空','低仰角','雨衰','LEO均值'],[[r['id']]+[r[n+'__'+s] for s in SCENES]+[r[n+'__leo_mean']] for r in rows])
lines += ['严格未知RX/未知日期的clean最高为V2/permanent_detach的75.1786%，仅比Ux_normalized高0.0214个百分点；三LEO的严格分组仍由V2/U0领先。不能将总体第一解释为每个子域都第一。', '', '## Macro-F1与尾部性能', '']
for key, title in [('macro_f1','Macro-F1（%）'),('worst_class','最弱类别召回率（%）'),('worst_strict_rx','严格分组最弱RX准确率（%）')]:
    lines += ['### '+title,'']
    table(['组','clean','晴空','低仰角','雨衰','三LEO均值'],[[r['id']]+[r[key+'__'+s] for s in SCENES]+[r[key+'__leo_mean']] for r in rows])
lines += ['每个场景最弱类别/接收机可能不同，尾部均值是各场景最小值的平均，不是同一固定对象的合并得分。对象ID保存在ranking.csv，全部类别及接收机明细保存在all_scores_long.csv。V2/Ux_normalized最弱严格RX均为RX2：clean60.7500%，晴空39.2333%、低仰角40.9833%、雨衰41.2667%；后面三项均低于V2/U0。', '', '## 训练成本（次要指标）', '']
table(['组','实际训练小时','峰值分配显存GiB'],[[r['id'],r['wall_hours'],r['allocated_gib']] for r in rows])
lines += ['耗时是共享GPU调度下的观测，不是隔离吞吐基准。V2/Ux_normalized约4.30小时，U0约3.01小时；不能据此精确声称算法慢多少。V2每组200epochs、计划9800步，实际成功9781—9793步，有7—19次梯度跳步，无非有限loss或运行错误，不能写成零跳步。', '',
 '## 结论边界与未测试实现', '',
 '1. V2/U0相对V1/U0的clean+1.5899、LEO均值+3.2128个百分点，而U0新增交叉机制关闭。因此跨版本绝对排名可以描述已测结果，不能将同名行差值直接归因于优化代码。',
 '2. 只有一个训练seed，没有多seed均值、方差和置信区间；大量测试样本不能替代独立训练重复。最终模型测试结果也不能冒充最佳测试epoch。',
 '3. head_only与U1在真实训练第1epoch已发生轨迹分歧，尚未由正式逐步追踪定位；其分数可报告，不能当成严格轨迹一致的纯辅助头因果证明。permanent_detach也是完整配置对照，不把差值全部归为切断单一路径。',
 '4. 本批V2未启动U2、U4_additive、U4_bilinear、U5、U3_delta、U3_delta_pairs、U4_decomposed、U5_reliable；它们无测试集分数，不填零，不参与排名。V1已有同名部分结果，但不能替代V2验证。',
 '5. 本报告是既有固定预测的回顾性比较，不以目标测试结果回流调参、选择重跑或宣称独立确认/正式晋级。', '',
 '## 来源', '',
 '- [V1完整结果](../../automation_reports/CV-SincNet/core90_cross_response_s392005_20260911_r1/completed_detail/all_completed_summary.json)',
 '- [V2完整结果](../../automation_reports/CV-SincNet/core90_cross_response_v2_s392005_20260911_r1/detailed_refresh_20260912_2347/summary.json)',
 '- [V2详细审计报告](../../automation_reports/CV-SincNet/core90_cross_response_v2_s392005_20260911_r1/detailed_refresh_20260912_2347/detailed_results.md)',
 '- [全18行派生指标](ranking.csv)', '- [全2664项分组成绩](all_scores_long.csv)', '']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'verified_groups':len(all_groups),'rows':len(rows),'output':str(OUT),'top_two':rows[:2]},ensure_ascii=False,indent=2))
