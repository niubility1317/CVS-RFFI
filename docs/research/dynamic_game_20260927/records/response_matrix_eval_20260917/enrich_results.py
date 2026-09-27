"""Add explicit joint cross-day/cross-receiver summaries from verified scores."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=read('summary.csv');groups=read('all_groups.csv');pairs=read('paired_seed_summary.csv')
day0=[r for r in groups if r['axis']=='day_scene' and r['key'].split('|')[0]=='0']
assert len(day0)==4*len(summary) and all(int(r['n'])==42000 for r in day0)
write('day0_cross_day_cross_RX.csv',day0)
day0rows=[]
for row in summary:
    selected={r['key'].split('|')[1]:r for r in day0 if r['row']==row['row']}
    out={k:row[k] for k in ('row','method','seed')}
    for scene,r in selected.items():
        for metric in ('accuracy','macro_f1'):out[scene+'_'+metric]=float(r[metric])
    for metric in ('accuracy','macro_f1'):out['leo_mean_'+metric]=statistics.mean(out[s+'_'+metric] for s in ('leo_clear_weak','leo_low_elev_weak','leo_rain_weak'))
    day0rows.append(out)
write('day0_summary.csv',day0rows)
aggs=[]
for method in sorted({r['method'] for r in day0rows}):
    rows=[r for r in day0rows if r['method']==method]
    for metric in [k for k in rows[0] if k not in ('row','method','seed')]:
        values=[r[metric] for r in rows]
        aggs.append(dict(method=method,metric=metric,n_seeds=len(values),mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 'NA'))
write('day0_seed_summary.csv',aggs)
means=[]
for method in sorted({r['method'] for r in summary}):
    rows=[r for r in summary if r['method']==method]
    out=dict(method=method,n_seeds=len(rows),seeds='|'.join(r['seed'] for r in rows))
    for metric in ('clean_accuracy','leo_mean_accuracy','leo_mean_macro_f1','training_hours'):
        values=[float(r[metric]) for r in rows]
        out[metric+'_mean']=statistics.mean(values)
        out[metric+'_sd']=statistics.stdev(values) if len(values)>1 else 'NA'
    values=[r['leo_mean_accuracy'] for r in day0rows if r['method']==method]
    out['day0_leo_accuracy_mean']=statistics.mean(values)
    out['day0_leo_accuracy_sd']=statistics.stdev(values) if len(values)>1 else 'NA'
    means.append(out)
write('method_summary.csv',means)
text=['## 本轮结果解读','',f'本次完成{len(summary)}行评估，详细状态见coverage36.csv。新增17行；原16行复用固定预测和评分。','',
      '以下为同模型seed配对LEO准确率差值；±为训练seed间样本SD，不是置信区间。不同完成seed集合不能直接混排。','',
      '|对比|配对seed数|LEO变化（百分点）|','|--|--:|--:|']
for p in pairs:
    if p['scene']=='LEO_MEAN':text.append(f"|{p['candidate']}−{p['reference']}|{p['n_seeds']}|{float(p['mean_difference_pp']):+.4f}±{float(p['sd_difference_pp']):.4f}|")
text+=['','全测试集均为跨接收机；source为day1/2/3，target含day0/1/2/3，因此只有day0的42000样本同时满足跨天＋跨接收机，占25%。day0_summary.csv与day0_seed_summary.csv单列这一严格子集；其余日为同日跨接收机。四个场景复用同一物理样本集。','',
       '完整文件保留每seed、每场景Accuracy/Macro-F1、RX/day/TX及交叉分组、各类F1与混淆矩阵、最弱RX/TX、训练时间、6600轮源域曲线、同seed配对与四格交互。未完成项没有填0，也没有选用中途checkpoint。','']
p=OUT/'report.md';old=p.read_text(encoding='utf-8');p.write_text(old.split('\n',1)[0]+'\n\n'+'\n'.join(text)+'\n'+old.split('\n',1)[1],encoding='utf-8')
historical=ROOT.parent/'all_exploration_20260913/all_final_results.csv'
with historical.open(encoding='utf-8-sig',newline='') as f:
    baselines=[r for r in csv.DictReader(f) if (r['run'],r['row']) in [('a1_fast_v2_s392005_20260909_r1','F0_FIXED_BASE'),('a1_mechanism_periodic_s392005_20260910_r1','B0_FIXED')]]
assert len(baselines)==2
comparison=[]
for b in baselines:
    for r in summary:
        if r['seed']!='392005':continue
        comparison.append(dict(historical_run=b['run'],historical_row=b['row'],current_row=r['row'],seed=392005,historical_clean_pct=float(b['clean']),current_clean_pct=100*float(r['clean_accuracy']),clean_delta_pp=100*float(r['clean_accuracy'])-float(b['clean']),historical_leo_pct=float(b['leo_mean']),current_leo_pct=100*float(r['leo_mean_accuracy']),leo_delta_pp=100*float(r['leo_mean_accuracy'])-float(b['leo_mean']),scope='descriptive historical reference; not strict matched ablation'))
write('historical_DAOT_RC4_seed392005_reference.csv',comparison)
with p.open('a',encoding='utf-8') as f:
    f.write('\n## 历史DAOT＋RC4的seed392005参考\n\n历史V2/F0_FIXED_BASE：Clean79.8464%、LEO68.1175%；机制矩阵B0_FIXED：Clean77.6738%、LEO67.8315%。二者为不同历史run，不能混用名称。逐行差值见historical_DAOT_RC4_seed392005_reference.csv。\n\n已核实历史与本批的source/target RX、日期范围、L/U/V比例、模型seed和E200口径相同。但历史训练启用AMP、fasttrust学习率调度；本批为FP32和本批固定优化配置。历史配置日志还记录不同信道种子/视图设置，且本次没有逐一核对历史物理角色ID与全部测试信道实现。因此这些数值仅作历史成绩参考，不能宣称为完全同数据实例、同配方的机制因果比较。历史来源：E:/type10-7/automation_reports/CV-SincNet/all_exploration_20260913/report.md及evidence.json.gz。\n')
print(json.dumps(dict(day0_rows=len(day0rows),day0_groups=len(day0),paired_leo=[r for r in pairs if r['scene']=='LEO_MEAN']),ensure_ascii=False))
