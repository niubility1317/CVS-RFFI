"""Recompute three-seed tables from published fixed A-F metric records.
Run after final_results.json and final_target_recount.json have been collected.
No datasets, checkpoints, target IQ or fitted classifiers are opened.
"""
import csv,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent

def stats(v):return dict(mean=statistics.mean(v),sample_SD=statistics.stdev(v),positive_count=sum(x>0 for x in v))
def main():
 data=json.loads((P/'final_results.json').read_text(encoding='utf8'));audit=json.loads((P/'final_target_recount.json').read_text(encoding='utf8'));result=data['result'];records=result['records']
 if data['status']!='COLLECTED_COMPLETE' or audit['status']!='VERIFIED' or len(records)!=324 or data['new_steps_recounted']!=166500:raise ValueError('Complete independently checked matrix required')
 summaries={};seedrows=[]
 for arm in 'ABCDEF':
  per=[]
  for seed in (2026092701,2026092702,2026092703):
   row=[r for r in records if r['arm']==arm and r['model_seed']==seed];overall=next(r for r in row if r['dimension']=='overall')
   if len(row)!=18 or overall['query_count']!=168000:raise ValueError('Complete physical strata required')
   cost=next(v for v in result['costs'] if v['arm']==arm and v['model_seed']==seed)
   r=dict(arm=arm,model_seed=seed,accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX_accuracy=min(v['accuracy'] for v in row if v['dimension']=='receiver'),worst_day_accuracy=min(v['accuracy'] for v in row if v['dimension']=='day'),source_elapsed_seconds=cost['elapsed_source_seconds'],total_parameters=cost['total_parameters'],feature_parameters=232033 if arm in ('B','F') else 195041,new_epochs=cost['new_epochs'],new_optimizer_steps=cost['new_optimizer_steps'],historical_reference_epochs=cost['historical_reference_epochs_separate']);per.append(r);seedrows.append(r)
  summaries[arm]=dict(seed_count=3,per_seed=per,**{k:stats([r[k] for r in per]) for k in ('accuracy','macro_f1','worst_RX_accuracy','worst_day_accuracy','source_elapsed_seconds')})
  if abs(summaries[arm]['accuracy']['mean']-audit['summary'][arm]['accuracy_mean'])>1e-12:raise ValueError('Published summary differs from independent recount')
 with (P/'per_seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(seedrows[0]));w.writeheader();w.writerows(seedrows)
 comparisons=audit['comparisons'];pairrows=[]
 for name,s in comparisons.items():
  for r in s['per_seed']:pairrows.append(dict(comparison=name,model_seed=r['model_seed'],difference_pp=r.get('accuracy_difference_pp',r.get('interaction_pp')),corrected_errors=r.get('corrected_errors'),introduced_errors=r.get('introduced_errors'),net_correct_interaction=r.get('net_correct_interaction')))
 with (P/'paired_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)
 fa=[100*(f['accuracy']-a['accuracy']) for f,a in zip(summaries['F']['per_seed'],summaries['A']['per_seed'])]
 summary=dict(status='VERIFIED',run_id=data['run_id'],summary=summaries,comparisons=comparisons,F_minus_A_descriptive_pp=stats(fa),goal84=summaries['F']['accuracy']['mean']>=.84,source_decision_commit=data['source_decision_upload']['commit'],new_epochs=750,historical_reference_epochs_separate=150,new_optimizer_steps=166500,rows=18,predictions=3024000,metric_records=324,head_fit_updates=0,truth_last=True,target_feedback=False)
 (P/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 pct=lambda x:f'{100*x:.4f}'
 lines=['# SIG_TIME＋RDrop A–F 完整三种子结果','',f"F clean均值为 **{pct(summaries['F']['accuracy']['mean'])}% ± {pct(summaries['F']['accuracy']['sample_SD'])} pp**。F−A的描述性配对差为{statistics.mean(fa):+.6f} ± {statistics.stdev(fa):.6f}个百分点，{sum(x>0 for x in fa)}/3正差。预定F三种子84%均值门槛：{'达到' if summary['goal84'] else '未达到'}。",'', '所有结果来自固定三seed2026092701/02/03、final50默认头，未另训或替换头。按A–F原顺序报告全部正负结果，不从target分数选择父权重、重跑或改变训练。','', '|条件|seed01|seed02|seed03|clean均值 ± 样本SD|Macro-F1均值|最差RX均值|最差day均值|','|---|---|---|---|---|---|---|---|']
 for arm,s in summaries.items():
  lines.append('|'+arm+'|'+ '|'.join(pct(v['accuracy'])+'%' for v in s['per_seed'])+'|'+pct(s['accuracy']['mean'])+'% ± '+pct(s['accuracy']['sample_SD'])+' pp|'+pct(s['macro_f1']['mean'])+'%|'+pct(s['worst_RX_accuracy']['mean'])+'%|'+pct(s['worst_day_accuracy']['mean'])+'%|')
 lines+=['','|固定比较|seed01 / 02 / 03 差（百分点）|均值 ± 样本SD（百分点）|正差数|','|---|---|---|---|']
 for name,s in comparisons.items():
  values=[r.get('accuracy_difference_pp',r.get('interaction_pp')) for r in s['per_seed']];lines.append('|'+name+'|'+' / '.join(f'{v:+.6f}' for v in values)+'|'+f"{s['mean_pp']:+.6f} ± {s['sample_SD_pp']:.6f}"+'|'+str(s['positive_seeds'])+'/3|')
 b=comparisons['B-A'];ec=comparisons['E-C'];ed=comparisons['E-D'];fe=comparisons['F-E'];fb=comparisons['F-B'];inter=comparisons['F-E-B+A']
 lines+=['','B：相对A的平均差'+f"{b['mean_pp']:+.6f}个百分点，{b['positive_seeds']}/3正差。"+' 三路增加36,992个实际特征参数；该对照包含容量变化，不能单独证明信息多样性是收益来源。','', 'E：相对C为'+f"{ec['mean_pp']:+.6f}个百分点（{ec['positive_seeds']}/3正），相对D为{ed['mean_pp']:+.6f}个百分点（{ed['positive_seeds']}/3正）。"+' 统一公式与网络执行正确，不等于性能一定更高；E−D同时含单位、裁剪度量及origin物理衰减差异，不能仅归因于学习率。E与C在精确算术下针对同一物理更新，实际差异可来自浮点表示、EMA及门控轨迹，不预设历史涨幅相加。','', 'F：相对E为'+f"{fe['mean_pp']:+.6f}个百分点（{fe['positive_seeds']}/3正），相对B为{fb['mean_pp']:+.6f}个百分点（{fb['positive_seeds']}/3正）。"+' 交互为'+f"{inter['mean_pp']:+.6f} ± {inter['sample_SD_pp']:.6f}个百分点，{inter['positive_seeds']}/3正；"+('正平均交互与互补相符' if inter['mean_pp']>0 else '非正平均交互不支持超可加收益')+'，仅3个模型seed不能证明因果互补或统计显著性。','', '完整逐seed和全部RX/day/TX混淆矩阵见final_results.json；独立324个CM、同物理ID正确性及整数配对复算见final_target_recount.json。所有单种子均保留，样本SD使用ddof=1；包数不充当模型seed数。','', '成本：15个新模型各scratch50＝33L＋17U，共750模型轮／166,500更新；3个匹配A历史150模型轮／33,300更新单列。18行各168,000个clean决策，共3,024,000逻辑决策／324分层指标，A的504,000决策复用。默认头960参数均随骨干联合训练，单路f_theta195,041参数，三路232,033参数；Sinc统一方案不增加可训练参数，增加48个FP32 origin缓冲值。','', '|条件|源阶段均值秒 ± 样本SD|完整网络参数|部署f_theta参数|','|---|---|---|---|']
 for arm,s in summaries.items():lines.append('|'+arm+'|'+f"{s['source_elapsed_seconds']['mean']:.2f} ± {s['source_elapsed_seconds']['sample_SD']:.2f}"+'|'+str(s['per_seed'][0]['total_parameters'])+'|'+str(s['per_seed'][0]['feature_parameters'])+'|')
 lines+=['','源阶段时间包括原source函数的加载、训练和源验证，不含随后特征诊断、预测及评分；并发负载和共享GPU存在差异，不将墙钟差直接解释为算法加速。完整源日志与实际优化器记录重计166,500步；首批55,500步复用既有不可变审计，不重复训练或审计。U_s共14,458,500包曝光、TX真值隐藏。B/F的真实局部投影更新、student/EMA Sinc核及joint160去W一致性均由已保存checkpoint直接检查，分析未拟合任何分类头。','', '输入固定ManySig原7/63/30角色：L_s6,300、U_s56,700、V27,000；source RX1/3/4/6/8、day1/2/3、全部6TX，split392005，训练池标注10%。NoExplicitCFO及内部逐包补偿保留，普通增强、LS0.01、RDrop alpha1、旧batch_neighbor＋ownEMA＋cos固定；两个随机L前向共享同次实际增强IQ。','', '源扩种子决定'+summary['source_decision_commit']+'已在读取首个新目标汇总前提交推送并核对远端；十个剩余模型随后原样执行。源V只作健康检查，固定final50，无target选模、调参、停止或选择重跑。r01首步来源身份比较失败、0新训练，r02仅修正已核实A02/03技术恢复的run_id/output_dir，科学检查与配置不变；旧失败产物保留。','', '这是已暴露ManySig clean基准的默认头结果。证明了改动进入实际f_theta及可正确导出；没有新TX或Phase2实验，不宣称新类迁移、真实卫星或LEO部署收益。未进行被用户禁止的换头或诊断分类器训练。','', '代码执行提交09554c3a06c976c53efc4dbc61f5c501a1e7b06e；不可变运行和复现入口、完整配置在experiment.json及report.md。源代码与公式CPU/CUDA、真实checkpoint no-query smoke、一次P0/P1及定向修复已完成，不增加重复审批或hash链。']
 fa_f1=100*(summaries['F']['macro_f1']['mean']-summaries['A']['macro_f1']['mean'])
 fa_rx=100*(summaries['F']['worst_RX_accuracy']['mean']-summaries['A']['worst_RX_accuracy']['mean'])
 fa_day=100*(summaries['F']['worst_day_accuracy']['mean']-summaries['A']['worst_day_accuracy']['mean'])
 lines[2:2]=['**本轮没有证明 B、E、F 的总体平均准确率优于原 SIG_TIME＋RDrop 基线 A。** F 达到 84% 均值门槛，但相对 A 的准确率平均差为'+f'{statistics.mean(fa):+.6f} 个百分点，Macro-F1 平均差为 {fa_f1:+.6f} 个百分点。'+f'F 的最差 RX/day 均值分别高 {fa_rx:.6f}/{fa_day:.6f} 个百分点，种子间 SD 较小；这是三种子中的权衡，不能替代总体准确率增益或证明新 TX 迁移。','', '六条件均使用 RDrop：A＝单路＋RAW 标准优化器；B＝三路＋RAW 标准；C＝单路＋RAW 单位修正；D＝单路＋原 CENTER_25K；E＝单路＋统一居中及单位修正；F＝三路＋统一方案。','']
 lines+=['','结果与配对差图：[final_performance.png](final_performance.png)。完整 18 行×50 轮源曲线：[final_source_curves.png](final_source_curves.png)。逐种子表与全部配对表：[per_seed_metrics.csv](per_seed_metrics.csv)、[paired_differences.csv](paired_differences.csv)。','', '曲线保留全部 900 个源 epoch：监督 CE 后期仍有正常增强波动，源 V 大体进入平台；U 接收量在 E34 按原计划启用，没有用 U 的真实 TX 测量伪标签精度。源健康与 Sinc 参数运动不等于目标泛化改善，也不据此断言 50 轮已达到全局最优。','', '本轮判断：B 未形成稳定增益；E 的统一物理更新实现有效，但性能未优于 C/D；F 达到 84% 均值，且对 B 有平均增量，但相对 E 的增量不稳定、相对 A 没有总体均值提升。固定矩阵已完成，不追加目标反馈调参或选择性重跑。下一步若有新研发任务，先用合法源证据解释 RDrop 下的特征冗余与 Sinc 物理更新作用，再固定一个独立训练机制假设；不把 D 的目标均值当成新选模依据。']
 (P/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps(dict(status='VERIFIED',goal84=summary['goal84'],summary={a:dict(mean=s['accuracy']['mean'],sample_SD=s['accuracy']['sample_SD']) for a,s in summaries.items()},comparisons={k:{p:v[p] for p in ('mean_pp','sample_SD_pp','positive_seeds')} for k,v in comparisons.items()})))
if __name__=='__main__':main()
