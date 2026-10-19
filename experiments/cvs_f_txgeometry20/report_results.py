"""Describe frozen factorial scores; never consumes IQ or changes experiments."""
import argparse,json,math
from pathlib import Path
from statistics import mean,stdev

def stats(values):
 return dict(n=len(values),mean=mean(values) if values else None,sample_SD=stdev(values) if len(values)>1 else None)

def summarize(result):
 if result.get('status')!='ARTIFACTS_COMPLETE' or result.get('truth_last') is not True or result.get('head_fit_updates')!=0:raise ValueError('Complete fixed default-head truth-last results required')
 weak=result['source_expansion']['weak_indices']
 if len(weak)!=2 or len(set(weak))!=2 or any(k not in range(6) for k in weak):raise ValueError('Use the original source-only weak class set')
 records=result['records'];lookup={(r['arm'],r['model_seed']):r for r in records}
 if len(lookup)!=len(records):raise ValueError('Duplicate arm/seed')
 views=list(result['summary']['BASE']);details=[];paired=[];interactions=[];aggregate={}
 for record in records:
  for view in views:
   metric=record['metrics'][view];recall=metric['per_TX_recall']
   details.append(dict(row_id=record['row_id'],arm=record['arm'],model_seed=record['model_seed'],view=view,accuracy=metric['accuracy'],macro_f1=metric['macro_f1'],weak_TX_mean_recall=mean(recall[k] for k in weak),other_TX_mean_recall=mean(recall[k] for k in range(6) if k not in weak),per_TX_recall=recall,worst_RX=metric['worst_RX'],worst_day=metric['worst_day']))
   baseline=lookup.get(('BASE',record['model_seed']))
   if record['arm']=='BASE' or baseline is None:continue
   ref=baseline['metrics'][view];changes=[100*(x-y) for x,y in zip(recall,ref['per_TX_recall'])]
   paired.append(dict(row_id=record['row_id'],model_seed=record['model_seed'],arm=record['arm'],view=view,accuracy_pp=100*(metric['accuracy']-ref['accuracy']),macro_f1_pp=100*(metric['macro_f1']-ref['macro_f1']),weak_TX_mean_pp=mean(changes[k] for k in weak),other_TX_mean_pp=mean(changes[k] for k in range(6) if k not in weak),minimum_other_TX_pp=min(changes[k] for k in range(6) if k not in weak),per_TX_recall_pp=changes,worst_RX_pp=100*(metric['worst_RX']-ref['worst_RX']),worst_day_pp=100*(metric['worst_day']-ref['worst_day'])))
 for arm in ('BASE','WEIGHT','GEOMETRY','BOTH'):
  aggregate[arm]={}
  for view in views:
   values=[r for r in details if r['arm']==arm and r['view']==view];p=[r for r in paired if r['arm']==arm and r['view']==view]
   aggregate[arm][view]=dict(metrics={k:stats([r[k] for r in values]) for k in ('accuracy','macro_f1','weak_TX_mean_recall','other_TX_mean_recall','worst_RX','worst_day')},paired={k:stats([r[k] for r in p]) for k in ('accuracy_pp','macro_f1_pp','weak_TX_mean_pp','other_TX_mean_pp','minimum_other_TX_pp','worst_RX_pp','worst_day_pp')},weak_TX_positive_pairs=sum(r['weak_TX_mean_pp']>0 for r in p),overall_positive_pairs=sum(r['accuracy_pp']>0 for r in p),paired_n=len(p))
 for seed in sorted({r['model_seed'] for r in records}):
  if not all((arm,seed) in lookup for arm in ('BASE','WEIGHT','GEOMETRY','BOTH')):continue
  for view in views:
   values={arm:lookup[(arm,seed)]['metrics'][view] for arm in ('BASE','WEIGHT','GEOMETRY','BOTH')}
   recalls={arm:mean(values[arm]['per_TX_recall'][k] for k in weak) for arm in values}
   interaction=lambda v:100*(v['BOTH']-v['WEIGHT']-v['GEOMETRY']+v['BASE'])
   interactions.append(dict(model_seed=seed,view=view,accuracy_interaction_pp=interaction({a:m['accuracy'] for a,m in values.items()}),weak_TX_interaction_pp=interaction(recalls)))
 return dict(run_id=result['run_id'],weak_indices=weak,weak_set_origin='preregistered sourceV risk only; never target ranking',details=details,paired=paired,aggregate=aggregate,interactions=interactions,cost=result['cost'],source_expansion=result['source_expansion'],target_feedback=False,truth_last=True,head_fit_updates=0)

def render(summary):
 classes=['14-10','14-7','20-15','20-19','6-15','8-20'];weak=summary['weak_indices']
 percent=lambda x:f'{100*x:.4f}'
 lines=['# F＋跨 RX：弱发射机特征学习结果','',f"Run：{summary['run_id']}。弱类在目标访问前由源V固定为 "+'、'.join(classes[k] for k in weak)+'。以下结果只作冻结后的同划分配对评价，不参与调参、扩种子或重跑。','', '分类头全程按原流程联合训练；评分直接使用checkpoint内默认头。clean与六种practical每个视图各168000物理样本。未扩种子的组只报告实际种子数，单种子没有样本标准差。','', '## clean 逐种子结果','', '| 条件 | 模型 seed | Accuracy% | Macro-F1% | 弱类平均 recall% | 其余类平均 recall% | 最差 RX% | 最差 day% |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in summary['details']:
  if row['view']=='clean':lines.append('| '+row['arm']+' | '+str(row['model_seed'])+' | '+' | '.join(percent(row[k]) for k in ('accuracy','macro_f1','weak_TX_mean_recall','other_TX_mean_recall','worst_RX','worst_day'))+' |')
 lines+=['','## clean 相对同 seed BASE 的差值（百分点）','','| 条件 | seed | 整体 | 弱类平均 | 其余类平均 | 其余类最差变化 | 最差 RX | 最差 day |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in summary['paired']:
  if row['view']=='clean':lines.append('| '+row['arm']+' | '+str(row['model_seed'])+' | '+' | '.join(f'{row[k]:+.4f}' for k in ('accuracy_pp','weak_TX_mean_pp','other_TX_mean_pp','minimum_other_TX_pp','worst_RX_pp','worst_day_pp'))+' |')
 lines+=['','## clean 每 TX 配对差（百分点）','','| 条件 | seed | '+' | '.join(classes)+' |','|---|---:|'+ '|'.join(['---:']*6)+'|']
 for row in summary['paired']:
  if row['view']=='clean':lines.append('| '+row['arm']+' | '+str(row['model_seed'])+' | '+' | '.join(f'{x:+.4f}' for x in row['per_TX_recall_pp'])+' |')
 lines+=['','## 各视图汇总','','Accuracy 的 ± 为模型种子的样本标准差；n=1 显示“无 SD”。差值始终按同 seed 配对，弱类正差数单独列出。','','| 条件 | 视图 | n | Accuracy%±SD | Macro-F1% | 弱类 recall% | 最差 RX% | 最差 day% | 整体配对差 pp | 弱类配对差 pp | 弱类正差数/配对数 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for arm,views in summary['aggregate'].items():
  for view,value in views.items():
   metric=value['metrics'];acc=metric['accuracy'];n=acc['n']
   if not n:continue
   spread='无 SD' if acc['sample_SD'] is None else percent(acc['sample_SD'])
   delta=lambda key:'—' if value['paired'][key]['mean'] is None else f"{value['paired'][key]['mean']:+.4f}"
   lines.append('| '+arm+' | '+view+' | '+str(n)+' | '+percent(acc['mean'])+' ± '+spread+' | '+' | '.join(percent(metric[k]['mean']) for k in ('macro_f1','weak_TX_mean_recall','worst_RX','worst_day'))+' | '+delta('accuracy_pp')+' | '+delta('weak_TX_mean_pp')+f" | {value['weak_TX_positive_pairs']}/{value['paired_n']} |")
 lines+=['','## 成本与边界','','| row | 更新数 | 源训练秒 | 预测秒 | 峰值 CUDA MiB | L 曝光 | U 曝光 |','|---|---:|---:|---:|---:|---:|---:|']
 for row in summary['cost']:lines.append('| '+row['row_id']+' | '+' | '.join(str(round(row[k],3)) for k in ('optimizer_steps','source_seconds','prediction_seconds','peak_cuda_MiB','L_exposure','U_exposure'))+' |')
 lines+=['','完整逐种子七视图、每类差值、因子交互和源域扩种子决定见 results_summary.json。并行时间不等于各worker耗时之和。此前技术失败成本另见保留的r01/r02/r03失败记录；没有把失败状态用于初始化或根据目标分数选择重跑。','', '这是已暴露的 ManySig 跨接收机代理测试；本轮E20监督clean＋sat配方与历史E50旧门控／ownEMA的85.113889%不作严格配对。未证明新的盲测、Phase2迁移或真实卫星部署性能。','']
 return '\n'.join(lines)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
 result=json.loads(Path(args.result).read_text(encoding='utf-8-sig'));summary=summarize(result);folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
 for name,value in [('results_summary.json',json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),('results.md',render(summary))]:
  with (folder/name).open('x',encoding='utf-8') as stream:stream.write(value)
 print(json.dumps(dict(status='REPORTED',run_id=summary['run_id'],rows=len(result['records']),target_feedback=False)))
