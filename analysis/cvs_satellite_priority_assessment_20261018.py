"""Read-only reporting of the user's satellite-first 0.1pp clean objective.
Consumes independently scored summaries, never IQ, model state or truth.
This does not select models/seeds or change source preregistration decisions.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
TOLERANCE=Fraction(1,1000)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def view(row,name):
 try:q=row['views'][name];n=q['count'];c=q['correct']
 except KeyError as e:raise ValueError('All seven independently scored views required') from e
 if type(n)!=int or type(c)!=int or n<=0 or not 0<=c<=n:raise ValueError('Invalid exact decision counts')
 if not math.isclose(q['accuracy'],c/n,abs_tol=1e-12,rel_tol=0):raise ValueError('Saved accuracy and exact counts differ')
 for k in ('macro_f1','worst_RX_accuracy'):
  if not math.isfinite(q[k]) or not 0<=q[k]<=1:raise ValueError('Invalid registered metric')
 return q,Fraction(c,n)
def paired_effects(summary,baseline,candidate,seed):
 result={}
 for q in summary.get('paired_effects',[]):
  if q['left']==candidate and q['right']==baseline:result[q['view']]=q
 if not result:
  for q in summary.get('paired_repairs',[]):
   if q['model_seed']==seed:result[q['view']]=q
 return result

def assess(summary,pairs):
 if summary.get('status')!='ANALYZED' or not pairs:raise ValueError('Completed independent analysis and explicit matched pairs required')
 grouped=defaultdict(list);seen=set();unique=set()
 for group,baseline,candidate in pairs:
  try:b=summary['rows'][baseline];a=summary['rows'][candidate]
  except KeyError as e:raise ValueError('Missing registered pair member') from e
  seed=b['model_seed']
  if seed!=a['model_seed'] or (group,seed) in seen:raise ValueError('Each group requires one unique matched pair per model seed')
  seen.add((group,seed));unique.add(seed)
  bc=b['source']['cost'];ac=a['source']['cost']
  for k in ('new_epochs','new_updates','L_identity_NN'):
   if bc[k]!=ac[k]:raise ValueError('Matched training budget differs: '+k)
  record=dict(model_seed=seed,baseline=baseline,candidate=candidate,views={},cost={'baseline':bc,'candidate':ac},paired_errors=paired_effects(summary,baseline,candidate,seed))
  delta={}
  for v in VIEWS:
   x,p=view(b,v);y,q=view(a,v)
   if x['count']!=y['count']:raise ValueError('Matched prediction population differs')
   delta[v]=q-p
   record['views'][v]=dict(baseline_accuracy_pct=100*float(p),candidate_accuracy_pct=100*float(q),accuracy_delta_pp=100*float(q-p),baseline_macro_F1_pct=100*x['macro_f1'],candidate_macro_F1_pct=100*y['macro_f1'],macro_F1_delta_pp=100*(y['macro_f1']-x['macro_f1']),baseline_worst_RX_accuracy_pct=100*x['worst_RX_accuracy'],candidate_worst_RX_accuracy_pct=100*y['worst_RX_accuracy'],worst_RX_delta_pp=100*(y['worst_RX_accuracy']-x['worst_RX_accuracy']))
  delta['urban_mean']=sum(delta[v] for v in VIEWS[-3:])/3
  record['urban_mean_delta_pp']=100*float(delta['urban_mean'])
  record['clean_tolerance_met']=delta['clean']>=-TOLERANCE
  record['_exact_delta']=delta;grouped[group].append(record)
 groups={}
 for group,records in grouped.items():
  records.sort(key=lambda x:x['model_seed']);paired={}
  for v in (*VIEWS,'urban_mean'):
   values=[100*float(r['_exact_delta'][v]) for r in records]
   paired[v]=dict(mean_pp=statistics.mean(values),sample_SD_pp=statistics.stdev(values) if len(values)>1 else None,seed_values_pp={str(r['model_seed']):val for r,val in zip(records,values)})
  clean_mean=sum(r['_exact_delta']['clean'] for r in records)/len(records)
  groups[group]=dict(model_seed_count=len(records),per_seed=records,paired=paired,clean_tolerance_mean_met=clean_mean>=-TOLERANCE,clean_tolerance_every_seed_met=all(r['clean_tolerance_met'] for r in records),all_observed_clean_positive=all(r['_exact_delta']['clean']>0 for r in records),all_observed_clean_gains_above_0_1pp=all(r['_exact_delta']['clean']>TOLERANCE for r in records),all_observed_low_mid_positive=all(r['_exact_delta'][v]>0 for r in records for v in VIEWS[-2:]),crossseed_evidence_available=len(records)>1)
  for r in records:del r['_exact_delta']
 return dict(status='REPORTED',absolute_clean_tolerance_pp=.1,unique_model_seed_count=len(unique),groups=groups,NN=0,IQ_reads=0,checkpoint_reads=0,truth_reads=0,updates=0,target_feedback=False,changes_source_decision=False,limits=['This is frozen-result reporting, not a promotion rule','Per-seed and mean clean tolerance are separate; no seed omitted','One model seed or sub-0.1pp gain cannot establish reproducible dual improvement','Different trajectory groups are not additional seeds; source decisions stay unchanged'])

def markdown(result):
 lines=['# 星地优先目标：冻结结果配对评估','','clean 容许下降 0.1 个百分点，按准确率绝对差计算。这里只补充报告，不改写源域决定或选择模型。','']
 for group,r in result['groups'].items():
  lines += ['## '+group,'',f"模型 seed 数：{r['model_seed_count']}。均值 clean 上限：{'满足' if r['clean_tolerance_mean_met'] else '不满足'}；每个 seed 均满足：{'是' if r['clean_tolerance_every_seed_met'] else '否'}。",'','|seed|clean Δ pp|low urban Δ pp|mid urban Δ pp|urban 均值 Δ pp|clean 上限|','|---|---:|---:|---:|---:|---|']
  for q in r['per_seed']:
   v=q['views'];lines.append(f"|{q['model_seed']}|{v['clean']['accuracy_delta_pp']:+.6f}|{v[VIEWS[-1]]['accuracy_delta_pp']:+.6f}|{v[VIEWS[-2]]['accuracy_delta_pp']:+.6f}|{q['urban_mean_delta_pp']:+.6f}|{'满足' if q['clean_tolerance_met'] else '不满足'}|")
  lines+=['','|视图|配对均值 pp|样本标准差 pp|','|---|---:|---:|']
  for v,q in r['paired'].items():
   sd='未估计（单 seed）' if q['sample_SD_pp'] is None else f"{q['sample_SD_pp']:.6f}"
   lines.append(f"|{v}|{q['mean_pp']:+.6f}|{sd}|")
  lines+=['','单 seed 或小于 0.1 pp 的正收益只是初步观察。可重复双提升需结合全部 seed 及波动判断，不能由阶段容许下降替代。','']
 lines+=['逐行准确率、宏 F1、最差 RX、已保存的修复／新增错误和完整成本保存在同目录 JSON。处理不读取 IQ、checkpoint 或 truth，不产生训练更新。','']
 return '\n'.join(lines)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--summary',required=True);ap.add_argument('--pair',action='append',required=True,help='group:baseline_row:candidate_row');ap.add_argument('--output',required=True);args=ap.parse_args()
 pairs=[tuple(v.split(':')) for v in args.pair]
 if any(len(p)!=3 or not all(p) for p in pairs):ap.error('--pair requires group:baseline_row:candidate_row')
 result=assess(read(args.summary),pairs);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (out/'assessment.md').write_text(markdown(result),encoding='utf-8')
 print(json.dumps(dict(status=result['status'],unique_model_seed_count=result['unique_model_seed_count'],groups={k:dict(model_seed_count=v['model_seed_count'],clean_tolerance_mean_met=v['clean_tolerance_mean_met'],clean_tolerance_every_seed_met=v['clean_tolerance_every_seed_met']) for k,v in result['groups'].items()})))
if __name__=='__main__':main()
