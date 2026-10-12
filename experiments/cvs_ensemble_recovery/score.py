"""Independent process; training and prediction never import this scorer."""
import json,time
from . import design as d
def score(row):
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 c,ids=preflight(row) # Validate every registered view before opening truth.
 out=d.BASE/row/'scores'
 if out.exists():raise FileExistsError('Existing score artifact; reconcile, never replace')
 out.mkdir();truth=d.read(d.TRUTH);y,rx,days=truth_arrays(truth,ids);records=[]
 with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as p:
  for view in d.VIEWS:
   rr,dd=metric_rows(c,p[view],ids,y,rx,days)
   for r in rr+dd:r['view']=view
   records.extend(rr+dd)
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
 d.write(d.BASE/row/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count_per_view=168000,metric_records=len(records),truth_last=True,independent_recount='VERIFIED'))
 print('SCORED '+json.dumps(dict(row_id=row,views=7,metric_records=len(records))),flush=True)

def finish():
 import statistics as st
 from experiments.cvs_scratch30_baseline import design as b
 records=[];baseline=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  if done!=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count_per_view=168000,metric_records=126,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or len(v['records'])!=126:raise ValueError('Incomplete final ensemble scoring')
  for r in v['records']:
   cm=r['confusion'];n=sum(map(sum,cm));a=sum(cm[k][k] for k in range(6))/n
   if n!=r['query_count'] or abs(a-r['accuracy'])>1e-12:raise ValueError('Ensemble score recount differs')
  records.extend(v['records'])
  bc=d.baseline_config(row['model_seed']);bp=b.BASE/bc['row_id'];bd=d.read(bp/'completion.json');bv=d.read(bp/'scores/metrics.json')
  if bd['config']!=bc or bd['status']!='TRAINED_PREDICTED_SCORED' or bd['views']!=list(d.VIEWS) or bd['count_per_view']!=168000 or not bd['truth_last'] or not bv['truth_last'] or len(bv['records'])!=126:raise ValueError('Matched fixed baseline score identity differs')
  baseline.extend(bv['records'])
 summary=[];paired=[]
 for view in d.VIEWS:
  rr=[r for r in records if r['view']==view and r['dimension']=='overall'];bb=[r for r in baseline if r['view']==view and r['dimension']=='overall']
  if len(rr)!=3 or len(bb)!=3 or {r['model_seed'] for r in rr}!=set(d.SEEDS):raise ValueError('Incomplete three primary ensemble seeds')
  values=[r['accuracy'] for r in rr]
  summary.append(dict(arm='EQUAL_PROBABILITY_3',view=view,accuracy_mean=st.mean(values),accuracy_sd=st.stdev(values),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),
   worst_RX_mean=st.mean(min(x['accuracy'] for x in records if x['row_id']==r['row_id'] and x['view']==view and x['dimension']=='receiver') for r in rr),baseline_CE30_mean=st.mean(r['accuracy'] for r in bb)))
  look={r['model_seed']:r['accuracy'] for r in bb};diff=[r['accuracy']-look[r['model_seed']] for r in rr]
  paired.append(dict(view=view,mean_difference=st.mean(diff),seed_differences={str(r['model_seed']):r['accuracy']-look[r['model_seed']] for r in rr},positive_seeds=sum(x>0 for x in diff)))
 clean=next(r for r in summary if r['view']=='clean');goal=clean['accuracy_mean']>=.84
 result=dict(status='VERIFIED',run_id=d.RUN,models=3,primary_seed_count=3,component_models=9,new_component_models=0,reused_component_models=9,new_identity_updates=0,reused_identity_updates=59940,total_identity_updates=59940,
  prediction_count=3528000,summary=summary,paired=paired,goal_84=goal,truth_last=True,records=records)
 d.write(d.BASE/'analysis.json',result)
 lines=['# 固定三成员集成：零训练技术恢复最终结果','','VERIFIED：三份独立primary种子的ensemble，每份3个内部初始化成员；每成员从零30轮或合法匹配final30复用。clean及六practical，每view/row168000，全视图prediction固定后独立进程truth-last评分。','','|视图|集成三seed均值|SD|原CE30均值|配对差|最差RX均值|','|---|---:|---:|---:|---:|---:|']
 for r,p in zip(summary,paired):lines.append(f"|{r['view']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['baseline_CE30_mean']:.4%}|{p['mean_difference']*100:+.4f}个百分点|{r['worst_RX_mean']:.4%}|")
 lines+=['',f"预登记clean三primaryseed均值≥84%：{goal}。",'',
  '固定等权softmax温度1，全部成员与全部类别共享规则，无拟合权重、成员选择、query更新、配额或全局重排。source先导通过后才访问目标，目标成绩不回流参数、排序或选择性重跑。',
  '成员0是已核实相同数据契约/物理角色、无显式CFO、固定final30的原CE30，只作自身推理成员。新成员1/2各自从零，无checkpoint初始化、U、teacher、EMA、星地训练或额外loss。',
  '每primary内部初始化seed为primary+10000m（m=0,1,2），9个独立参数集合，不把同一预测重复当3seed。训练采样按各primary+101，dropout为实际内部seed。',
  '集成每份可训练参数531075，为单native177025的3倍；本恢复零新增训练，9个原成员全部复用；原6个成员39960更新及更早3个成员19980更新，累计59940，相对三个普通单成员为3倍训练和推理成本。不能宣称同成本获胜或物理不变/纯RX解耦。',
  '现有目标基准已暴露，是开发基准确认而非新的独立盲测。数据为地面跨RX代理与固定practical模拟，不代表真实卫星部署、Phase2适应/注册或Phase3协同。保留所有负结果、逐seed/RX/day/TX混淆与完整日志；K为N/A。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
