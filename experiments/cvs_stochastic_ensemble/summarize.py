import statistics as st
import torch
from . import design as d
from .model import probabilities
from experiments.cvs_quality_film import design as q
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_wave_geometry.run import score

def analyze(base,persist=False):
 L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config'])
 if U is not None:raise ValueError('Source ensemble U must stay closed')
 rows=[]
 for rx in d.RX:
  bc=q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==rx));bp=d.REFERENCE/d.baseline_id(rx);b=d.read(bp/'completion.json')
  if b['config']!=bc or b['identity_updates']!=6660 or b['target_access'] or b['U_opened'] or not b['parameters_buffers_unchanged']:raise ValueError('Frozen baseline source provenance differs')
  frozen=torch.load(bp/'source_predictions.pt',map_location='cpu',weights_only=False);ids=frozen['ids'];zz=[frozen['logits']];components=[b['metrics']]
  for member in (1,2):
   row=next(r for r in d.rows() if r['held_rx']==rx and r['member']==member);c=d.config(row);out=base/row['row_id'];v=d.read(out/'completion.json')
   if v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['held_V_count']!=5400:raise ValueError('Incomplete fixed native member')
   p=torch.load(out/'source_predictions.pt',map_location='cpu',weights_only=False)
   if p['ids']!=ids or p['target_access']:raise ValueError('All member physical IDs must match')
   zz.append(p['logits']);components.append(v['metrics'])
  mask=V['rx']==rx;wanted=[V['ids'][i] for i in mask.nonzero().flatten().tolist()]
  if ids!=wanted:raise ValueError('Held source physical IDs differ')
  probs=probabilities(zz)
  # All members already frozen; persist the final mixture before source truth score.
  predpath=base/f'ENSEMBLE-RX{rx}'/'source_predictions.pt'
  if not predpath.exists():
   if not persist:raise ValueError('Missing fixed source mixture prediction artifact')
   predpath.parent.mkdir(exist_ok=False);torch.save(dict(probabilities=probs,ids=ids,target_access=False,formula='uniform mean softmax T1'),predpath)
  else:
   old=torch.load(predpath,map_location='cpu',weights_only=False)
   if old['ids']!=ids or old['target_access'] or not torch.equal(old['probabilities'],probs):raise ValueError('Existing source mixture differs; never overwrite')
  metrics=score(probs.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);cm=metrics['confusion']
  if sum(map(sum,cm))!=5400 or abs(sum(cm[i][i] for i in range(6))/5400-metrics['accuracy'])>1e-7:raise ValueError('Mixture source score recount')
  mean=st.mean(m['accuracy'] for m in components)
  rows.append(dict(held_rx=rx,primary_seed=d.SEED,component_seed=[d.member_seed(d.SEED,i) for i in range(3)],component_metrics=components,
   mean_component_accuracy=mean,ensemble_metrics=metrics,gain=metrics['accuracy']-mean,baseline_CE30_accuracy=b['metrics']['accuracy']))
 avg=st.mean(r['mean_component_accuracy'] for r in rows);ensemble=st.mean(r['ensemble_metrics']['accuracy'] for r in rows)
 worst_avg=min(r['mean_component_accuracy'] for r in rows);worst_ens=min(r['ensemble_metrics']['accuracy'] for r in rows)
 gains=[r['gain'] for r in rows];eligible=ensemble>=avg+.01 and sum(v>=0 for v in gains)>=4 and worst_ens>=worst_avg-.01
 return dict(status='VERIFIED',new_models=10,reused_models=5,new_identity_updates=66600,historical_identity_updates=33300,
  primary_ensemble_seed_count=1,component_count=3,source_only=True,target_access=False,eligible=eligible,selected='EQUAL_PROBABILITY_3' if eligible else None,
  mean_component_accuracy=avg,ensemble_accuracy=ensemble,worst_mean_component=worst_avg,worst_ensemble=worst_ens,gains=gains,
  summary=[dict(arm='MEAN_COMPONENT',mean=avg,worst_RX=worst_avg,parameters=177025),dict(arm='EQUAL_PROBABILITY_3',mean=ensemble,worst_RX=worst_ens,parameters=531075)],
  rule=d.config(d.rows()[0])['ensemble']['scientific_rule'],rows=rows)
def finish():
 v=analyze(d.BASE,persist=True);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(eligible=v['eligible'],selected=v['selected'],target_access=False,automatic_target_launch=False))
 lines=['# 固定三成员概率集成：源端结果','','VERIFIED：一个primary模型种子的五RX源留出先导，每个ensemble含3个独立初始化的native。10个新成员各从零30轮，复用5个匹配成员0；不是三份独立ensemble种子统计。','','|方案|源V均值|最差RX|参数|','|---|---:|---:|---:|']
 for r in v['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|{r['parameters']}|")
 lines+=['',f"预登记源条件：{v['eligible']}；逐RX gain={v['gains']}。",'',
 '概率固定等权，softmax温度1，没有拟合权重、选择成员、阈值、源校准、U、教师、目标支持或更新。所有成员与所有类别使用同一规则，物理ID对齐后先固定最终混合概率再连接源V真值。',
 '集成的训练、参数和推理成本是单成员的3倍；普通组件均值用于区分方差收益，不宣称相同计算量胜出、物理不变性或RX纯解耦。',
 '只有固定源规则通过，才进入primary种子2026092701/02/03的三份全源30轮ensemble及逐row自动clean+六practical预测、独立truth-last。每份ensemble有3个内部初始化子种子；复用合法final30成员0，不将其用作其他成员的初始化。累计训练成本全部披露，84%仍待最终测试证明。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
