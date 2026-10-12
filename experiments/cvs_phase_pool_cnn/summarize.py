import statistics as st
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_stochastic_ensemble.summarize import analyze as native_analysis
from experiments.cvs_wave_geometry.run import score
def analyze(base,persist=False):
 native=native_analysis(d.REFERENCE);L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config']);records=[]
 for row in d.rows():
  c=d.config(row);p=base/c['row_id'];v=d.read(p/'completion.json');pr=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False);mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()]
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or pr['ids']!=ids or v['augmented_L_packets']!=c['mechanism']['augmentation_count']:raise ValueError('Frozenphase/poolsource differs')
  old=torch.load(Path(c['ensemble']['native_reference'])/'source_predictions.pt',map_location='cpu',weights_only=False)
  if old['ids']!=ids or old['target_access'] or old['formula']!='uniform mean softmax T1':raise ValueError('Native frozen probability differs')
  probs=.5*pr['logits'].softmax(1)+.5*old['probabilities'];mp=base/(c['row_id']+'-MIXTURE')/'source_predictions.pt'
  if not mp.exists():
   if not persist:raise ValueError('Missingfrozenmixture')
   mp.parent.mkdir(exist_ok=False);torch.save(dict(probabilities=probs,ids=ids,target_access=False),mp)
  else:
   saved=torch.load(mp,map_location='cpu',weights_only=False)
   if saved['ids']!=ids or not torch.equal(saved['probabilities'],probs):raise ValueError('Sourcemixchanged')
  metrics=score(probs.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);nr=next(x for x in native['rows'] if x['held_rx']==c['held_rx']);records.append(dict(arm=c['arm'],held_rx=c['held_rx'],CNN=v['metrics'],native3=nr['ensemble_metrics'],mixture=metrics,gain=metrics['accuracy']-nr['ensemble_metrics']['accuracy'],parameters=v['parameters']))
 summary=[];qualified=[]
 for arm in d.ARMS:
  rr=[r for r in records if r['arm']==arm];mean=st.mean(r['mixture']['accuracy'] for r in rr);worst=min(r['mixture']['accuracy'] for r in rr);eligible=mean>=native['ensemble_accuracy']+.01 and sum(r['gain']>=0 for r in rr)>=4 and worst>=native['worst_ensemble']-.01
  item=dict(arm=arm,CNN_mean=st.mean(r['CNN']['accuracy'] for r in rr),mixture_mean=mean,worst_RX=worst,eligible=eligible,parameters=rr[0]['parameters']);summary.append(item)
  if eligible:qualified.append(item)
 selected=sorted(qualified,key=lambda r:(-r['mixture_mean'],r['parameters'],r['arm']))[0]['arm'] if qualified else None
 return dict(status='VERIFIED',models=15,identity_updates=99900,source_only=True,target_access=False,native3_mean=native['ensemble_accuracy'],summary=summary,records=records,eligible=bool(qualified),selected=selected)
def finish():
 a=analyze(d.BASE,True);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 全局相位与统计池化：源端消融','','|方案|CNN源均值|固定混合源均值|最差RX|资格|','|---|---:|---:|---:|---|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['CNN_mean']:.4%}|{r['mixture_mean']:.4%}|{r['worst_RX']:.4%}|{r['eligible']}|")
 lines+=['',f"完整15row源选择：{a['selected']}。",'', '只在训练期对每个L包随机统一相位旋转，unitcomplex乘法保功率，非频偏、标签混合或星地增强；推理不旋转、不fitquery。POOL按通道time mean+populationstd，FLAT保留时间展平。两个因素分开消融，另保留原无augFLAT历史对照。', '所有学生从零固定30/6660，每row fitL5040/heldV5400、heldRX从训练排除、U/target关闭。混合固定0.5CNN+0.5native3，全部类别相同，无权重/温度拟合。15新模型99900更新，旧CNN/native历史另计；参数/计算非主目标，相同预算下真实泛化决定。','预登记资格vsnative3均值+1pp，4/5非负，worst非劣1pp；仅有资格候选按source均值最高冻结后可三seed正式7视图truthlast。84仍待验证，开发基准已暴露不称新盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
