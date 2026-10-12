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
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or pr['ids']!=ids:raise ValueError('CNN source frozen identity differs')
  old=torch.load(Path(c['ensemble']['native_reference'])/'source_predictions.pt',map_location='cpu',weights_only=False)
  if old['ids']!=ids or old['target_access'] or old['formula']!='uniform mean softmax T1':raise ValueError('Native frozen source mixture differs')
  probs=.5*pr['logits'].softmax(1)+.5*old['probabilities'];mp=base/f'MIXTURE-RX{c["held_rx"]}'/'source_predictions.pt'
  if not mp.exists():
   if not persist:raise ValueError('Missing frozen source mixture')
   mp.parent.mkdir(exist_ok=False);torch.save(dict(probabilities=probs,ids=ids,target_access=False,formula=c['ensemble']['formula']),mp)
  else:
   saved=torch.load(mp,map_location='cpu',weights_only=False)
   if saved['ids']!=ids or not torch.equal(saved['probabilities'],probs):raise ValueError('Source mixture changed')
  metrics=score(probs.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);nr=next(x for x in native['rows'] if x['held_rx']==c['held_rx'])
  records.append(dict(held_rx=c['held_rx'],CNN=v['metrics'],native3=nr['ensemble_metrics'],mixture=metrics,gain=metrics['accuracy']-nr['ensemble_metrics']['accuracy']))
 summary=[dict(arm=arm,mean=st.mean(r[key]['accuracy'] for r in records),worst_RX=min(r[key]['accuracy'] for r in records)) for arm,key in (('NATIVE3','native3'),('CNN','CNN'),('MIXTURE_CNN_NATIVE3','mixture'))];look={x['arm']:x for x in summary};eligible=look['MIXTURE_CNN_NATIVE3']['mean']>=look['NATIVE3']['mean']+.01 and sum(r['gain']>=0 for r in records)>=4 and look['MIXTURE_CNN_NATIVE3']['worst_RX']>=look['NATIVE3']['worst_RX']-.01
 return dict(status='VERIFIED',models=5,identity_updates=33300,source_only=True,target_access=False,summary=summary,records=records,eligible=eligible,selected='MIXTURE_CNN_NATIVE3' if eligible else None)
def finish():
 a=analyze(d.BASE,True);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 独立全波形卷积骨干互补：源端结果','','|方案|源V均值|最差RX|','|---|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"预登记source资格：{a['eligible']}。",'', '5个CNN均从零固定30轮、整RX源V留出；full256逐层卷积保留时间布局，固定RMS/lag20数值补偿，不向classifier提供CFO标量、不称任意RX不变或已知激励。概率0.5CNN+0.5冻结native3，全部类别同规则，无拟合融合参数。', '新增5CNN33300更新，复用15native历史99900更新；四模型混合不称相同成本优势。所有负结果保留，0target、U关闭；只有固定源条件通过才三seed全源30及自动7views独立truthlast。84尚未达成。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
