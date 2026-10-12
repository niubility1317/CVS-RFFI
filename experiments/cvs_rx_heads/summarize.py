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
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or pr['ids']!=ids or v['head_partition_counts']!=[1260]*4:raise ValueError('RXheads frozen source differs')
  old=torch.load(Path(c['ensemble']['native_reference'])/'source_predictions.pt',map_location='cpu',weights_only=False)
  if old['ids']!=ids or old['target_access']:raise ValueError('Native frozen probability differs')
  probs=.5*pr['logits'].softmax(1)+.5*old['probabilities'];mp=base/(c['row_id']+'-MIXTURE')/'source_predictions.pt'
  if not mp.exists():
   if not persist:raise ValueError('Missingfrozenmixture')
   mp.parent.mkdir(exist_ok=False);torch.save(dict(probabilities=probs,ids=ids,target_access=False),mp)
  else:
   saved=torch.load(mp,map_location='cpu',weights_only=False)
   if saved['ids']!=ids or not torch.equal(saved['probabilities'],probs):raise ValueError('Sourcemixchanged')
  metrics=score(probs.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);records.append(dict(arm=c['arm'],held_rx=c['held_rx'],alone=v['metrics'],mixture=metrics,parameters=v['parameters']))
 summary=[dict(arm=arm,alone_mean=st.mean(r['alone']['accuracy'] for r in records if r['arm']==arm),mixture_mean=st.mean(r['mixture']['accuracy'] for r in records if r['arm']==arm),worst_RX=min(r['mixture']['accuracy'] for r in records if r['arm']==arm)) for arm in d.ARMS];look={r['arm']:r for r in summary}
 gains={other:[next(r['mixture']['accuracy'] for r in records if r['arm']=='RX_HEAD' and r['held_rx']==rx)-(next(r['mixture']['accuracy'] for r in records if r['arm']=='RANDOM_HEAD' and r['held_rx']==rx) if other=='RANDOM_HEAD' else next(r['ensemble_metrics']['accuracy'] for r in native['rows'] if r['held_rx']==rx)) for rx in d.RX] for other in ('RANDOM_HEAD','NATIVE3')}
 eligible=look['RX_HEAD']['mixture_mean']>=native['ensemble_accuracy']+.01 and look['RX_HEAD']['mixture_mean']>=look['RANDOM_HEAD']['mixture_mean']+.005 and all(sum(x>=0 for x in values)>=4 for values in gains.values()) and look['RX_HEAD']['worst_RX']>=native['worst_ensemble']-.01 and look['RX_HEAD']['worst_RX']>=look['RANDOM_HEAD']['worst_RX']-.01
 return dict(status='VERIFIED',models=10,identity_updates=66600,source_only=True,target_access=False,native3_mean=native['ensemble_accuracy'],summary=summary,records=records,gains=gains,eligible=eligible,selected='RX_HEAD' if eligible else None)
def finish():
 a=analyze(d.BASE,True);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 源RX分类边界专门化：源端结果','','|方案|单网络源均值|native3混合源均值|最差RX|','|---|---:|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['alone_mean']:.4%}|{r['mixture_mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"预登记源条件：{a['eligible']}。",'', '相同随机native骨干+4cos头，原globalCE+.25指定专家CE。RX_HEAD按4fitRX，RANDOM按固定均衡随机L划分，每头1260。推理全部专家统一概率平均，不输入目标RX/域标签，无queryfit或重排。固定后与native3按0.5/0.5混合；不校准温度或学融合权重。', '10新网络从零30、66600更新，匹配180865参数、相同额外读出和loss预算，U/增强/teacher/EMA/target关闭；heldRX整域训练排除。native3历史成本另计。未过源条件不访问目标，84尚未达成。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
