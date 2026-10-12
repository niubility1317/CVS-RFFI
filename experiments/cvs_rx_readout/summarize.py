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
  c=d.config(row);p=base/c['row_id'];done=d.read(p/'completion.json');pr=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False);mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()]
  if done!=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,new_training_updates=0,held_V_count=5400,source_only=True,target_access=False,parameters_buffers_unchanged=True) or pr['config']!=c or pr['ids']!=ids or pr['target_access'] or pr['new_training_updates'] or not pr['parameters_buffers_unchanged']:raise ValueError('Frozenreadout differs')
  old=torch.load(d.REFERENCE/f'ENSEMBLE-RX{c["held_rx"]}'/'source_predictions.pt',map_location='cpu',weights_only=False)
  if old['ids']!=ids or old['target_access']:raise ValueError('NativeIDs differ')
  for parent in ('RANDOM_HEAD','RX_HEAD'):
   z=pr['probabilities'][parent]
   if z.shape!=(5400,6) or not torch.isfinite(z).all() or (z<0).any() or not torch.allclose(z.sum(1),torch.ones(5400),atol=1e-5):raise ValueError('Finiteprobability required')
   probs=.5*z+.5*old['probabilities'];mp=base/(c['row_id']+'-MIXTURE-'+parent)/'source_predictions.pt'
   if not mp.exists():
    if not persist:raise ValueError('Missing frozenmixture')
    mp.parent.mkdir(exist_ok=False);torch.save(dict(probabilities=probs,ids=ids,target_access=False),mp)
   else:
    saved=torch.load(mp,map_location='cpu',weights_only=False)
    if saved['ids']!=ids or not torch.equal(saved['probabilities'],probs):raise ValueError('Sourcemix changed')
   metrics=score(probs.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);alone=score(z.clamp_min(1e-12).log(),V['y'][mask],V['rx'][mask],V['day'][mask]);records.append(dict(arm=c['arm'],parent=parent,held_rx=c['held_rx'],alone=alone,mixture=metrics))
 summary=[];qualified=[]
 for arm in d.ARMS:
  rr=[r for r in records if r['arm']==arm and r['parent']=='RX_HEAD'];ctrl=[r for r in records if r['arm']==arm and r['parent']=='RANDOM_HEAD'];mean=st.mean(r['mixture']['accuracy'] for r in rr);random=st.mean(r['mixture']['accuracy'] for r in ctrl);worst=min(r['mixture']['accuracy'] for r in rr)
  gains=[r['mixture']['accuracy']-next(x['ensemble_metrics']['accuracy'] for x in native['rows'] if x['held_rx']==r['held_rx']) for r in rr];paired=[r['mixture']['accuracy']-next(x['mixture']['accuracy'] for x in ctrl if x['held_rx']==r['held_rx']) for r in rr];eligible=mean>=native['ensemble_accuracy']+.01 and mean>=random+.005 and sum(x>=0 for x in gains)>=4 and sum(x>=0 for x in paired)>=4 and worst>=native['worst_ensemble']-.01 and worst>=min(r['mixture']['accuracy'] for r in ctrl)-.01
  item=dict(arm=arm,RX_alone_mean=st.mean(r['alone']['accuracy'] for r in rr),RX_mixture_mean=mean,RANDOM_mixture_mean=random,worst_RX=worst,gains=gains,paired=paired,eligible=eligible);summary.append(item)
  if eligible:qualified.append(item)
 selected=sorted(qualified,key=lambda r:(-r['RX_mixture_mean'],d.ARMS.index(r['arm'])))[0]['arm'] if qualified else None
 return dict(status='VERIFIED',source_only=True,target_access=False,new_training_updates=0,parents=10,source_predictions=162000,mixture_predictions=162000,native3_mean=native['ensemble_accuracy'],summary=summary,records=records,eligible=bool(qualified),selected=selected)
def finish():
 a=analyze(d.BASE,True);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 已冻结RX分类器：零训练读出源检验','','|读出|RX模型源均值|RX混合源均值|Random混合源均值|资格|','|---|---:|---:|---:|---|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['RX_alone_mean']:.4%}|{r['RX_mixture_mean']:.4%}|{r['RANDOM_mixture_mean']:.4%}|{r['eligible']}|")
 lines+=['',f"固定源选择：{a['selected']}。",'', '0训练/0优化/0校准，复用10个相同fitL5040/heldV5400 fixedfinal30 native4head父模型，真实物理角色/fit IDs/配置/继承严格核对。统一源RX训练相对随机分组已有+0.9778pp混合收益，本次只区分读出；新父模型不因成绩重新训练。', 'GLOBAL_ONLY仅原global；ENTROPY_GATE各专家熵softmax(-4H)作packet内共同权重再与global等分；CONFIDENT_HEAD最大maxprob专家再与global等分。规则和常数结果前固定，全部类同公式，无RX或querytruth/metadata/真实批次数/配额/全局重排，不更新query或parents。', '新readout再与native3固定0.5/0.5，source预测和混合先保存再source评分。完整15row后仅合格读出按source均值最高冻结；相对native3+1pp、random+0.5pp，各4/5非负，worst非劣1pp。所有负结果保留，无target，84仍待验证。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
