import statistics as st
import torch
from . import design as d
from experiments.cvs_stochastic_ensemble.summarize import analyze as existing
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_wave_geometry.run import score
def analyze(base):
 old=existing(d.REFERENCE);L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config']);records=[]
 for row in d.rows():
  c=d.config(row);p=base/c['row_id'];v=d.read(p/'completion.json');pr=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False);mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()]
  if v!=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,count=5400,new_training_updates=0,target_access=False) or pr['config']!=c or pr['ids']!=ids or pr['target_access'] or pr['new_training_updates'] or pr['scores'].shape!=(5400,6) or not torch.isfinite(pr['scores']).all():raise ValueError('Frozenreadout differs')
  metrics=score(pr['scores'],V['y'][mask],V['rx'][mask],V['day'][mask]);baseline=next(r['ensemble_metrics']['accuracy'] for r in old['rows'] if r['held_rx']==c['held_rx']);records.append(dict(arm=c['arm'],held_rx=c['held_rx'],metrics=metrics,gain=metrics['accuracy']-baseline))
 summary=[];qualified=[]
 for arm in d.ARMS:
  rr=[r for r in records if r['arm']==arm];mean=st.mean(r['metrics']['accuracy'] for r in rr);worst=min(r['metrics']['accuracy'] for r in rr);eligible=mean>=old['ensemble_accuracy']+.01 and sum(r['gain']>=0 for r in rr)>=4 and worst>=old['worst_ensemble']-.01;item=dict(arm=arm,mean=mean,worst_RX=worst,gains=[r['gain'] for r in rr],eligible=eligible);summary.append(item)
  if eligible:qualified.append(item)
 selected=sorted(qualified,key=lambda r:(-r['mean'],d.ARMS.index(r['arm'])))[0]['arm'] if qualified else None
 return dict(status='VERIFIED',source_only=True,target_access=False,new_training_updates=0,source_predictions=81000,baseline_mean=old['ensemble_accuracy'],summary=summary,records=records,eligible=bool(qualified),selected=selected)
def finish():
 a=analyze(d.BASE);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 三成员集成固定读出：源端零训练检验','','|读出|源均值|最差RX|资格|','|---|---:|---:|---|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|{r['eligible']}|")
 lines+=['',f"源选择：{a['selected']}。",'', '0训练/0拟合/0校准，复用相同15个source fixedfinal30 native组成单primaryensemble种子的5RX留出，先固定81000新source决策再评分。MEAN_LOGIT几何共识；UNIT_MARGIN每packet每模型中心化6classlogits并L2规范化；MAJORITY单packet三模型预测票+0.001固定probmean破tie，非query真实批次数/配额。全部类同公式、query独立、不update父模型或query。', '与原固定probmean61.7778%对照，预登记+1pp、4/5非负、worst非劣1pp；完整15后合格最高均值才正式三primary已冻结9权重自动7views独立truthlast，不根据target成绩选择；84尚未达成。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
