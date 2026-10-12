import statistics as st
from pathlib import Path
from . import design as d
from experiments.cvs_quality_film import design as q
def analyze(base):
 records=[]
 for rx in d.RX:
  cc=q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==rx));old=d.read(Path(cc['output'])/'completion.json')
  if old['config']!=cc or old['identity_updates']!=6660 or old['target_access']:raise ValueError('Matching baseline differs')
  records.append(dict(arm='CE30',held_rx=rx,accuracy=old['metrics']['accuracy']))
  for arm in d.ARMS:
   c=d.config(next(r for r in d.rows() if r['arm']==arm and r['held_rx']==rx));v=d.read(base/c['row_id']/'completion.json')
   if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_truth_read'] or not v['parameters_buffers_unchanged'] or v['fit_U_count']!=45360 or v['distinct_U']!=45360 or v['counts']['EMA_updates']!=6660:raise ValueError('Incomplete source EMA row')
   records.append(dict(arm=arm,held_rx=rx,accuracy=v['metrics']['accuracy']))
 summary=[dict(arm=arm,mean=st.mean(r['accuracy'] for r in records if r['arm']==arm),worst_RX=min(r['accuracy'] for r in records if r['arm']==arm)) for arm in ('CE30',*d.ARMS)]
 look={r['arm']:r for r in summary};gains={other:[next(r['accuracy'] for r in records if r['arm']=='BOOTSTRAP_EMA' and r['held_rx']==rx)-next(r['accuracy'] for r in records if r['arm']==other and r['held_rx']==rx) for rx in d.RX] for other in ('CE30','SELF_EMA')}
 eligible=look['BOOTSTRAP_EMA']['mean']>=look['CE30']['mean']+.01 and look['BOOTSTRAP_EMA']['mean']>=look['SELF_EMA']['mean']+.005 and all(sum(v>=0 for v in values)>=4 for values in gains.values()) and all(look['BOOTSTRAP_EMA']['worst_RX']>=look[other]['worst_RX']-.01 for other in ('CE30','SELF_EMA'))
 return dict(status='VERIFIED',source_only=True,target_access=False,models=10,identity_updates=66600,summary=summary,records=records,gains=gains,eligible=eligible,selected='BOOTSTRAP_EMA' if eligible else None)
def finish():
 a=analyze(d.BASE);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 固定源集成向EMA移交：源端结果','','每个候选5个RX完整留出、单模型种子、从零固定30轮；heldRX同时排除L/U及teacher输入。','', '|方案|源V均值|最差RX|','|---|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"预登记源资格：{a['eligible']}。",'', '随机学生30轮，EMA每次student更新后更新，前5轮只L；第6轮开始U，历史2pass且confidence0.90，noise0.015、KL0.5五轮ramp。bootstrap固定3个原fitRX纯CE30教师软概率权重0.8从step1111线性移交到step4440的0，后10轮纯当前学生EMA。final30 student预测，未挑EMA或epoch。', '新增10模型66600更新，bootstrap独立继承源teacher90轮每RX的历史成本及U预测缓存成本，控制不加载旧教师。U真值隐藏；不访问target，不称84%已达成或新盲测。', '机制采用权重EMA，依据Tarvainen与Valpola Mean Teacher (2017)，本次handoff为待证假设：[原论文](https://arxiv.org/abs/1703.01780)。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
