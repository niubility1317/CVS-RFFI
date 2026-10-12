import statistics as st
from . import design as d
from experiments.cvs_quality_film import design as q

def analyze(base):
 rows=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json');c=d.config(row)
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['held_V_count']!=5400:raise ValueError('Incomplete relation source row')
  if [x['level'] for x in v['actual_layers']]!=['full256_relation_branch']:raise ValueError('Actual branch missing')
  rows.append(dict(**row,metrics=v['metrics'],parameters=v['parameters'],reused=False))
 for rx in d.RX:
  c=q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==rx))
  v=d.read(d.REFERENCE/d.baseline_id(rx)/'completion.json')
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['held_V_count']!=5400:raise ValueError('Matched CE30 source artifacts differ')
  rows.append(dict(row_id=d.baseline_id(rx),arm='CE30',held_rx=rx,model_seed=d.SEED,metrics=v['metrics'],parameters=v['parameters'],reused=True,artifact_root=str(d.REFERENCE/d.baseline_id(rx))))
 for r in rows:
  m=r['metrics'];cm=m['confusion'];n=sum(map(sum,cm));a=sum(cm[k][k] for k in range(6))/n
  if n!=5400 or abs(a-m['accuracy'])>1e-7 or len(m['class_pairs'])!=15:raise ValueError('Source metrics recount differs')
 summary=[]
 for arm in ('CE30',*d.ARMS):
  rr=[r for r in rows if r['arm']==arm]
  summary.append(dict(arm=arm,mean=st.mean(r['metrics']['accuracy'] for r in rr),worst_RX=min(r['metrics']['accuracy'] for r in rr),parameters=rr[0]['parameters'],per_RX={str(r['held_rx']):r['metrics']['accuracy'] for r in rr}))
 look={r['arm']:r for r in summary};b=look['CE30'];eligible=[];pairs={}
 for arm in d.ARMS:
  x=look[arm];dv=[x['per_RX'][str(rx)]-b['per_RX'][str(rx)] for rx in d.RX];pairs[arm]=dv
  if x['mean']>=b['mean']+.01 and sum(v>=0 for v in dv)>=4 and x['worst_RX']>=b['worst_RX']-.01:eligible.append(arm)
 selected=max(eligible,key=lambda arm:(look[arm]['mean'],arm=='RAW_REL_BRANCH')) if eligible else None
 weak=look['COMPLEX_REL_BRANCH'];raw=look['RAW_REL_BRANCH'];dw=[weak['per_RX'][str(rx)]-raw['per_RX'][str(rx)] for rx in d.RX]
 attributed=weak['mean']>=raw['mean']+.005 and sum(v>=0 for v in dw)>=4 and weak['worst_RX']>=raw['worst_RX']-.01
 return dict(status='VERIFIED',new_models=10,reused_CE_models=5,new_identity_updates=66600,historical_matched_identity_updates=33300,
  source_only=True,target_access=False,summary=summary,paired_vs_CE=pairs,weak_vs_RAW=dw,eligible_arms=eligible,selected=selected,
  eligible=selected is not None,invariance_attribution_supported=attributed,rule=d.RULE,rows=rows)

def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(selected=v['selected'],eligible=v['eligible'],eligible_arms=v['eligible_arms'],target_access=False,automatic_target_launch=False,invariance_attribution_supported=v['invariance_attribution_supported']))
 lines=['# 完整波形＋关系互补分支：源端结果','','VERIFIED：10 个从零训练 30 轮模型；5 个同 seed/fitL/heldV/预算的 CE30 比较结果复用，不加载权重。','','|方案|留出源V clean均值|最差RX|可训练参数|','|---|---:|---:|---:|']
 for x in v['summary']:lines.append(f"|{x['arm']}|{x['mean']:.4%}|{x['worst_RX']:.4%}|{x['parameters']}|")
 lines+=['',f"预登记合格方案：{v['eligible_arms']}；source-only performance selection：{v['selected']}。",f"弱不变分支相对普通关系分支的独立机制条件：{v['invariance_attribution_supported']}。",'',
  '原网络全部保留，仅以零初始化有界残差接入同包投影关系，使用原 cosine 分类头。普通关系与弱幅相不变关系使用完全相同分支、参数和更新预算。两个分支均采用同一固定数值频率补偿，不将其称为已经核实的前导或 TX 振荡器恢复，不提供 CFO 标量给分类器。',
  '弱不变性只属于新增表示通道；整个模型保留原波形证据，不宣称全模型/RX 任意信道不变。没有 U、教师、目标支持、额外 CE、RX 对齐或硬投影。',
  '全部 15 个 TX 类对、各 day/RX、混淆和完整训练日志保留。源端 oracle 纠错潜力不等同可学习融合或目标收益。只有合格方案进入固定三种子全源30轮及自动clean＋六practical预测、独立truth-last评分。84% 尚未证明；不得按目标反馈调尺度、seed或选择性重跑。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
