import statistics as st
import torch
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source
def analyze(base):
 L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config']);records=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id'];v=d.read(p/'completion.json');pr=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False);mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()];y=V['y'][mask]
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened']!=c['source_U'] or v['U_truth_read'] or not v['parameters_buffers_unchanged'] or pr['ids']!=ids or pr['target_access'] or not torch.isfinite(pr['logits']).all() or v['parameters']!=177025 or v['counts']['L_packets']!=852480 or v['counts']['U_packets']!=(710400 if c['source_U'] else 0):raise ValueError('CrossRX feedback source proof differs')
  cm=torch.bincount(y*6+pr['logits'].argmax(1),minlength=36).reshape(6,6)
  if cm.tolist()!=v['metrics']['confusion']:raise ValueError('Raw source confusion differs')
  records.append(dict(**row,metrics=v['metrics'],parameters=v['parameters'],accepted_U=v['counts']['accepted_U'],last_epoch=v['epochs'][-1]))
 summary=[dict(arm=a,mean=st.mean(r['metrics']['accuracy'] for r in records if r['arm']==a),worst_RX=min(r['metrics']['accuracy'] for r in records if r['arm']==a)) for a in d.ARMS];look={r['arm']:r for r in summary}
 gains={a:[next(r['metrics']['accuracy'] for r in records if r['arm']=='META_SSL' and r['held_rx']==rx)-next(r['metrics']['accuracy'] for r in records if r['arm']==a and r['held_rx']==rx) for rx in d.RX] for a in ('CE_MATCHED','UNIFORM_SSL')}
 eligible=look['META_SSL']['mean']>=look['CE_MATCHED']['mean']+.01 and look['META_SSL']['mean']>=look['UNIFORM_SSL']['mean']+.005 and all(sum(v>=0 for v in g)>=4 for g in gains.values()) and all(look['META_SSL']['worst_RX']>=look[a]['worst_RX']-.01 for a in gains)
 return dict(status='VERIFIED',models=15,identity_updates=99900,records=records,summary=summary,gains=gains,eligible=eligible,selected='META_SSL' if eligible else None,target_access=False)
def finish():
 a=analyze(d.BASE);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 普通 cos 头：跨 RX 元反馈源端结果','','|方案|源 RX 留出均值|最差 RX|','|---|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"固定源晋级条件：{a['eligible']}。",'', '15 新 native 单模型从头各30轮、99900更新，同一单常规cos头、177025参数。每步128个合法L TX/day均衡样本，4个fitRX各32，其中1个RX轮换反馈；SSL的128个U只来自其余3RX、U真值隐藏；整个外层heldRX的L/U均排除。反馈RX是训练域，不是外层验证。', 'UNIFORM/META匹配L/U/前向/梯度诊断预算；CE_MATCHED没有U昂贵分支，计算成本分别实报。META用冻结当前AdamW二阶矩的P内积近似，对4个32样本微批的梯度做非负有界重加权；不是单样本元导数、不是精确AdamW虚拟风险，也不证明伪标签正确。分类权重不参与反馈打分，但普通CE/SSL仍训练它；Phase2可丢弃并support-only重拟合。', '实际raw/weighted KL、RX/day接受率、预测类分布、P梯度内积、加权U/L骨干梯度比与全步日志保留。前三seed旧SELF全量遥测及U概率/物理输入普查，有限45组梯度探针另见source_diagnostics.json；探针不冒充全U梯度结论。', '原L/U/V=7/63/30，训练池标签率10%。所有源方案固定后按原V评分，未通过不访问目标，负结果保留。通过才固定三primaryseed全源单模型scratch30及自动clean+六practical预测后独立truthlast。84%仍为待证目标，benchmark已暴露，不称新盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
