import statistics as st
import torch
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source
def analyze(base):
 L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config']);records=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id'];v=d.read(p/'completion.json');pr=torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False);mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()];y=V['y'][mask]
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or pr['ids']!=ids or pr['target_access'] or not torch.isfinite(pr['logits']).all() or v['parameters']!=177025:raise ValueError('Conditional moment source proof differs')
  cm=torch.bincount(y*6+pr['logits'].argmax(1),minlength=36).reshape(6,6)
  if cm.tolist()!=v['metrics']['confusion']:raise ValueError('Raw source confusion differs')
  records.append(dict(**row,metrics=v['metrics'],parameters=v['parameters']))
 control=[]
 for rx in d.RX:
  old=d.read(d.REFERENCE/f'CE30-RX{rx}-s{d.SEED}'/'completion.json')
  if old['config']!=d.q.config(next(r for r in d.q.rows() if r['arm']=='CE30' and r['held_rx']==rx)) or old['identity_updates']!=6660 or old['target_access'] or old['U_opened']:raise ValueError('Matching scratch30 CE control differs')
  control.append(dict(arm='CE30',held_rx=rx,metrics=old['metrics'],parameters=old['parameters']))
 summary=[dict(arm=a,mean=st.mean(r['metrics']['accuracy'] for r in records+control if r['arm']==a),worst_RX=min(r['metrics']['accuracy'] for r in records+control if r['arm']==a)) for a in ('CE30',*d.ARMS)];look={r['arm']:r for r in summary}
 gains={a:[next(r['metrics']['accuracy'] for r in records if r['arm']=='TX_CONDITIONAL' and r['held_rx']==rx)-next(r['metrics']['accuracy'] for r in records+control if r['arm']==a and r['held_rx']==rx) for rx in d.RX] for a in ('CE30','UNCONDITIONAL')}
 eligible=look['TX_CONDITIONAL']['mean']>=look['CE30']['mean']+.01 and look['TX_CONDITIONAL']['mean']>=look['UNCONDITIONAL']['mean']+.005 and all(sum(v>=0 for v in g)>=4 for g in gains.values()) and all(look['TX_CONDITIONAL']['worst_RX']>=look[a]['worst_RX']-.01 for a in gains)
 return dict(status='VERIFIED',models=10,identity_updates=66600,records=records,control=control,summary=summary,gains=gains,eligible=eligible,selected='TX_CONDITIONAL' if eligible else None,target_access=False)
def finish():
 a=analyze(d.BASE);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 普通 cos 分类头：浅中层条件矩对齐结果','','|方案|源 RX 留出均值|最差 RX|','|---|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"固定源晋级条件：{a['eligible']}。",'', '10 新模型从头各 30 轮、66600 更新，同一 native 身份网络和单一常规 cos 头，177025 参数。训练特征观察及批内统计仅用于浅中层辅助损失；推理没有 RX/TX 条件、附加分类器、路由、集成、源统计或持久对齐状态。U 和 target 未访问，held RX 整机排除。', '原 L/U/V=7/63/30，训练池 TX 标注率 10%；fit L5040，held V5400。CE 对照复用匹配的五份合法 scratch30 已固定源产物，仅作比较，不作初始化。Phase2 可丢弃旧类别权重，以 joint160 和合法 support 重拟合普通 cos 头；本次不执行 Phase2。', '对齐假设尚待数据检验，不预先宣称接收机不变性。只有源条件通过才自动执行固定三种子、每模型从头30及 clean+六 practical、prediction 固定后独立 truth-last；84% 仍待实际结果。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
