import statistics as st
import torch
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source
def analyze(base):
 L,U,V,_=load_source(d.config(d.rows()[0])['source_loader_config']);records=[];controls=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id'];v=d.read(p/'completion.json');mask=V['rx']==c['held_rx'];ids=[V['ids'][i] for i in mask.nonzero().flatten().tolist()];y=V['y'][mask]
  if v['config']!=c or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['parameters']!=177025 or v['swa_snapshots']!=10:raise ValueError('SWA source proof differs')
  for name in d.READOUTS:
   pr=torch.load(p/(name+'_source_predictions.pt'),map_location='cpu',weights_only=False);cm=torch.bincount(y*6+pr['logits'].argmax(1),minlength=36).reshape(6,6)
   if pr['ids']!=ids or pr['target_access'] or not torch.isfinite(pr['logits']).all() or cm.tolist()!=v['readouts'][name]['confusion']:raise ValueError('SWA raw prediction differs')
   records.append(dict(readout=name,held_rx=row['held_rx'],metrics=v['readouts'][name],parameters=v['parameters']))
  old=d.read(d.REFERENCE/f"CE30-RX{c['held_rx']}-s{d.SEED}"/'completion.json')
  if old['config']!=d.q.config(next(r for r in d.q.rows() if r['arm']=='CE30' and r['held_rx']==c['held_rx'])) or old['identity_updates']!=6660 or old['target_access'] or old['U_opened']:raise ValueError('Frozen matching CE control differs')
  controls.append(dict(readout='CE30',held_rx=row['held_rx'],metrics=old['metrics'],parameters=old['parameters']))
 summary=[dict(readout=a,mean=st.mean(r['metrics']['accuracy'] for r in records+controls if r['readout']==a),worst_RX=min(r['metrics']['accuracy'] for r in records+controls if r['readout']==a)) for a in ('CE30',*d.READOUTS)];look={r['readout']:r for r in summary}
 gains={a:[next(r['metrics']['accuracy'] for r in records if r['readout']=='SWA' and r['held_rx']==rx)-next(r['metrics']['accuracy'] for r in records+controls if r['readout']==a and r['held_rx']==rx) for rx in d.RX] for a in ('CE30','FINAL')}
 eligible=look['SWA']['mean']>=look['CE30']['mean']+.01 and look['SWA']['mean']>=look['FINAL']['mean']+.005 and all(sum(v>=0 for v in g)>=4 for g in gains.values()) and all(look['SWA']['worst_RX']>=look[a]['worst_RX']-.01 for a in gains)
 return dict(status='VERIFIED',models=5,identity_updates=33300,records=records,controls=controls,summary=summary,gains=gains,eligible=eligible,selected='SWA' if eligible else None,target_access=False)
def finish():
 a=analyze(d.BASE);d.write(d.BASE/'analysis.json',a);d.write(d.BASE/'source_decision.json',dict(eligible=a['eligible'],selected=a['selected'],target_access=False));lines=['# 同轨迹平均：30轮源检验结果','','|模型|源RX均值|最差RX|','|---|---:|---:|']
 for r in a['summary']:lines.append(f"|{r['readout']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"固定source晋级：{a['eligible']}。",'', '5条新native轨迹从头各30/6660，总33300更新，末轮和21至30轮均值两个冻结模型分别单独预测，二者均177025参数/单普通cos；无U、teacher、EMA、增强、BN统计或专用分类器。源heldRX完全排除，原L/U/V=7/63/30、训练池TX标签10%。平均67个真实optimizer参数张量，不跨随机seed平均、不追加训练。', '前20轮coslr2e-4至5e-5，尾10轮交替1e-4/5e-5；FINAL匹配同轨迹隔离平均效果，原CE30控制同时披露日程差异。只有完整5RX源规则通过才三primaryseed同30轨迹自动FINAL/SWA clean+6practical七视图预测后独立truthlast，84待实际检验。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
