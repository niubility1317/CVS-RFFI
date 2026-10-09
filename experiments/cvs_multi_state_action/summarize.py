"""Source-only complete-matrix reconciliation. Never opens prediction/truth."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
from experiments.cvs_multi_state_action import design as d

def csvwrite(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def source_summary(folder):
 folder=Path(folder);flat=[];rr=[];hold=[];steps=0;roles0=None
 for row in d.audit_rows():
  base=folder/row['row_id'];done=d.read(base/'completion.json')
  if done['status']!='SOURCE_AUDIT_COMPLETE' or not done['identity_unchanged'] or done['target_access'] or done['tx_holdouts']!=list(range(6)):
   raise ValueError('Incomplete audit row')
  roles=d.read(base/'physical_roles.json');ids={k:{r['id'] for r in roles[k]} for k in ('fit','audit')}
  if len(ids['fit'])!=1920 or len(ids['audit'])!=960 or ids['fit']&ids['audit']:raise ValueError('Physical roles differ')
  if roles0 is None:roles0=ids
  elif roles0!=ids:raise ValueError('Across seed physical roles differ')
  for name in ('clean','source_practical_mid','joint',*('heldout_tx'+str(i) for i in range(6))):
   a=d.read(base/(name+'_actions.json'));modes=d.AUDIT['heldout_modes'] if name.startswith('heldout') else d.AUDIT['modes']
   logs=[json.loads(s) for s in (base/(name+'_actions_steps.jsonl')).read_text().splitlines()]
   if len(logs)!=len(modes)*2*200:raise ValueError('Incomplete action update budget')
   steps+=len(logs)
   for kind in ('linear','temporal'):
    for mode in modes:
     records=[r for r in logs if r['kind']==kind and r['mode']==mode]
     if [r['step'] for r in records]!=list(range(1,201)):raise ValueError('Step coverage differs')
     for donor,result in a['kinds'][kind]['modes'][mode]['audit'].items():
      for stratum,m in result['strata'].items():
       if not m['count']:continue
       for space in ('h','z','normalized_z','margin'):
        v=m[space]
        if v['zero_mse']>1e-12 and abs(v['skill_vs_zero']-(1-v['mse']/v['zero_mse']))>1e-6:raise ValueError('Metric recount differs')
       if not name.startswith('heldout'):
        flat.append(dict(seed=row['model_seed'],fitting=name,kind=kind,mode=mode,donor=donor,stratum=stratum,count=m['count'],
         **{space+'_skill':m[space]['skill_vs_zero'] for space in ('h','z','normalized_z','margin')},margin_mae=m['margin_mae'],parameter_mse=m['parameter_mse']))
      if name.startswith('heldout'):
       tx=name.removeprefix('heldout_tx');m=result['tx_strata'].get(tx)
       if m:hold.append(dict(seed=row['model_seed'],heldout_tx=int(tx),kind=kind,mode=mode,donor=donor,count=m['count'],
        **{space+'_skill':m[space]['skill_vs_zero'] for space in ('h','z','normalized_z','margin')},margin_mae=m['margin_mae']))
   if name.startswith('heldout'):continue
   r=d.read(base/(name+'_receiver.json'))
   if r['status']!='VERIFIED' or r['steps']!=200 or r['identity_updated'] or r['target_access']:raise ValueError('R incomplete')
   steps+=r['optimizer_steps']
   rr.extend(dict(seed=row['model_seed'],fitting=name,**v) for v in r['predictive_metrics'])
 for name,rows in (('action_metrics',flat),('receiver_metrics',rr),('heldout_tx_metrics',hold)):
  csvwrite(folder/(name+'.csv'),rows)
 groups=defaultdict(list)
 for r in flat:
  if r['stratum'].endswith('/all'):groups[(r['fitting'],r['kind'],r['mode'],r['donor'],r['stratum'])].append(r)
 summary=[]
 for key,rs in groups.items():
  summary.append(dict(zip(('fitting','kind','mode','donor','stratum'),key),seeds=len(rs),
   **{k+'_mean':statistics.mean(r[k] for r in rs if r[k] is not None) if any(r[k] is not None for r in rs) else None for k in ('h_skill','z_skill','normalized_z_skill','margin_skill','margin_mae')}))
 result=dict(status='VERIFIED',rows=4,source_only=True,identity_unchanged=True,target_access=False,
  action_records=len(flat),receiver_records=len(rr),heldout_tx_records=len(hold),optimizer_steps=steps,
  metrics_recount=True,physical_roles_verified=True,summary=summary,
  identity_accuracy=None,decision='Predeclared known-parameter,state-conditioned,second-order,exact-reference architecture proceeds as fixed controls. Independent online calibration weights determine trusted virtual contributions; no score-driven retuning.',
  remaining='Identity optimization efficacy requires completed E200 and independent frozen target predictions.')
 d.write(folder/'source_action_summary.json',result)
 lines=['# 状态条件作用：完整源端诊断','','4个seed；分别拟合clean/源practical_mid与同参数联合拟合；6折辅助TX留出。',
  '所有结果均为作用预测指标，不是身份识别准确率。源物理包对新辅助拟合独立，原身份骨干训练见过这些源包。',
  '固定200步与相同增量/曝光日程；预算末仍下降不能解释为能力上限。已知p不含估计误差，但也不是作用模型能力上界。',
  '', '|拟合|分支|模式|donor|条件|G-skill|归一化G-skill|margin-skill|', '|---|---|---|---|---|---:|---:|---:|']
 for r in summary:
  if r['donor']!='cross_tx_matched':continue
  fmt=lambda k:'N/A' if r[k] is None else f'{100*r[k]:.3f}%'
  lines.append(f"|{r['fitting']}|{r['kind']}|{r['mode']}|{r['donor']}|{r['stratum']}|{fmt('z_skill_mean')}|{fmt('normalized_z_skill_mean')}|{fmt('margin_skill_mean')}|")
 lines+=['','完整own/同TX异包/匹配跨TX/不匹配跨TX/打乱参数、0/±p及分TX记录见action_metrics.csv；6折辅助TX留出见heldout_tx_metrics.csv；R强基线及重复分袋见receiver_metrics.csv与逐row原JSON。',
  '在线参考漂移、完整原生损失梯度及身份性能结论由后续身份矩阵产物提供，不由此诊断替代。']
 (folder/'source_action_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return {k:v for k,v in result.items() if k!='summary'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();print(json.dumps(source_summary(a.folder)))
