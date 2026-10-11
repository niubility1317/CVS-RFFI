"""Read every completed source step/epoch/stdout; no prediction/truth access."""
import argparse
import json
from pathlib import Path
from . import design as d

PAYLOAD=r'''
import json,gzip,math,re,time
from pathlib import Path
base=Path(BASE);rows=[];pending=[]
def read(p):return json.loads(p.read_text())
def numeric_ok(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(numeric_ok(v) for v in x.values())
 if isinstance(x,list):return all(numeric_ok(v) for v in x)
 return True
for registered in ROWS:
 rid=registered['row_id'];source=base/rid/'source'
 if not (source/'completion.json').is_file():pending.append(rid);continue
 done=read(source/'completion.json');c=done['config'];n=0;sums={};per_epoch={};gradients={};layers={};first={};details={};counts={k:0 for k in ('U_packets','physical_L','physical_U','feature_U','margin_U','weak_forwards','auxiliary_steps','auxiliary_L','auxiliary_forwards','fishr_steps','mixstyle_steps')}
 with gzip.open(source/'step_metrics.jsonl.gz','rt') as stream:
  for line in stream:
   item=json.loads(line);n+=1
   if item['step']!=n or not numeric_ok(item):raise ValueError('Nonconsecutive or nonfinite step '+rid)
   metrics=item['metrics'];detail=item['details'];epoch=per_epoch.setdefault(item['epoch'],dict(steps=0,sums={}))
   epoch['steps']+=1
   for key,value in metrics.items():
    if isinstance(value,(int,float)):
     sums[key]=sums.get(key,0.)+value;epoch['sums'][key]=epoch['sums'].get(key,0.)+value
   for key in counts:
    counts[key]+=detail[key]
    if detail[key] and key not in first:first[key]=n
   for key in ('U_confidence','unit_feature_std','raw_margin_mse','raw_feature_distance'):
    if key in detail:
     a=details.setdefault(key,dict(count=0,sum=0.,first=detail[key],last=None,min=None,max=None));value=detail[key]
     a['count']+=1;a['sum']+=value;a['last']=value;a['min']=value if a['min'] is None else min(a['min'],value);a['max']=value if a['max'] is None else max(a['max'],value)
   audit=item.get('gradient_audit')
   if audit:
    for loss,groups in audit['weighted_dense_gradient_norms'].items():
     for group,value in groups.items():
      key=loss+'/'+group;a=gradients.setdefault(key,dict(count=0,positive=0,sum=0.,min=None,max=None))
      a['count']+=1;a['positive']+=int(value>0);a['sum']+=value;a['min']=value if a['min'] is None else min(a['min'],value);a['max']=value if a['max'] is None else max(a['max'],value)
   multi=detail.get('multilevel')
   if multi:
    for layer,info in multi['levels'].items():
     a=layers.setdefault(layer,dict(steps=0,fishr_steps=0,variance_sum=0.,gap_sum=0.,mixed_steps=0,changed_packets=0,eligible_packets=0,delta_ratio_sum=0.))
     a['steps']+=1
     if 'variance_gap_mse' in info:a['fishr_steps']+=1;a['variance_sum']+=info['variance_mean'];a['gap_sum']+=info['variance_gap_mse']
     style=info.get('style')
     if style:a['mixed_steps']+=int(style['enabled']);a['changed_packets']+=style['changed_packets'];a['eligible_packets']+=style.get('eligible_packets',0);a['delta_ratio_sum']+=style.get('delta_ratio',0.)
 if n!=4440 or done['optimizer_steps']!=n or len(per_epoch)!=20 or any(e['steps']!=222 for e in per_epoch.values()):raise ValueError('Incomplete full source log '+rid)
 if any(done['mechanism_execution'][k]!=v for k,v in counts.items()):raise ValueError('Log/completion counter mismatch '+rid)
 epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
 if [e['epoch'] for e in epochs]!=list(range(1,21)):raise ValueError('Epoch coverage differs '+rid)
 for ep in epochs:
  aggregate=per_epoch[ep['epoch']]
  for key,value in aggregate['sums'].items():
   if key in ep and not math.isclose(ep[key],value/222,rel_tol=1e-9,abs_tol=1e-10):raise ValueError('Step/epoch metric mismatch '+rid+' '+key)
 stdout=(base/'logs'/f'{rid}.log').read_text(errors='replace')
 errors=[line[:500] for line in stdout.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|RuntimeError|ValueError|\bKilled\b',line)]
 warnings=[line[:300] for line in stdout.splitlines() if 'Warning:' in line or 'warning:' in line]
 for value in list(gradients.values())+list(details.values()):value['mean']=value['sum']/value['count']
 rows.append(dict(row_id=rid,arm=c['arm'],seed=c['model_seed'],steps=n,epochs=epochs,step_means={k:v/n for k,v in sums.items()},counts=counts,first_execution=first,detail_summary=details,gradient_summary=gradients,layer_summary=layers,stdout_errors=errors,stdout_warnings=warnings,stdout_lines=len(stdout.splitlines()),resources=done['resources'],distinct_U_visited=done['distinct_U_visited']))
print(json.dumps(dict(status='COMPLETE' if not pending else 'PARTIAL',read_at=time.time(),rows=rows,pending=pending,total_steps=sum(r['steps'] for r in rows),source_only=True,all_complete_step_and_stdout_records_read=True)))
'''

def collect():
    from experiments.cvs_phase1_stack.publish import ssh
    payload=PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows()))
    return json.loads(ssh(payload))

def report(value):
    lines=['# 全量源域训练诊断','',f"状态 {value['status']}；完整读取 {len(value['rows'])}/30 行，共 {value['total_steps']} 次源域训练记录。未完成行不从尾部样本推断完整训练。未读取目标预测或 truth。",'', '|方案/种子|L CE 首轮→末轮|U 接受数/128 末轮|辅助步数|Fishr 骨干梯度均值|耗时秒|','|---|---:|---:|---:|---:|---:|']
    for r in value['rows']:
        begin,end=r['epochs'][0],r['epochs'][-1];g=r['gradient_summary'].get('fishr/encoder',{}).get('mean',0.)
        lines.append(f"|{r['row_id']}|{begin['ce']:.5f} → {end['ce']:.5f}|{end['accepted_U']:.2f}|{r['counts']['auxiliary_steps']}|{g:.3g}|{r['resources']['identity_training_and_source_diagnostics_seconds']:.1f}|")
    lines+=['','Fishr 梯度均值包含预热阶段的零值；各次审计的非零计数和范围保留在 JSON。辅助 L 使用与 U 覆盖、每层实际交换率、方差差异、所有 epoch 的损失／学习率／V／权重位移均保留。V 仅作健康诊断，不用于选择 epoch 或方案。','', '待完成行：'+(', '.join(value['pending']) or '无。')]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect()
    d.write(a.output/'evidence/source_full_analysis.json',v);(a.output/'source_trajectory.md').write_text(report(v),encoding='utf-8')
    print(json.dumps(dict(status=v['status'],complete_rows=len(v['rows']),steps=v['total_steps'],errors={r['row_id']:r['stdout_errors'] for r in v['rows'] if r['stdout_errors']},source_only=True)))
