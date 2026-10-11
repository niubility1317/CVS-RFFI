from pathlib import Path
import sys,json
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_scratch_calibrated_style import design as d
from experiments.cvs_phase1_stack.publish import ssh
payload=r'''
from pathlib import Path
import json,zlib,time,statistics
base=Path(BASE);launch=json.loads((base/'launch.json').read_text());rows=[]
for item in launch['rows']:
 root=base/item['row_id']/'source';p=root/'step_metrics.jsonl.gz';resolved=root/'initial_resolved_config.json'
 if not p.is_file() or not resolved.is_file():continue
 c=json.loads(resolved.read_text());obj=zlib.decompressobj(31);raw=obj.decompress(p.read_bytes());end=raw.rfind(b'\n')
 records=[json.loads(line) for line in raw[:end+1].splitlines()[:444]]
 if len(records)!=444:continue
 if [r['step'] for r in records]!=list(range(1,445)):raise ValueError('Missing warmup steps')
 for r in records:
  x=r['details'];m=r['metrics']
  if any(x[k] for k in ('U_packets','auxiliary_steps','mixstyle_steps','style_search_forwards')):raise ValueError('Unexpected warmup mechanism')
  if abs(m['total_loss']-m['ce'])>1e-7:raise ValueError('Warmup loss is not pure CE')
 rows.append(dict(row_id=item['row_id'],seed=c['model_seed'],flags=c['backend_flags'],ce=[r['metrics']['ce'] for r in records],
  gradient=[r['metrics']['gradient_norm'] for r in records]))
comparisons=[]
for seed in sorted({r['seed'] for r in rows}):
 group=[r for r in rows if r['seed']==seed];reference=group[0]
 for row in group:
  delta=[abs(a-b) for a,b in zip(row['ce'],reference['ce'])]
  comparisons.append(dict(row_id=row['row_id'],seed=seed,reference_row=reference['row_id'],steps=444,flags=row['flags'],
   CE_max_abs_difference=max(delta),CE_mean_abs_difference=statistics.mean(delta),CE_first_difference_over_1e_7=next((i+1 for i,v in enumerate(delta) if v>1e-7),None),
   CE_step1_difference=delta[0],CE_step444_difference=delta[-1],gradient_step1_difference=abs(row['gradient'][0]-reference['gradient'][0]),
   CE_step444=row['ce'][-1],reference_CE_step444=reference['ce'][-1]))
print(json.dumps(dict(status='VERIFIED_SOURCE_WARMUP',read_at=time.time(),source_only=True,target_access=False,rows=comparisons,
 scope='All first444 persisted pure-CE updates for each available row; losses/gradient norms only, no checkpoint equality or test effect inferred')))
'''.replace('BASE',repr(d.BASE.as_posix()))
wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
value=json.loads(ssh(wrapper));f=R/'automation_reports/CV-SincNet'/d.RUN;d.write(f/'evidence/warmup_pairing_audit.json',value)
lines=['# 相同种子的前444步配对范围','',
 '逐行读取前444条源训练记录，验证U、辅助约束、MixStyle均未执行，总损失等于CE。对同seed按固定启动次序第一行作数值参照；没有读取目标结果。','',
 '|行|参照行|CE首次差异>1e-7的步数|CE最大绝对差|CE平均绝对差|第444步CE差|','|---|---|---:|---:|---:|---:|']
for row in value['rows']:
 lines.append(f"|{row['row_id']}|{row['reference_row']}|{row['CE_first_difference_over_1e_7']}|{row['CE_max_abs_difference']:.7g}|{row['CE_mean_abs_difference']:.7g}|{row['CE_step444_difference']:.7g}|")
lines+=['','配置使用同一模型seed、固定物理划分、独立确定的L采样流与学习率日程；实际cuDNN deterministic=False。因此同seed配对不意味着GPU执行轨迹逐位相同。这里观测到的差异不能被归因于尚未启用的辅助项；也尚未单独证明由哪个算子导致。',
 '校准时点报告保留不同方案系数范围；不能把这些运行视为共享同一份第444步权重的严格反事实。固定实验按既定三种子完整报告；不热修改数值策略、重启健康任务或按结果筛选种子。后续若研究数值可重复性，应单独预登记源域重复性诊断。']
(f/'warmup_pairing_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],rows=len(value['rows']),steps=444*len(value['rows']),max_CE_difference=max(r['CE_max_abs_difference'] for r in value['rows']),target_access=False)))
