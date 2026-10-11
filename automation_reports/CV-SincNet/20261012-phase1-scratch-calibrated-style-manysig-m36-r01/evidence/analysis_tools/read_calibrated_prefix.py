"""Read all currently persisted source step records, never target artifacts."""
from pathlib import Path
import sys,json
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_scratch_calibrated_style import design as d
from experiments.cvs_phase1_stack.publish import ssh
payload=r'''
import json,sys,time,zlib,math,statistics
from pathlib import Path
base=Path(BASE);release=Path(RELEASE);sys.path.insert(0,str(release))
from experiments.cvs_scratch_calibrated_style import design as d
from experiments.cvs_receiver_residual_capacity4.control import process
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def stats(values):
 if not values:return dict(count=0,mean=None,min=None,max=None,median=None)
 return dict(count=len(values),mean=statistics.mean(values),min=min(values),max=max(values),median=statistics.median(values))
launch=read(base/'launch.json') or {'rows':[]};records=[];errors=[]
for launched in launch['rows']:
 rowid=launched['row_id'];source=base/rowid/'source';path=source/'step_metrics.jsonl.gz'
 if not path.is_file():continue
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));resolved=read(source/'resolved_config.json')
 if not resolved:continue
 if any(resolved.get(k)!=v for k,v in c.items()):raise ValueError('Actual config differs: '+rowid)
 proc=process(launched['pid'])
 if proc and proc['cwd']!=str(release):raise ValueError('Worker ownership differs')
 blob=path.read_bytes();stream=zlib.decompressobj(31);raw=stream.decompress(blob)
 # A live writer may be in the middle of one JSON line; only complete lines count.
 last=raw.rfind(b'\n');lines=raw[:last+1].splitlines() if last>=0 else []
 cap_values=[];ratios={g:[] for g in ('shallow','middle','rest')};raw_ratios={g:[] for g in ratios};caps=0;aux=0;violations=[]
 ce=[];kl=[];weighted_f=[];grad=[];styles=[];last_step=0;cap_seconds=[];limited=0
 for line in lines:
  item=json.loads(line);step=item['step'];detail=item['details'];m=item['metrics']
  if step!=last_step+1:raise ValueError('Noncontiguous source prefix '+rowid)
  last_step=step;cap=detail['joint_gradient_cap'];expected_aux=step>444 and step%4==0
  if bool(detail['auxiliary_steps'])!=expected_aux:raise ValueError('Auxiliary schedule mismatch')
  aux+=int(expected_aux)
  if bool(cap['applied'])!=bool(expected_aux and c['multilevel']['joint_gradient_cap']['enabled']):raise ValueError('Cap schedule mismatch')
  for key,values in [('ce',ce),('weighted_KL',kl),('weighted_fishr',weighted_f),('gradient_norm',grad),('weighted_style_ce',styles)]:
   value=m[key]
   if not math.isfinite(value):raise ValueError('Nonfinite source metric')
   values.append(value)
  if not cap['applied']:
   if cap['gradient_evaluations'] or cap['groups']:raise ValueError('Disabled cap did work')
   continue
  caps+=1;s=cap['scalar'];cap_values.append(s);cap_seconds.append(cap['seconds']);limited+=int(s<1)
  if not cap['scalar_detached'] or not cap['complete_disjoint_parameter_coverage'] or cap['gradient_evaluations']!=2:raise ValueError('Cap implementation contract differs')
  if sum(v['parameter_scalars'] for v in cap['groups'].values())!=resolved['trainable_parameters']:raise ValueError('Incomplete parameter partition')
  expected=min([1.]+[.2*v['norm_id']/(v['raw_R']+1e-12) for v in cap['groups'].values() if v['raw_R']>0])
  if not math.isclose(s,expected,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Incorrect cap scalar')
  if not math.isclose(cap['raw_weighted_fishr']*s,m['weighted_fishr'],rel_tol=1e-5,abs_tol=1e-7):raise ValueError('Effective training loss differs')
  for g,v in cap['groups'].items():
   if not math.isclose(v['norm_bound'],.2*v['norm_id'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Incorrect bound')
   if v['capped_R']>v['norm_bound']+1e-7*max(1,v['norm_bound']):violations.append(dict(step=step,group=g))
   if not math.isclose(v['capped_R'],s*v['raw_R'],rel_tol=1e-6,abs_tol=1e-9):raise ValueError('Incorrect capped gradient')
   if v['norm_id']>0:
    ratios[g].append(v['capped_R']/v['norm_id']);raw_ratios[g].append(v['raw_R']/v['norm_id'])
 if violations:raise ValueError('Observed gradient bound violations: '+rowid)
 records.append(dict(row_id=rowid,arm=c['arm'],model_seed=c['model_seed'],live_process=proc,source_prefix_steps=last_step,
  gzip_stream_finished=stream.eof,source_completion_present=(source/'completion.json').is_file(),captured_compressed_bytes=len(blob),
  discarded_trailing_unterminated_bytes=len(raw)-last-1,auxiliary_steps=aux,cap_steps=caps,cap_limited_steps=limited,
  scalar=stats(cap_values),cap_seconds=stats(cap_seconds),raw_ratio_to_identity={k:stats(v) for k,v in raw_ratios.items()},
  capped_ratio_to_identity={k:stats(v) for k,v in ratios.items()},bound_violations=violations,
  metrics={k:stats(v) for k,v in [('CE',ce),('weighted_KL',kl),('weighted_fishr',weighted_f),('gradient_norm',grad),('weighted_style_CE',styles)]},
  calibration_status=resolved['fishr_calibration']['status'],calibration_after_updates=resolved['fishr_calibration'].get('after_identity_updates')))
print(json.dumps(dict(status='VERIFIED_SOURCE_PREFIX',run_id=d.RUN,read_at=time.time(),source_only=True,target_access=False,
 scope='All complete lines of each captured persisted source gzip prefix; unfinished steps/rows not claimed complete; no target/scorer/stdout read',
 rows=records,total_prefix_steps=sum(r['source_prefix_steps'] for r in records),total_cap_steps=sum(r['cap_steps'] for r in records),errors=errors)))
'''.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
value=json.loads(ssh(wrapper));folder=R/'automation_reports/CV-SincNet'/d.RUN
d.write(folder/'evidence/source_prefix_audit.json',value)
lines=['# 在训源日志：联合梯度上限的实际执行','',value['scope'],'',f"读回时间 Unix {value['read_at']}，覆盖 {len(value['rows'])} 行已写入的 {value['total_prefix_steps']} 条步骤，含 {value['total_cap_steps']} 个实际限幅辅助步。不是完整训练结果，未读取目标评分或调整在训任务。",'',
 '|行|已写入步骤|辅助/限幅步|s均值/最小|限幅触发次数|限幅后各组最大比例|','|---|---:|---:|---:|---:|---|']
for row in value['rows']:
 s=row['scalar'];sc='N/A' if not s['count'] else f"{s['mean']:.6g}/{s['min']:.6g}"
 ratios=', '.join(g+':'+('N/A' if v['max'] is None else f"{v['max']:.6g}") for g,v in row['capped_ratio_to_identity'].items())
 lines.append(f"|{row['row_id']}|{row['source_prefix_steps']}|{row['auxiliary_steps']}/{row['cap_steps']}|{sc}|{row['cap_limited_steps']}|{ratios}|")
lines+=['','完整分组、原始/限幅后比例、标量、求导耗时和源损失统计见 evidence/source_prefix_audit.json。校验覆盖参数数目、联合标量公式、实际加权损失及各组范数界。s变小只表明运行时缩放生效，不证明抑制了有害方向或提高了泛化。该上限不保证AdamW位移。无辅助梯度组/未执行限幅记N/A，不填0冒充有效观测。']
(folder/'source_prefix_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],rows=len(value['rows']),steps=value['total_prefix_steps'],cap_steps=value['total_cap_steps'],violations=sum(len(r['bound_violations']) for r in value['rows']),target_access=False)))
