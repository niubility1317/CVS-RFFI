from pathlib import Path
import sys,json,shutil,time
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_e20_response import design as d
from experiments.cvs_phase1_stack.publish import ssh
artifact=R/'local_artifacts'/d.RELEASE;v=d.read(artifact/'readback.json')
assert v['status']=='VERIFIED' and not v['failure'] and not v['queue']['failures'] and v['preflight']['status']=='PASS' and len(v['preflight']['rows'])==3
assert all(row['response_identity_gradient_norm']>0 and row['identity_updates']==0 and not row['target_access'] and row['parameters_buffers_unchanged'] for row in v['preflight']['rows'])
script="import sys,json\nsys.path.insert(0,"+repr(d.PROJECT+'/releases/'+d.RELEASE)+")\nfrom experiments.cvs_multi_state_action.dispatch import capacity\nprint(json.dumps({str(g):dict(pids=sorted(s['pids']),free_mb=s['free_mb']) for g,s in capacity().items()}))\n"
wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(script)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
cap=json.loads(ssh(wrapper));assert len(cap)==8 and all(len(x['pids'])<=3 for x in cap.values());d.write(artifact/'capacity_readback.json',cap)
folder=R/'automation_reports/CV-SincNet'/d.RUN;s=d.read(folder/'experiment.json');s.update(status='RUNNING',execution_commit=v['submit']['commit'],delivery_verdict='VERIFIED')
s['code']['commit']=v['submit']['commit'];s['independent_review']['remote_source_preflight']='PASS'
workers={w['row_id']:w for w in v['workers']};resolved=0
for row in s['rows']:
 w=workers.get(row['row_id'])
 if w is None:row['status']='QUEUED';continue
 rc=w['resolved'];row.update(pid=w['pid'],gpu=w['gpu'],launch_evidence=w['process'])
 if rc:
  c=d.config(next(r for r in d.rows() if r['row_id']==row['row_id']));e=rc['execution']
  assert rc['config']==c and rc['commit']==v['submit']['commit'] and e['E20_weights_loaded'] and e['optimizer_reset'] and e['total_epochs']==40 and not e['EMA'] and not e['satellite_generator'] and not e['auxiliary_optimizer']
  assert e['U_iterator']==c['unlabeled'] and e['response_input_gradient']==c['response']['enabled'] and e['response_readers']==c['response']['tasks'] and not e['explicit_CFO_identity_input']
  row['resolved_config_verified']=True;resolved+=1
 if w['completion']:assert w['completion']['status']=='TRAINED_AND_SCORED';row['status']='TRAINED_AND_SCORED'
 else:
  assert w['process'] and w['process']['cwd']==d.PROJECT+'/releases/'+d.RELEASE
  assert 'experiments.cvs_e20_response.dispatch' in w['process']['argv'] and row['row_id'] in w['process']['argv']
  row['status']='RUNNING' if rc else 'STARTING'
s['progress']=dict(launched=len(workers),resolved_verified=resolved,queued=len(v['queue']['pending']),completed=len(v['queue']['completed']),failures=v['queue']['failures'],GPU_capacity=cap)
d.write(folder/'experiment.json',s)
for name in ['package.json','submit.json','readback.json','remote_before.json','capacity_readback.json']:
 if (artifact/name).exists():shutil.copy2(artifact/name,folder/'evidence'/name)
d.write(folder/'evidence/remote_preflight.json',v['preflight'])
shutil.copy2(__file__,folder/'evidence/launch_collection_script.py')
with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='RUNNING_VERIFIED',time=time.time(),execution_commit=v['submit']['commit'],dispatcher_pid=v['submit']['pid'],**s['progress']))+'\n')
with (folder/'report.md').open('a',encoding='utf-8') as f:f.write(f"\n发布 VERIFIED：{len(workers)}/21行已启动、{resolved}份实际配置已核对、{len(v['queue']['pending'])}行待资源。真实L/U三种子预检PASS，额外响应产生非零身份梯度、固定D及身份参数/buffer未更新，预检identity_updates=0。8GPU总CUDA进程均≤3。执行commit `{v['submit']['commit']}`。实际每row续训20/4440→固定自身E40→clean168000→truth-last评分已串联；目前尚无本批最终成绩，不把启动写成84%达成。\n")
p=R/'analysis/e20_response_traceability.md';t=p.read_text(encoding='utf-8');t+=f"\n远端发布 VERIFIED：真实源三种子父权重/peer/四作用读出来源预检PASS；身份响应梯度非零、固定D参数无梯度/无更新。实际{len(workers)}/21启动、{resolved} resolved、{len(v['queue']['pending'])}排队，8GPU总进程≤3。执行commit `{v['submit']['commit']}`。N01–N12的启动路径已经落实；完整20epoch训练/clean性能与全量收尾仍待完成，不能宣称84%。\n";p.write_text(t,encoding='utf-8')
d.write(folder/'handoff.json',dict(status='RUNNING',run_id=d.RUN,execution_commit=v['submit']['commit'],dispatcher=v['submit'],progress=s['progress'],next='Do not relaunch or edit immutable release. Eachrow fixed E20+20 automatically own40 clean168000 and independent truth-last; complete all21/full93240 source logs and420epoch, collect all7x3 scores/negative outcomes/cost, update original record and Git. Never target-driven teacher/parameter/arm selection.',evidence='evidence/readback.json and remote_preflight.json',goal='clean3seed mean>=84% remains active, not yet achieved'))
print(json.dumps(dict(status='VERIFIED',**s['progress']),ensure_ascii=False))
