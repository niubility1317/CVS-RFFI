from pathlib import Path
import sys,json,time,math,statistics,shutil
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_e20_response import design as d
from experiments.cvs_e20_response.source import expected_counts
from experiments.cvs_phase1_stack.publish import ssh
folder=R/'automation_reports/CV-SincNet'/d.RUN
payload=r'''
from pathlib import Path
import sys,json
p=Path(BASE);sys.path.insert(0,RELEASE)
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_e20_response.dispatch import require_scored
from experiments.cvs_e20_response import design as d
done=d.read(p/'completion.json');assert done['status']=='EXPERIMENT_COMPLETE' and done['trained_models']==done['tested_models']==21 and set(done['rows'])=={r['row_id'] for r in d.rows()}
require_scored(d.read(p/'scoring_complete.json'),21)
names=['completion.json','experiment.json','analysis.md','final_analysis.md','scoring_complete.json','scores.json','scores.csv','day_scores.json','day_scores.csv','parent_scores.json','summary.json','summary.csv','paired_results.json','resources.json','goal_assessment.json','evidence/source_full_analysis.json']
launch=d.read(p/'launch.json');submit=d.read(Path(RELEASE)/'submit.json');states=[]
for w in launch['rows']:
 row=w['row_id'];base=p/row;require_scored(d.read(p/'row_scoring'/row/'scoring_complete.json'),1)
 states.append(dict(row_id=row,pid=w['pid'],process=process(w['pid']),completion=d.read(base/'completion.json')))
 for name in ['resolved_config.json','initialization.json','source_contract.json','completion.json','epoch_metrics.jsonl','epoch_metrics.csv']:names.append(row+'/source/'+name)
 for name in ['source_frozen.json','completion.json','prediction/provenance.json','prediction/complete.json','prediction/resolved_config.json']:names.append(row+'/'+name)
 names.append('row_scoring/'+row+'/scoring_complete.json')
print(json.dumps(dict(files={name:(p/name).read_text(encoding='utf-8') for name in names},evidence=dict(dispatcher_process=process(submit['pid']),workers=states,completion=done,scoring=d.read(p/'scoring_complete.json')))))
'''
payload=payload.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
value=json.loads(ssh(wrapper));files=value['files'];ev=value['evidence'];source=json.loads(files['evidence/source_full_analysis.json']);scores=json.loads(files['scores.json'])['results'];summary=json.loads(files['summary.json'])['results'];goal=json.loads(files['goal_assessment.json'])
assert source['status']=='COMPLETE' and source['total_steps']==93240 and len(source['rows'])==21 and not source['target_feedback']
assert len(scores)==294 and len(json.loads(files['day_scores.json'])['results'])==84 and len(summary)==8 and all(x['complete_three_seeds'] and x['seed_count']==3 for x in summary)
assert all(w['completion']['status']=='TRAINED_AND_SCORED' and w['process'] is None for w in ev['workers']) and ev['dispatcher_process'] is None
by_source={r['row_id']:r for r in source['rows']};overall={}
for row in d.rows():
 c=d.config(row);s=by_source[row['row_id']];assert s['source_steps']==4440 and len(s['epochs'])==20 and s['complete_source_read'] and s['completion']['config']==c and s['completion']['counts']==expected_counts(c)
 assert [e['epoch'] for e in s['epochs']]==list(range(1,21)) and s['epochs'][-1]['counts']==expected_counts(c)
 assert len(s['gradient_audits'])==10 and s['completion']['distinct_U_visited']==(56700 if c['unlabeled'] else 0)
 assert s['completion']['resources']['response_parameters_buffers_unchanged'] and s['completion']['resources']['response_trainable_parameters']==0
 rec=[x for x in scores if x['row_id']==row['row_id']];assert len(rec)==14 and all(x['query_count']>0 for x in rec)
 for x in rec:
  matrix=x['confusion'];count=sum(sum(a) for a in matrix);correct=sum(matrix[i][i] for i in range(6));assert count==x['query_count'] and math.isclose(correct/count,x['accuracy'],abs_tol=1e-12)
 total=next(x for x in rec if x['dimension']=='overall');assert total['query_count']==168000;overall[row['arm'],row['model_seed']]=total
for x in summary:
 if x['arm'] in d.ARMS:assert math.isclose(x['accuracy_mean'],statistics.mean(overall[x['arm'],s]['accuracy'] for s in d.SEEDS),abs_tol=1e-12)
for name,text in files.items():
 if name=='experiment.json':continue
 p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
d.write(folder/'evidence/completion_readback.json',dict(status='VERIFIED',**ev));shutil.copy2(__file__,folder/'evidence/completion_collection_script.py')
s=d.read(folder/'experiment.json');assert s['status']=='RUNNING';s.update(status='ANALYZED',delivery_verdict='VERIFIED',progress=dict(completed_rows=21,source_updates=93240,source_epochs=420,predictions=3528000,metric_records=294,day_metric_records=84,goal_status=goal['performance_status'],complete_arms_reaching_goal=goal['complete_arms_reaching_goal'],target_feedback=False))
for row in s['rows']:row.update(status='TRAINED_AND_SCORED',resolved_config_verified=True)
d.write(folder/'experiment.json',s)
with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='COMPLETE_ANALYZED_VERIFIED',time=time.time(),**s['progress']))+'\n')
with (folder/'report.md').open('a',encoding='utf-8') as f:f.write('\nVERIFIED / ANALYZED：21/21完整训练与clean独立评分，93240追加身份更新、420完整epoch、3528000固定预测。全量步记录由远端实际analyzer消费，本地完整读回epoch/梯度审计/统计与294+84分层，混淆矩阵逐项独立重算准确率。primary RESP_LU 目标状态 '+goal['performance_status']+'。不根据target选择教师、配置或重跑，全部负结果保留。\n')
d.write(folder/'handoff.json',dict(status='ANALYZED',run=d.RUN,execution_commit=s['execution_commit'],results='final_analysis.md / scores.json / summary.csv',evidence='evidence/source_full_analysis.json and completion_readback.json',next='No relaunch/rescore. Full7x3 results retained. Source-only representation/gradient evidence can inform an independently preregistered later plan; never use target score rankings to choose teacher/arm/parameter or selective rerun.',goal_assessment=goal))
p=R/'analysis/e20_response_traceability.md';t=p.read_text(encoding='utf-8');t+='\n完整实际收尾 VERIFIED：21/21训练、93240追加更新、420epoch、3528000clean预测、294+84分层、严格自身E40与独立truth-last评分全部闭合。D参数/buffer不变，U各SSL行完整56700覆盖。实施闭合不等于有效泛化；性能目标状态 '+goal['performance_status']+'，完整结果和负结果保留。\n';p.write_text(t,encoding='utf-8')
print(json.dumps(dict(status='VERIFIED',goal=goal,summary=summary),ensure_ascii=False))
