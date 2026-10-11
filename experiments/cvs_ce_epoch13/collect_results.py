"""Collect already scored fixed rows; read neither IQ nor truth, never rescore."""
import argparse
import json
from pathlib import Path
import statistics
from . import design as d

PAYLOAD=r'''
import json,sys,time
from pathlib import Path
base=Path(BASE);sys.path.insert(0,RELEASE)
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_ce_epoch13.dispatch import require_scored
def read(p):return json.loads(p.read_text()) if p.is_file() else None
result=dict(run_id=RUN,read_at=time.time(),queue=read(base/'queue_state.json'),failure=read(base/'failure.json'),completion=read(base/'completion.json'),rows=[])
launch=read(base/'launch.json') or {'rows':[]}
for r in launch['rows']:
 root=base/r['row_id'];done=read(root/'completion.json');row=dict(row_id=r['row_id'],process=process(r['pid']),launch=r,completion=done)
 if done:
  if done['status']!='TRAINED_AND_SCORED':raise ValueError('Unexpected row completion')
  score=base/'row_scoring'/r['row_id'];scored=read(score/'scoring_complete.json');require_scored(scored,1)
  row['artifacts']={name:read(score/name) for name in ['scores.json','day_scores.json','parent_scores.json','paired_results.json','resources.json','scoring_complete.json']}
  row['source_completion']=read(root/'source/completion.json');row['freeze']=read(root/'source_frozen.json');row['prediction_completion']=read(root/'prediction/complete.json')
  if any(value is None for value in row['artifacts'].values()) or not row['source_completion'] or not row['freeze'] or not row['prediction_completion']:raise ValueError('Missing completed row artifacts')
 result['rows'].append(row)
if result['completion']:
 require_scored(read(base/'scoring_complete.json'),3)
 result['aggregate']={name:read(base/name) for name in ['scores.json','day_scores.json','parent_scores.json','summary.json','paired_results.json','compression_assessment.json','resources.json','scoring_complete.json']}
 if any(v is None for v in result['aggregate'].values()):raise ValueError('Missing final aggregate artifact')
print(json.dumps(result,allow_nan=False))
'''


def collect():
    from experiments.cvs_phase1_stack.publish import ssh
    p=PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE)).replace('RUN',repr(d.RUN))
    wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(p)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
    return json.loads(ssh(wrapper))


def analyze(value):
    records=[];days=[];parents={};resources=[];completed=[]
    for row in value['rows']:
        if not row['completion']:continue
        c=d.config(next(r for r in d.rows() if r['row_id']==row['row_id']));source=row['source_completion'];frozen=row['freeze']
        if source['config']!=c or source['optimizer_steps']!=c['updates'] or source['epoch']!=c['epochs'] or source['target_access']:raise ValueError('Completed source differs')
        if frozen['config']!=c or frozen['target_access'] or frozen['optimizer_steps']!=c['updates']:raise ValueError('Frozen source differs')
        a=row['artifacts'];rs=a['scores.json']['results'];ds=a['day_scores.json']['results']
        if len(rs)!=14 or len(ds)!=4 or any(r['row_id']!=row['row_id'] for r in rs+ds):raise ValueError('Row metric count or identity differs')
        for record in rs+ds:
            confusion=record['confusion'];n=sum(sum(r) for r in confusion);correct=sum(confusion[i][i] for i in range(len(confusion)))
            if abs(correct/n-record['accuracy'])>1e-12:raise ValueError('Accuracy/confusion mismatch')
        overall=next(r for r in rs if r['dimension']=='overall')
        if sum(sum(r) for r in overall['confusion'])!=168000:raise ValueError('Wrong fixed query count')
        records.extend(rs);days.extend(ds);resources.extend(a['resources.json']['rows']);completed.append(row['row_id'])
        for r in a['parent_scores.json']['results']:
            key=(r['arm'],r['model_seed'],r['dimension'],r['stratum'])
            if key in parents and parents[key]!=r:raise ValueError('Parent scores differ across repeated references')
            parents[key]=r
    summaries=[]
    for arm in ['P0',*d.ARMS]:
        rs=[r for r in records+list(parents.values()) if r['arm']==arm and r['dimension']=='overall']
        if not rs:continue
        values=[r['accuracy'] for r in rs]
        summaries.append(dict(arm=arm,seeds=[r['model_seed'] for r in rs],seed_count=len(rs),accuracy_mean=statistics.mean(values),accuracy_sd=statistics.stdev(values) if len(values)>1 else None,complete_three_seeds=len(rs)==3,per_seed=[dict(seed=r['model_seed'],accuracy=r['accuracy']) for r in rs]))
    if value['completion'] and len(completed)!=3:raise ValueError('Terminal run missing rows')
    if 'aggregate' in value:
        ar=value['aggregate']['scores.json']['results']
        key=lambda r:(r['row_id'],r['dimension'],r['stratum'])
        if sorted(ar,key=key)!=sorted(records,key=key):raise ValueError('Aggregate/row score mismatch')
    return dict(run_id=d.RUN,read_at=value['read_at'],status='COMPLETE' if len(completed)==3 and value['completion'] else 'PARTIAL',completed_rows=completed,missing_rows=[r['row_id'] for r in d.rows() if r['row_id'] not in completed],summary=summaries,records=records,day_records=days,parent_records=list(parents.values()),resources=resources,complete_arms_reaching_84=[r['arm'] for r in summaries if r['arm'] in d.ARMS and r['complete_three_seeds'] and r['accuracy_mean']>=.84],checks=dict(confusion_recount=True,source_config_budget=True,row_aggregate_agreement='aggregate' in value),target_feedback_scope='user-authorized epoch-budget search only')


def compression_decisions(result):
    parents={r['model_seed']:r['accuracy'] for r in result['parent_records'] if r['dimension']=='overall'}
    decisions=[]
    for arm in d.ARMS:
        values={r['model_seed']:r['accuracy'] for r in result['records'] if r['arm']==arm and r['dimension']=='overall'}
        seeds=[s for s in d.SEEDS if s in values and s in parents]
        delta=statistics.mean(values[s]-parents[s] for s in seeds) if seeds else None
        complete=len(seeds)==3
        decisions.append(dict(arm=arm,paired_seed_count=len(seeds),complete_three_seeds=complete,mean_accuracy_difference=delta,margin=.003,within_margin=(delta>=-.003) if complete else None))
    return decisions


def markdown(result):
    out=['# CE＋旧门控＋cos：压缩轮数 clean 测试','',f"状态 {result['status']}，完成 {len(result['completed_rows'])}/3 行。读取 Unix {result['read_at']}。每行 168000 个相同 query，独立 truth-last 评分，E200 复用固定原预测。",'', '|预算|种子1 (%)|种子2 (%)|种子3 (%)|实际均值 (%)|SD（百分点）|种子数|','|---|---:|---:|---:|---:|---:|---:|']
    for row in result['summary']:
        seeds={r['seed']:r['accuracy'] for r in row['per_seed']};cols=['N/A' if s not in seeds else f'{seeds[s]*100:.4f}' for s in d.SEEDS];sd='N/A' if row['accuracy_sd'] is None else f"{row['accuracy_sd']*100:.4f}"
        out.append('|'+row['arm']+'|'+'|'.join(cols)+f"|{row['accuracy_mean']*100:.4f}|{sd}|{row['seed_count']}/3|")
    out+=['','## 与 E200 配对比较','', '|预算|完整种子数|平均差（百分点）|下降不超过 0.3 个百分点|','|---|---:|---:|---|']
    for row in compression_decisions(result):
        delta='N/A' if row['mean_accuracy_difference'] is None else f"{100*row['mean_accuracy_difference']:+.4f}"
        verdict='待齐三种子' if row['within_margin'] is None else '满足' if row['within_margin'] else '不满足'
        out.append(f"|{row['arm']}|{row['paired_seed_count']}/3|{delta}|{verdict}|")
    out+=['','未完成种子不补值，部分种子均值不作为压缩决策。用户明确允许仅对 epoch 预算探索参考本测试集决定下一轮预算；该基准已暴露，不声称独立盲测或统计非劣效检验。源域 V 不用于本轮预算判断。','', '## 实测耗时','', '|行|训练分钟|更新次数|','|---|---:|---:|']
    for row in result['resources']:
        out.append(f"|{row['row_id']}|{row['training_seconds']/60:.2f}|{row['optimizer_steps']}|")
    out+=['','耗时来自当前共享 GPU 实测，不把 epoch 比例当作实测加速比。完整分层及混淆矩阵见 evidence/test_results_analysis.json。']
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();value=collect();result=analyze(value);result['compression_decisions']=compression_decisions(result);args.output.mkdir(parents=True,exist_ok=True)
    d.write(args.output/'evidence/test_results_readback.json',value);d.write(args.output/'evidence/test_results_analysis.json',result);(args.output/'test_results.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],completed=len(result['completed_rows']),summary=result['summary'],compression_decisions=result['compression_decisions'])))
