"""Land verified fixed results and update only the six owned run records."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from experiments.cvs_receiver_residual_test_recovery import design as d
from experiments.cvs_receiver_residual_v2 import finalize_local as mirror
from experiments.cvs_receiver_residual_v2.publish import ssh


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def record(run,status,evidence,note):
    subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(d.ROOT),
        'record',run,'--status',status,'--evidence',evidence,'--note',note],check=True)


def main():
    artifact=d.ROOT/'local_artifacts'/d.SCORE_RELEASE;folder=artifact/'final_results'
    check=read(artifact/'final_verified.json')
    if check['status']!='VERIFIED':raise ValueError('No verified results')
    script='''
import json,os
from pathlib import Path
p=Path(PROJECT);r=p/'releases'/RELEASE
receipt=json.loads((r/'submit.json').read_text());pid=receipt['pid']
state=dict(status='VERIFIED',submit=receipt,completion=json.loads((r/'completion.json').read_text()),
    controller_alive=Path('/proc/'+str(pid)).exists(),failure_exists=(r/'failure.json').exists(),
    score_runs={x['run_id']:json.loads((p/'runs'/x['run_id']/'completion.json').read_text()) for x in RUNS})
print(json.dumps(state))
'''
    script=script.replace('PROJECT',repr(d.PROJECT)).replace('RELEASE',repr(d.SCORE_RELEASE)).replace('RUNS',repr(d.SCORE_RUNS))
    post=json.loads(ssh(script));write(artifact/'final_remote_readback.json',post)
    if post['completion']['status']!='ANALYZED' or post['failure_exists'] or any(v['status']!='ANALYZED' for v in post['score_runs'].values()):
        raise ValueError('Current remote state differs')
    write(artifact/'submit.json',post['submit'])
    notes=[]
    for run in d.SCORE_RUNS:
        report=d.ROOT/'automation_reports/CV-SincNet'/run['run_id'];source=folder/run['run_id']
        results=report/'results';results.mkdir(exist_ok=True)
        for p in source.rglob('*'):
            if p.is_file():
                out=results/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        evidence=report/'evidence';evidence.mkdir(exist_ok=True)
        for p,name in [(artifact/'final_remote_readback.json','final_remote_readback.json'),(artifact/'final_verified.json','final_verified.json'),
            (d.ROOT/'local_artifacts'/d.RELEASE/'completed_training_audit.json','completed_training_audit.json'),
            (artifact/'formal_training_diagnostics.json','formal_training_diagnostics.json')]:shutil.copyfile(p,evidence/name)
        write(evidence/'independent_review.json',dict(status='PASS',scope='scorer-only physicalRX/day metadata recovery',
            reviewer='/root/p0_p1_review',unresolved_P0_P1=[]))
        spec=read(report/'experiment.json');spec['code']['runtime_commit']=post['submit']['commit']
        spec['results_ref']='results/full_results.md';spec['execution']['scorer_pid']=post['submit']['pid']
        spec['execution']['actual_argv']=post['submit']['argv'];spec['execution']['gpu']=False
        write(report/'experiment.json',spec)
        (report/'report.md').write_text((report/'report.md').read_text(encoding='utf-8')+
            '\n## 最终独立测试结果\n\nANALYZED / VERIFIED：16模型×7视图，各168000样本。1568条总体/RX/TX主结果+448条完整日期结果。全部混淆矩阵指标、分区与四seed统计经本地独立重算。\n\n'
            '[完整本run结果](results/full_results.md)、[逐seed/RX/TX原始CSV](results/scores.csv)、[逐seed/day CSV](results/day_scores.csv)、[混淆矩阵JSON](results/scores.json)。\n\n'
            '原r03/r04预测保持固定；无重新训练/预测、适应或目标反馈。R1仅为随机混杂诊断；R2为正式四seed消融，四seed不声明统计显著性。Phase2/K/新增类/H均N/A。\n',encoding='utf-8')
        record(run['run_id'],'RUNNING','evidence/final_remote_readback.json','Independent readback reconciles actual CPU scorer launch PID; receipt parsing failure did not trigger duplicate submission')
        record(run['run_id'],'ANALYZED','evidence/final_verified.json','VERIFIED full1568+448 rows; fixed existing predictions; all confusion and strata recount; no target feedback')
        previous=d.ROOT/'automation_reports/CV-SincNet'/run['prediction_run_id']
        with (previous/'report.md').open('a',encoding='utf-8') as f:
            f.write('\n## 固定预测与独立评分交接\n\nPREDICTIONS_COMPLETE：本run的16×7固定数组保留。原评分在物理RX名称/编号检查处失败，未输出指标，旧failure保留。最终评分在独立run `'+run['run_id']+'` 完成，VERIFIED；无重预测。\n')
        prior=read(previous/'experiment.json');prior['results_ref']='../'+run['run_id']+'/results/full_results.md';write(previous/'experiment.json',prior)
        record(run['prediction_run_id'],'PREDICTIONS_COMPLETE','../'+run['run_id']+'/evidence/final_verified.json','All16x7 immutable predictions complete; original scorer failure retained; separate scoring run ANALYZED without new prediction')
        original=d.ROOT/'automation_reports/CV-SincNet'/run['parent_run_id']
        with (original/'report.md').open('a',encoding='utf-8') as f:
            f.write('\n## 完整测试已独立交付\n\n原训练控制器FAILED保留，16个ownE200模型源训练完整。其全部固定测试在预测run `'+run['prediction_run_id']+'` 与评分run `'+run['run_id']+'` 完成，结果VERIFIED。[完整结果](../'+run['run_id']+'/results/full_results.md)。原R1混杂处置不变。\n')
        orig=read(original/'experiment.json');orig['results_ref']='../'+run['run_id']+'/results/full_results.md';write(original/'experiment.json',orig)
        record(run['parent_run_id'],'FAILED','../'+run['run_id']+'/evidence/final_verified.json','Original pipeline schema failure preserved; allsourceE200 complete and fixed child scoring fully ANALYZED; consult linked result')
        notes.extend([run['run_id'],run['prediction_run_id'],run['parent_run_id']])
    subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(d.ROOT),'build','--managed-only'],check=True)
    for run_id in notes:
        mirror.d.RUN=run_id;mirror.REPORT=d.ROOT/'automation_reports/CV-SincNet'/run_id;mirror.mirror_root()
    print(json.dumps(dict(status='VERIFIED',canonical_mirror='E:/type10-7/automation_reports/CV-SincNet',runs=notes)))


if __name__=='__main__':main()
