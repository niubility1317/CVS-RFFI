"""Register/publish two support-only reproductions; never reuse an output root."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = "20260928-diagnostic-d42-support-manytx-m2-r01"
RELEASE = "d42_support_diagnostic_20260928_r01"
COMMIT = "281b92099b76233c1d4dc341372426894006bd84"
PROJECT = "/home/szu2070436088/2510044040/CV-SincNet"
PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CAPSULE = PROJECT + "/runs/20260927-phase2-practical-data-manytx-s2026092705-r01/capsule"
OLD = PROJECT + "/runs/20260927-phase2-cvs-d92-practical-manytx-m5-r01"
PATHS = ["code", "tools/cvs_d92_matched.py", "tools/cvs_d92_covariance_matched.py", "tools/diagnose_d42_support_only.py"]
REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run']
if release.exists() or run.exists():raise FileExistsError('Reconcile existing diagnostic; no resubmit')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer drift')
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe member')
 tar.extractall(project/'releases')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'code'),str(release/'tools')],cwd=release,env=env,check=True)
run.mkdir(); launches=[]
for row in c['rows']:
 log=run/(row['row_id']+'.log')
 with log.open('x') as stream:
  process=subprocess.Popen(row['argv'],cwd=release,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 launches.append(dict(row_id=row['row_id'],pid=process.pid,argv=row['argv'],log=str(log),cwd=str(release)))
(run/'launch.json').write_text(json.dumps(launches,indent=2))
print(json.dumps(launches))
'''


def prepare():
    source = ROOT / 'automation_reports/CV-SincNet/20260927-phase2-cvs-d92-practical-manytx-m5-r01/experiment.json'
    spec = json.loads(source.read_text(encoding='utf-8'))
    spec.update(run_id=RUN, group_id='d42-support-only-numerical-recovery', display_name='D42部署预测漂移的support-only复现',
        description='只复现两个技术失败row的首个失败support split；无query预测、评分、调参。', kind='diagnostic', stage='Phase2-support-fit',
        status='LOCAL_VERIFIED', tags=['d42','d92','diagnostic','support_only'], parent_run_ids=['20260927-phase2-cvs-d92-practical-manytx-m5-r01'],
        authorization='当前CVS实验技术恢复授权；主Agent指定d92_version_wiring为本诊断唯一launch owner。')
    spec['code'].update(commit=COMMIT,cwd=PROJECT+'/releases/'+RELEASE)
    spec['permissions'].update(query_use='forbidden; only support indices selected',claim_scope='Numerical failure diagnosis only; no accuracy claims')
    spec['checkpoint'].update(initialization='No checkpoint/model load; reuse exact frozen received-feature artifacts and matched v2 ground.',
        selection_rule='No selection; first failed split inferred from immutable failed log.')
    spec['execution']=dict(host='N607',launch_owner='codex/root/d92_version_wiring',gpu_policy='CPU only two processes one BLAS/Torch thread each',
        remote_run_root=PROJECT+'/runs/'+RUN,remote_log_root=PROJECT+'/runs/'+RUN,
        local_artifact_root='E:/type10-7/local_artifacts/'+RUN,
        launch_command='python tools/publish_d42_support_diagnostic.py launch --output E:/type10-7/local_artifacts/'+RUN,
        stop_rule='No automatic retry; exceptions and captured inner support fit are retained.')
    spec['metrics_plan']=dict(accuracy='not computed',diagnostic='FP64/FP32/sklearn mismatch counts, covariance condition, support-only coefficients')
    rows=[]
    for oldrow in spec['rows']:
        if oldrow['seeds']['model'] not in (392005,2026092703):continue
        row=copy.deepcopy(oldrow);seed=row['seeds']['model'];rowid=row['row_id']
        row.update(status='PLANNED',purpose='first_failed_support_split_reproduction',output_root=PROJECT+'/runs/'+RUN+'/'+rowid,
            log_path=PROJECT+'/runs/'+RUN+'/'+rowid+'.log',budget_ref='one failed support split; unchanged original method',
            expected_artifacts=['diagnostic.json','failed_inner_support_fit.npz'])
        row.pop('pid',None)
        row['argv']=[PYTHON,PROJECT+'/releases/'+RELEASE+'/tools/diagnose_d42_support_only.py','--row',OLD+'/'+rowid,
            '--capsule',CAPSULE,'--seed',str(seed),'--output',row['output_root']]
        row['command']=' '.join(row['argv']);rows.append(row)
    spec['rows']=rows;spec['expected_artifacts']=['launch.json','*/diagnostic.json','*/failed_inner_support_fit.npz']
    directory=ROOT/'automation_reports/CV-SincNet'/RUN;directory.mkdir(parents=True,exist_ok=False)
    (directory/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'report.md').write_text('# D42 support-only技术诊断\n\n只读取两失败row的固定support和新训练模型v2 ground。禁止query/truth或目标准确率参与。逐行配置见experiment.json。原失败输出保留。\n\n状态：LOCAL_VERIFIED。诊断代码commit '+COMMIT+'；本地compile通过；独立P0/P1由主Agent确认。\n',encoding='utf-8')
    (directory/'events.jsonl').write_text(json.dumps(dict(status='LOCAL_VERIFIED',note='support-only diagnostic preregistered; no launch yet'))+'\n',encoding='utf-8')


def launch(output):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=head:raise ValueError('Unpushed preregistration')
    spec=json.loads((ROOT/'automation_reports/CV-SincNet'/RUN/'experiment.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False);archive=output/(RELEASE+'.tar')
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),COMMIT,*PATHS],cwd=ROOT,check=True)
    c=dict(project=PROJECT,release=RELEASE,run=RUN,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),python=PYTHON,rows=spec['rows'])
    flags=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*flags,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    script=REMOTE.replace('CONFIG',repr(c));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*flags,'-T','N607','python3 -'],input=script.encode(),capture_output=True)
    (output/'landing.stdout').write_bytes(result.stdout);(output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','launch']);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:launch(a.output)
