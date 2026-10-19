"""Publish one immutable raw-IQ source readout via the verified N607 route.

This entry point submits once.  The launch owner performs the independent
post-launch process/artifact reconciliation from the recorded submit handle.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

RUN='20261019-diagnostic-F-rawIQAdapter-sourceFit-e30-s2026092701-m6-r01'
RELEASE='cvs_F_rawIQAdapter_sourceFit_E30_20261019_r01'
REFERENCE_RELEASE='cvs_F_rawIQ_dualTeacher_E30_20261019_r01'
REFERENCE_COMMIT='6ea2f71f6654f4ceeadd460905a9d719b5dc1b89'
FROZEN_REFERENCE_RUN='20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01'
MAX_NN=4321
FILES=[
    'analysis/cvs_raw_iq_adapter_source_'+n+'_20261019.py'
    for n in ('fit','checks','publish')
]+[
    'analysis/cvs_dual_teacher_adapter_source_'+n+'_20261019.py'
    for n in ('panel','metrics')
]


def remote(script,rt):
    result=subprocess.run(['ssh','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',rt['python']+' -'],input=script.encode('utf8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
    return json.loads(result.stdout)


def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--artifacts',required=True)
    a=parser.parse_args();rt=json.loads(Path(a.runtime).read_text(encoding='utf-8-sig'))
    checkout=Path(__file__).resolve().parents[1];local=Path(a.artifacts)
    local.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=checkout,text=True).strip():
        raise ValueError('Commit actual diagnostic code before release')
    c=dict(project=rt['project'],python=rt['python'],user=rt['user'],host=rt['host'],commit=commit,
        release=rt['project']+'/releases/'+RELEASE,archive=rt['project']+'/releases/'+RELEASE+'.tar.gz',
        run=rt['project']+'/runs/'+RUN,reference_release=rt['project']+'/releases/'+REFERENCE_RELEASE,
        reference_commit=REFERENCE_COMMIT,
        frozen_reference_root=rt['project']+'/runs/'+FROZEN_REFERENCE_RUN,NN_packets=MAX_NN)
    script='''
import importlib.util,json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Immutable root exists; reconcile before any retry')
if shutil.disk_usage(c['project']).free<2*1024**3:raise ValueError('Insufficient output space')
ref=Path(c['reference_release'])
if (ref/'release_commit.txt').read_text().strip()!=c['reference_commit']:raise ValueError('Pinned raw-IQ release commit differs')
s=importlib.util.spec_from_file_location('_raw_sourcefit_parent',ref/'experiments/cvs_raw_iq_adapter30/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
from experiments.cvs_raw_iq_adapter30 import design as d
from experiments.cvs_multi_state_action.dispatch import capacity
decision=d.read(d.BASE/'source_matrix_decision.json')
if decision['status']!='SOURCE_DECISION_FIXED' or decision['run_id']!=d.RUN:raise ValueError('Prior raw-IQ source decision is not fixed')
roles=d.read(d.SOURCE)['role_ids']
if any(len(roles[k])!=n for k,n in [('L_s',6300),('U_s',56700),('V',27000)]):raise ValueError('Original source role handles differ')
for row in d.rows():
 cfg=d.config(row);p=Path(cfg['output_root'])
 if d.read(p/'completion.json')['config']!=cfg or d.read(p/'source_contract.json')['role_ids']!=roles:raise ValueError('Frozen fixed row config/source role differs')
 if not (p.parent/'source_qualification/complete.json').is_file():raise ValueError('Fixed raw-IQ source qualification is incomplete')
frozen=Path(c['frozen_reference_root'])
if d.read(frozen/'complete.json')['status']!='SOURCE_ONLY_ZERO_UPDATE_COMPLETE':raise ValueError('Frozen Tc/Ts source readout is incomplete')
if not all((frozen/n).is_file() for n in ('source_panel.json','analysis.json','predictions.npz')):raise ValueError('Frozen Tc/Ts source readout artifacts are missing')
states=capacity();resources={str(g):dict(processes=len(v['pids']),free_mb=v['free_mb']) for g,v in states.items()}
slots=[g for g in (4,6,7) if len(states[g]['pids'])<2 and states[g]['free_mb']>=12000]
if not slots:raise ValueError('No current registered GPU slot; no launch mutation')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,source_roles='EXACT_MATCH',fixed_rows='VERIFIED',source_decision_fixed=True,frozen_Tc_Ts_readout='AVAILABLE',GPU_capacity=resources,available=slots,NN=0,planned_NN_packets=c['NN_packets'],reused_Tc_Ts_NN=0,target_access=False)))
'''
    preflight=remote(script.replace('CONFIG',repr(c)),rt);write(local/'preflight.json',preflight)
    record='automation_reports/CV-SincNet/'+RUN+'/experiment.json'
    raw=subprocess.check_output(['git','archive',commit,*FILES,record,'automation_reports/CV-SincNet/'+RUN+'/report.md'],cwd=checkout)
    archive=local/(RELEASE+'.tar.gz')
    runtime=dict(run_id=RUN,project=c['project'],reference_release=c['reference_release'],reference_commit=REFERENCE_COMMIT,
        frozen_reference_root=c['frozen_reference_root'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
        for member in source.getmembers():
            if member.isfile():target.addfile(member,source.extractfile(member))
        for name,data in [('release_commit.txt',(commit+chr(10)).encode()),('runtime.json',json.dumps(runtime,indent=2).encode())]:
            member=tarfile.TarInfo(name);member.size=len(data);target.addfile(member,io.BytesIO(data))
    c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
    script='''
import hashlib,importlib.util,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Existing mutation; reconcile without resubmission')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive path')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'analysis')],cwd=release,env=env,check=True)
s=importlib.util.spec_from_file_location('_raw_sourcefit_capacity_parent',Path(c['reference_release'])/'experiments/cvs_raw_iq_adapter30/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
with lock(Path(c['project'])/'runs/receiver_residual_capacity4.lock'):
 states=capacity();slots=[(len(states[g]['pids']),-states[g]['free_mb'],g) for g in (4,6,7) if len(states[g]['pids'])<2 and states[g]['free_mb']>=12000]
 if not slots:raise ValueError('No GPU slot at launch; no worker started')
 gpu=min(slots)[2]
 command=[c['python'],'-u',str(release/'analysis/cvs_raw_iq_adapter_source_fit_20261019.py'),'--runtime',str(release/'runtime.json'),'--output',str(run)]
 log=release/'sourceFit.stdout.log'
 with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),GPU=gpu,commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',target_access=False,planned_NN_packets=c['NN_packets'],reused_Tc_Ts_NN=0)
(release/'submit.json').write_text(json.dumps(result,indent=2)+chr(10))
print(json.dumps(result))
'''
    result=remote(script.replace('CONFIG',repr(c)),rt);write(local/'submit.json',result)
    print(json.dumps({k:result[k] for k in ('status','pid','GPU','commit','archive_transfer','remote_compile','planned_NN_packets')}))


if __name__=='__main__':main()
