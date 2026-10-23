"""Resource-only CPU scoring handover; reuse all frozen scientific jobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from .publish_load_balance import REMOTE

RUN='20261023-phase1-PAMF-cpu-balance-control-r01'


def runtime_for(previous,commit):
    project=previous['project']
    rt=dict(previous,run_id=RUN,release_root=project+'/releases/'+RUN,
        control_root=project+'/runs/'+RUN,runtime_file=project+'/releases/'+RUN+'/runtime.json',
        code_commit=commit,launch_owner='codex-CPU-score-balance-20261023',
        previous_state=previous['control_root']+'/pipeline_state.json',
        previous_runtime=previous['runtime_file'],previous_control_root=previous['control_root'],
        previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        previous_controller_cwd=previous['release_root'],source_jobs=previous['jobs'],jobs=previous['jobs'],
        max_cpu_score_workers=4,poll_seconds=1)
    for name in ('G_teacher_acceleration','F_source_binding_recovery','supervised_extension',
                 'core_extension','external_dura_extension','pamf_gfactor_extension','physics_device_recovery'):
        rt[name]=False
    return rt


def publish(repo,output,previous_file,ssh_config):
    repo=Path(repo);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:
        raise ValueError('Remote branch OID differs')
    previous=json.loads(Path(previous_file).read_text(encoding='utf-8'))
    probe="""
import json,os
from pathlib import Path
rt=PREVIOUS
for _ in range(8):
 h=Path(rt['control_root'])/'multiprocess_handoff.json'
 if not h.is_file():break
 rt=json.loads(Path(json.loads(h.read_text())['new_runtime']).read_text())
else:raise ValueError('Too many controller handovers')
state=json.loads((Path(rt['control_root'])/'pipeline_state.json').read_text())
jobs=json.loads(Path(rt['jobs']).read_text())
pending=[j for j in jobs if state['jobs'][j['id']]['status'] in ('QUEUED','RUNNING')]
if not pending or any(j['kind']!='score' for j in pending):raise ValueError('Expected remaining CPU scoring jobs only')
if os.geteuid()==0:raise ValueError('Administrator forbidden')
print(json.dumps(dict(runtime=rt,pid=state['pid'],remaining=len(pending),counts=state.get('effective_counts'))))
""".replace('PREVIOUS',repr(previous))
    connection=['-F',str(ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    def ssh(code):
        result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=code.encode('utf-8'),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
        return json.loads(result.stdout)
    before=ssh(probe);previous=before['runtime'];rt=runtime_for(previous,commit)
    paths=['experiments/cric_dura_comparison/'+name for name in
           ('__init__.py','io.py','pipeline.py','source_queue.py','multiprocess_control.py','resource_balance.py')]
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Controller sources not committed')
    archive=output/(RUN+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for name,blob in [(p,git('show',commit+':'+p)) for p in paths]+[('release_commit.txt',(commit+'\n').encode())]:
            member=tarfile.TarInfo(name);member.size=len(blob);member.mode=0o644;tar.addfile(member,io.BytesIO(blob))
    package=dict(runtime=rt,expected_user=rt['project'].split('/')[2],
        archive_remote=rt['project']+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('runtime.json',rt),('preflight.json',before)]:
        with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    subprocess.run(['scp',*connection,str(archive),'N607:'+package['archive_remote']],check=True,timeout=55)
    receipt=ssh(REMOTE.replace('CONFIG',repr(package)))
    with (output/'submit.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(dict(status='SUBMITTED_NOT_YET_VERIFIED',receipt=receipt,remaining_before=before['remaining'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','output','previous','ssh-config'):p.add_argument('--'+name,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.previous,a.ssh_config)
