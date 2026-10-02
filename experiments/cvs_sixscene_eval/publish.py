"""Publish committed files once; read back the remote state independently."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_sixscene_eval.common import RUN,RELEASE,PROJECT

ROOT=Path(__file__).resolve().parents[2]
CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
REMOTE=r'''
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release']
run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if release.exists() or run.exists() or logs.exists(): raise FileExistsError('Reconcile existing delivery; no resubmit')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']: raise ValueError('Archive transfer mismatch')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host/user mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not (release/member.name).resolve().is_relative_to(release.resolve()) or not member.isfile():
            raise ValueError('Unsafe archive member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_identity_ce'),str(release/'experiments/adv3b02_xuc/code'),str(release/'baselines/common'),str(release/'code')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_selected_concat')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_sixscene_eval.views','--help'],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
spec=release/'experiments/cvs_sixscene_eval/configs/launch_spec.json'
d=json.loads(spec.read_text())
if d['run_id']!=c['run'] or len(d['rows'])!=8: raise ValueError('Unexpected matrix')
for row in d['rows']:
    cfg=json.loads(Path(row['config']).read_text())
    for name in ['last.pt','completion.json','initialization.json','source_contract.json','resolved_config.json']:
        if not (Path(cfg['source_output'])/name).is_file():raise FileNotFoundError(name)
    if not Path(cfg['source_contract']).is_file():raise FileNotFoundError(cfg['source_contract'])
if not (Path(d['capsule'])/'manifest.json').is_file() or not Path(d['truth']).is_file():raise FileNotFoundError('Test inputs')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_sixscene_eval')],cwd=release,env=env,check=True)
command=[python,'-u','-m','experiments.cvs_sixscene_eval.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

def ssh(script):
    compile(script,'remote','exec')
    result=subprocess.run(['ssh',*CONNECTION,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+'\n'+result.stdout.decode('utf-8',errors='replace'))
    return result.stdout


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit: raise ValueError('Remote branch differs from HEAD')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    selected=[n for n in names if (any(n.startswith('experiments/'+p+'/') for p in ('cvs_sixscene_eval','cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity','cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity','cvs_residual_identity','cvs_clean_design','cvs_identity_ce')) or n.startswith('baselines/cvcnn_ce/') or
        n.startswith('experiments/adv3b02_xuc/code/') or n.startswith('baselines/common/') or
        n.startswith('code/leo_practical/') or n in {'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'})
        and Path(n).suffix in {'.py','.json'}]
    prefixes=['experiments/'+p+'/' for p in ('cvs_sixscene_eval','cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity','cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity','cvs_residual_identity','cvs_clean_design','cvs_identity_ce','adv3b02_xuc/code')]
    prefixes+=['baselines/common/','baselines/cvcnn_ce/','code/leo_practical/','code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py']
    dirty=subprocess.check_output(['git','status','--porcelain','--',*prefixes],cwd=ROOT,text=True)
    if dirty.strip(): raise ValueError('Uncommitted release code/config')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists(): raise FileExistsError('Existing archive; reconcile delivery before retry')
    with tarfile.open(archive,'w:gz') as tar:
        for name in selected:
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            item=tarfile.TarInfo(name);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    cfg=dict(project=PROJECT,release=RELEASE,run=RUN,archive=archive.name,sha256=sha,commit=commit)
    (output/'package.json').write_text(json.dumps(dict(cfg,files=len(selected)),indent=2),encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    publish(a.output)
