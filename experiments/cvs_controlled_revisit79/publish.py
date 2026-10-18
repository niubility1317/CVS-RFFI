"""Minimal ordinary-account source-only P79 release; no truth or training stage."""
import argparse,hashlib,io,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_current_controls_E30_20261019_r01'
PACKAGE='experiments/cvs_controlled_revisit79'
HELPERS=['experiments/cvs_fullsource_residual_diag24/'+f for f in ['__init__.py','design.py','engine.py','model.py','observe.py']]
ANALYSIS='analysis/cvs_current_controls79_analysis_20261019.py'
def remote(script,runtime):
 r=subprocess.run([runtime['ssh'],'-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -'],input=script.encode('utf8'),capture_output=True,timeout=55)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 a=argparse.ArgumentParser();a.add_argument('--settings',required=True);args=a.parse_args();p=Path(args.settings);runtime=json.loads(p.read_text(encoding='utf8'));out=p.parent
 if runtime['run_id']!=RUN:raise ValueError('Wrong diagnostic settings')
 intended=[PACKAGE,*HELPERS,ANALYSIS]
 if subprocess.check_output([runtime['git'],'status','--porcelain','--',*intended],cwd=ROOT,text=True).strip():raise ValueError('Commit actual code and config first')
 commit=subprocess.check_output([runtime['git'],'rev-parse','HEAD'],cwd=ROOT,text=True).strip();project=runtime['project'];release=project+'/releases/'+RELEASE
 c=dict(project=project,release=release,archive=release+'.zip',run=project+'/runs/'+RUN,python=runtime['python'],user=runtime['user'],host=runtime['host'],commit=commit,reference=runtime['reference_release'],reference_commit='19aedc3604fed9c193237dd92366164178af0983')
 script=r'''
import importlib.util,json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Immutable output exists;reconcile,no relaunch')
if shutil.disk_usage(c['project']).free<2*1024**3:raise ValueError('Insufficient output space')
r=Path(c['reference'])
if (r/'release_commit.txt').read_text().strip()!=c['reference_commit']:raise ValueError('Fixed P43 runtime differs')
spec=importlib.util.spec_from_file_location('P79_PARENT_METADATA',r/'experiments/cvs_noise_shape_factorial43/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
from experiments.cvs_noise_shape_factorial43 import design as d
roles=d.read(d.SOURCE)['role_ids']
for arm in ['B00','B01']:
 for epoch in [20,30]:
  cfg=d.config(next(x for x in d.rows() if x['arm']==arm),epoch);p=Path(cfg['output_root']);done=d.read(p/'completion.json');init=d.read(p/'initialization.json');f=d.read(p/'source_frozen.json')
  if done['config']!=cfg or done['epoch']!=(20 if epoch==20 else 10) or done['optimizer_steps']!=(4440 if epoch==20 else 2220) or done['target_access'] or init['target_contact'] or init['checkpoint_sources']!=cfg['checkpoint_sources'] or d.read(p/'source_contract.json')['role_ids']!=roles or f['selection']!='final_only' or f['target_access']:raise ValueError('Source lineage/config/physical roles differ')
  if epoch==20 and (not init['scratch_only'] or init['ancestors']):raise ValueError('OwnE20 scratch origin required')
  if epoch==30 and (init['scratch_only'] or init['ancestors']!=[cfg['parent_source']]):raise ValueError('OwnE20+10 lineage required')
panel=Path(d.PANEL);done=d.read(panel/'observations/completion.json')
if done['status']!='SOURCE_OBSERVATIONS_COMPLETE' or done['unique_V']!=1800 or done['draws']!=2 or done['target_access'] or done['training_updates']:raise ValueError('Original source-only cache missing')
from experiments.cvs_multi_state_action.dispatch import capacity
caps=capacity();eligible=[g for g,x in caps.items() if len(x['pids'])<2 and x['free_mb']>=4000]
if not eligible:raise RuntimeError('No allowed GPU capacity; do not relaunch')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,source_lineage='EXACT_MATCH',source_role_counts={k:len(v) for k,v in roles.items()},cached_observations=True,checkpoint_or_IQ_reads=0,target_access=False,capacity={str(g):dict(total_processes=len(x['pids']),free_mb=x['free_mb']) for g,x in caps.items()})))
'''
 (out/'landing_preflight.json').write_text(json.dumps(remote(script.replace('CONFIG',repr(c)),runtime),indent=2)+'\n',encoding='utf8')
 raw=subprocess.check_output([runtime['git'],'archive','--format=zip',commit,*intended],cwd=ROOT);archive=out/(RELEASE+'.zip')
 with zipfile.ZipFile(io.BytesIO(raw)) as src,zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as dst:
  for member in src.infolist():
   if not member.is_dir():dst.writestr(member.filename,src.read(member.filename))
  dst.writestr('release_commit.txt',commit+'\n')
  dst.writestr('registration/experiment.json',subprocess.check_output([runtime['git'],'show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json'],cwd=ROOT))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
 subprocess.run([runtime['scp'],'-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=55)
 script=r'''
import importlib.util,json,hashlib,os,py_compile,subprocess,time,zipfile
from pathlib import Path
c=CONFIG;root=Path(c['release']);archive=Path(c['archive'])
if root.exists() or Path(c['run']).exists():raise FileExistsError('Possible prior mutation;reconcile')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
root.mkdir()
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():
  if not (root/n).resolve().is_relative_to(root.resolve()):raise ValueError('Archive path escape')
 z.extractall(root)
files=list(root.rglob('*.py'))
for f in files:py_compile.compile(str(f),doraise=True)
r=Path(c['reference']);s=importlib.util.spec_from_file_location('P79_CAPACITY',r/'experiments/cvs_noise_shape_factorial43/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock,process
with lock(Path(c['project'])/'runs/receiver_residual_capacity4.lock'):
 caps=capacity();eligible=[g for g,x in caps.items() if len(x['pids'])<2 and x['free_mb']>=4000]
 if not eligible:raise RuntimeError('No capacity; landed but not launched, reconcile')
 gpu=min(eligible,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 cmd=[c['python'],'-u',str(root/'experiments/cvs_controlled_revisit79/runner.py'),'--project',c['project'],'--device','cuda:0'];log=root/'source_controls79.stdout.log'
 with log.open('x') as f:child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 actual=process(child.pid)
 if actual is None:raise RuntimeError('Launched process missing;reconcile,no retry')
 result=dict(status='OBSERVED',pid=child.pid,cwd=str(root),argv=cmd,GPU=gpu,actual=actual,log=str(log),archive_transfer='VERIFIED',remote_compile='PASS',compiled_files=len(files),commit=c['commit'],launch_time=time.time(),source_only=True,training_updates=0,target_access=False)
 (root/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
'''
 result=remote(script.replace('CONFIG',repr(c)),runtime);(out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:result[k] for k in ['status','pid','GPU','archive_transfer','remote_compile','compiled_files','commit']}))
if __name__=='__main__':main()
