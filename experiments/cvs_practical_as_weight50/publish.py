"""One ordinary-account E50 paired-budget training release;private runtime stays outside Git."""
import hashlib,io,json,subprocess,tarfile
from . import design as d

def remote(script,runtime):
 args=['ssh','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -'];r=subprocess.run(args,input=script.encode('utf8'),capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 package='experiments/cvs_practical_as_weight50';local=d.ROOT/'local_artifacts/F_As_weight_E50_20261018';runtime=d.read(local/'runtime.json')
 if runtime['run_id']!=d.RUN or runtime['reference_commit']!=d.REFERENCE_COMMIT or runtime['parent_commit']!=d.PARENT_COMMIT:raise ValueError('Private runtime for another run/reference')
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual diagnostic before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();project=runtime['project'];release=project+'/releases/'+d.RELEASE
 c=dict(project=project,release=release,archive=release+'.tar.gz',run=project+'/'+'runs/'+d.RUN,config=[d.config(r) for r in d.rows()],python=runtime['python'],user=runtime['user'],host=runtime['host'],commit=commit,reference_release=runtime['reference_release'],reference_commit=runtime['reference_commit'],parent_release=runtime['parent_release'],parent_commit=runtime['parent_commit'])
 script=r'''
import importlib.util,json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Existing immutable release/run;reconcile,no resubmission')
for alias in [c['config'][0]['source_contract'],'PROJECT/runs/'+"20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01"+'/B00-s2026092701/source/completion.json','PROJECT/runs/'+"20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01"+'/B01-s2026092701/source/completion.json']:
 p=Path(c['project'])/alias[len('PROJECT/'):] if alias.startswith('PROJECT/') else Path(alias)
 if not p.is_file():raise FileNotFoundError(p)
if shutil.disk_usage(c['project']).free<10*1024**3:raise ValueError('Insufficient output space')
parent=Path(c['parent_release'])
if (parent/'release_commit.txt').read_text().strip()!=c['parent_commit']:raise ValueError('Pinned actual weighted30 release differs')
spec=importlib.util.spec_from_file_location('ASWEIGHT50_PREFLIGHT_PARENT',parent/'experiments/cvs_practical_as_weight30/runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
from experiments.cvs_practical_as_weight30 import design as weighted
from experiments.cvs_noise_shape_factorial43 import design as original
roles=original.read(original.SOURCE)['role_ids']
for row in weighted.rows():
 cfg=weighted.config(row);out=Path(cfg['output_root']);done=weighted.read(out/'completion.json');init=weighted.read(out/'initialization.json');frozen=weighted.read(out/'source_frozen.json');contract=weighted.read(out/'source_contract.json');result=weighted.read(out/'result.json')
 if done['config']!=cfg or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or done.get('target_evaluated',False) or init['target_contact'] or init['checkpoint_sources']!=cfg['checkpoint_sources'] or init['scratch_only'] or init['ancestors']!=[cfg['parent_source']] or contract['role_ids']!=roles or frozen['config']!=cfg or frozen['selection']!='final_only' or frozen['target_access'] or result['cumulative_epochs']!=30 or result['cumulative_optimizer_steps']!=6660:raise ValueError('Actual own weightedFINAL30 frozen/source/cumulative handles differ')
 original20=original.config(next(r for r in original.rows() if r['row_id']==row['row_id']),20);ancestor=Path(original20['output_root']);meta=original.read(ancestor/'completion.json');origin=original.read(ancestor/'initialization.json')
 if meta['config']!=original20 or meta['epoch']!=20 or meta['optimizer_steps']!=4440 or meta['target_access'] or not origin['scratch_only'] or origin['checkpoint_sources'] or origin['ancestors'] or origin['target_contact'] or original.read(ancestor/'source_contract.json')['role_ids']!=roles:raise ValueError('Original own20 ancestor metadata differs')
from experiments.cvs_multi_state_action.dispatch import capacity
resources={str(g):dict(total_GPU_processes=len(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()}
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,own_weighted30_and_original20_metadata='EXACT_MATCH',source_roles='EXACT_MATCH',CPU_threads=2,GPU='dispatcher CPU;worker allocation in capacity queue',GPU_capacity=resources,checkpoint_or_target_access=False)))
'''
 d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c)),runtime))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT);archive=local/(d.RELEASE+'.tar.gz')
 runtime_bytes=json.dumps(dict(run_id=d.RUN,project=project,reference_release=runtime['reference_release'],reference_commit=runtime['reference_commit'],parent_release=runtime['parent_release'],parent_commit=runtime['parent_commit']),indent=2).encode()
 with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
  for member in source.getmembers():
   if member.isfile():target.addfile(member,source.extractfile(member))
  for name,data in [('release_commit.txt',(commit+chr(10)).encode()),('registration/experiment.json',record),('runtime.json',runtime_bytes)]:
   item=tarfile.TarInfo(name);item.size=len(data);target.addfile(item,io.BytesIO(data))
 c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=60)
 script=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);run=Path(c['run']);archive=Path(c['archive'])
if release.exists() or run.exists():raise FileExistsError('Existing mutation;reconcile before relaunch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_practical_as_weight50')],cwd=release,env=env,check=True)
check=[c['python'],'-u',str(release/'experiments/cvs_practical_as_weight50/runner.py'),'--checks']
with (release/'checks.stdout.log').open('x') as stream:subprocess.run(check,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_practical_as_weight50/runner.py')];log=release/'practical_AsWeight50.stdout.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=c['commit'],archive_transfer='VERIFIED',remote_compile='VERIFIED',CPU_threads=2,GPU='none')
(release/'submit.json').write_text(json.dumps(result,indent=2)+chr(10));proc=Path('/proc')/str(child.pid);deadline=time.monotonic()+2
while time.monotonic()<deadline and child.poll() is None:
 try:
  actual=dict(cwd=os.readlink(proc/'cwd'),argv=(proc/'cmdline').read_bytes().decode().split(chr(0))[:-1],cuda=[v for v in (proc/'environ').read_bytes().decode().split(chr(0)) if v.startswith('CUDA_VISIBLE_DEVICES=')])
  if actual['cwd']==str(release) and actual['argv']==command and actual['cuda']==['CUDA_VISIBLE_DEVICES=']:result.update(status='OBSERVED',actual=actual);break
 except FileNotFoundError:pass
 time.sleep(.05)
(release/'submit.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result))
'''
 result=remote(script.replace('CONFIG',repr(c)),runtime);d.write(local/'submit.json',result);print(json.dumps({k:result[k] for k in ('status','pid','commit','archive_transfer','remote_compile','GPU')}))
if __name__=='__main__':main()
