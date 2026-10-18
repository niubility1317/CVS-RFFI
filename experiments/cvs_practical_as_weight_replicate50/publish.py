"""One ordinary-account E50 paired-budget training release;private runtime stays outside Git."""
import hashlib,io,json,subprocess,tarfile
from . import design as d

def remote(script,runtime):
 args=['ssh','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -'];r=subprocess.run(args,input=script.encode('utf8'),capture_output=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 package='experiments/cvs_practical_as_weight_replicate50';local=d.ROOT/'local_artifacts/F_As_weight_E50_s0203_20261018';runtime=d.read(local/'runtime.json')
 if runtime['run_id']!=d.RUN or runtime['originalreplication_commit']!=d.PARENT_COMMIT or runtime['weighted50_commit']!=d.WEIGHTED50_COMMIT or runtime['weighted30_commit']!=d.WEIGHTED30_COMMIT:raise ValueError('Private runtime for another run/reference')
 if subprocess.check_output(['git','status','--porcelain','--',package],cwd=d.ROOT,text=True).strip():raise ValueError('Commit actual diagnostic before release')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();project=runtime['project'];release=project+'/releases/'+d.RELEASE
 c=dict(project=project,release=release,archive=release+'.tar.gz',run=project+'/runs/'+d.RUN,python=runtime['python'],user=runtime['user'],host=runtime['host'],commit=commit,runtime={k:runtime[k] for k in ('originalreplication_release','originalreplication_commit','weighted50_release','weighted50_commit','weighted30_release','weighted30_commit')},eligibility=project+'/runs/'+d.SEED01_WEIGHT50_RUN+'/analysis_auto_r01/source_summary.json',eligibility_run=d.SEED01_WEIGHT50_RUN)
 script=r'''
import importlib.util,json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Existing immutable release/run;reconcile,no resubmission')
if shutil.disk_usage(c['project']).free<10*1024**3:raise ValueError('Insufficient output space')
rt=c['runtime']
for label,commit in [('originalreplication','63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47'),('weighted50','64176e7ff1aa5cd2477a504f8b442bb7966079dc'),('weighted30','3befd47c8041d81f9430eda53562d0821624ff04')]:
 release=Path(rt[label+'_release'])
 if rt[label+'_commit']!=commit or (release/'release_commit.txt').read_text().strip()!=commit:raise ValueError('Actual conditional algorithm/originalseed runtime differs')
summary=json.loads(Path(c['eligibility']).read_text());decision=summary['decision']
if summary['run_id']!=c['eligibility_run'] or summary['target_scores_read'] is not False or any(summary[k]!=0 for k in ('NN','IQ_reads','checkpoint_reads','updates')) or any(decision[k] is not True for k in ('source_criteria_pass','fixed_02_03_source_eligible','clean_nonnegative','mid_low_gain_at_least_0p5pp')) or decision['newAs50_minus_ownAs30_fullV_clean_pp']<0. or decision['newAs50_minus_ownAs30_mid_low_mean_pp']<.5:raise ValueError('Fixed seed01 source-only conjunction has not passed;do not release')
parent=Path(rt['originalreplication_release']);spec=importlib.util.spec_from_file_location('ASWEIGHT_REPLICA_PREFLIGHT',parent/'experiments/cvs_As_replicate49/runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.bootstrap()
from experiments.cvs_As_replicate49 import design as original
roles=original.read(original.SOURCE)['role_ids']
for row in original.rows():
 cfg=original.config(row,20);out=Path(cfg['output_root']);done=original.read(out/'completion.json');origin=original.read(out/'initialization.json');native=original.read(out/'actual_native_args.json')
 if done['config']!=cfg or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or not origin['scratch_only'] or origin['checkpoint_sources'] or origin['ancestors'] or origin['target_contact'] or original.read(out/'source_contract.json')['role_ids']!=roles or native['seed']!=row['model_seed'] or not (out/'final_ssdg.pth').is_file():raise ValueError('Actual own02/03 scratch20/seed/roles/checkpoint differs')
from experiments.cvs_multi_state_action.dispatch import capacity
resources={str(g):dict(total_GPU_processes=len(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()}
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,own_original20_true_seed_metadata='EXACT_MATCH',source_roles='EXACT_MATCH',CPU_threads=2,GPU='dispatcher CPU;worker allocation in capacity queue',GPU_capacity=resources,checkpoint_or_target_access=False)))
'''
 d.write(local/'preflight.json',remote(script.replace('CONFIG',repr(c)),runtime))
 raw=subprocess.check_output(['git','archive',commit,package],cwd=d.ROOT);record=subprocess.check_output(['git','show',commit+':automation_reports/CV-SincNet/'+d.RUN+'/experiment.json'],cwd=d.ROOT);archive=local/(d.RELEASE+'.tar.gz')
 runtime_bytes=json.dumps(dict(run_id=d.RUN,project=project,**c['runtime']),indent=2).encode()
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
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_practical_as_weight_replicate50')],cwd=release,env=env,check=True)
check=[c['python'],'-u',str(release/'experiments/cvs_practical_as_weight_replicate50/runner.py'),'--checks']
with (release/'checks.stdout.log').open('x') as stream:subprocess.run(check,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
command=[c['python'],'-u',str(release/'experiments/cvs_practical_as_weight_replicate50/runner.py')];log=release/'practical_AsWeight50_s0203.stdout.log'
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
