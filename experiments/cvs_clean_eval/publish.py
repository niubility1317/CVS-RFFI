"""Immutable committed baseline-evaluation release and independent readback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_clean_eval.prepare import RUN,RELEASE,PROJECT
from experiments.cvs_residual_identity.publish import ssh,CONNECTION
ROOT=Path(__file__).resolve().parents[2]

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
p=Path(c['project']);release=p/'releases'/c['release'];run=p/'runs'/c['run'];logs=p/'logs'/c['run']
if any(q.exists() for q in (release,run,logs)):raise FileExistsError('Reconcile existing delivery;no resubmit')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for m in tar.getmembers():
        if not (release/m.name).resolve().is_relative_to(release.resolve()) or not m.isfile():raise ValueError('Unsafe member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
for module in ('predict','score','dispatch'):
    subprocess.run([python,'-m','experiments.cvs_clean_eval.'+module,'--help'],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
spec=release/c['spec_ref']
d=json.loads(spec.read_text())
if d['run_id']!=c['run'] or len(d['rows'])!=c['matrix_rows']:raise ValueError('Unexpected clean matrix')
for row in d['rows']:
    if row.get('reuse_from_run'):continue
    cfg=json.loads(Path(row['config']).read_text())
    for key in ('source_contract',):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    for name in ('completion.json','initialization.json','source_contract.json','resolved_config.json','last.pt'):
        if not (Path(cfg['source_output'])/name).is_file():raise FileNotFoundError(name)
    for name in ('manifest.json','index.npz','clean.npy'):
        if not (Path(cfg['p1_capsule'])/name).is_file():raise FileNotFoundError(name)
smoke='import torch;from experiments.cvs_clean_eval.contracts import BASELINES,build_model;torch.set_num_threads(2);models=[build_model(v).eval() for v in BASELINES];scores=[m(torch.zeros(2,2,256)) for m in models];assert all(s.shape==(2,6) and torch.isfinite(s).all() for s in scores)'
subprocess.run([python,'-c',smoke],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
preflight='import torch;from pathlib import Path;from experiments.cvs_clean_eval.contracts import read,frozen_selection,validate_predict_config,checkpoint_contract;from experiments.cvs_clean_eval.dispatch import validate_spec;d=validate_spec(read('+repr(str(spec))+'));selection=frozen_selection(d["selection_file"]);'+ '\nfor row in d["rows"]:\n if row.get("reuse_from_run"):continue\n c=validate_predict_config(read(row["config"]),selection);q=Path(c["source_output"]);checkpoint_contract(c,read(q/"completion.json"),read(q/"initialization.json"),read(q/"source_contract.json"),read(c["source_contract"]),read(q/"resolved_config.json"),torch.load(q/"last.pt",map_location="cpu",weights_only=False))'
subprocess.run([python,'-c',preflight],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
command=[python,'-u','-m','experiments.cvs_clean_eval.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

INSPECT=r'''
import json,subprocess,time
from pathlib import Path
p=Path(PROJECT);run=p/'runs'/RUN;release=p/'releases'/RELEASE
def read(q):return json.loads(q.read_text()) if q.exists() else None
def proc(pid):
    q=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((q/'cwd').resolve()),argv=(q/'cmdline').read_bytes().decode().split('\0')) if q.exists() else None
    except OSError:return dict(pid=pid,state='UNKNOWN')
state=read(run/'pipeline_state.json')
result=dict(read_at=time.time(),run_id=RUN,identity=dict(user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip()),
    submit=read(release/'submit.json'),pipeline=state,dispatcher_process=proc(state['pid']) if state else None,rows=[],
    marker=read(run/'scoring_clean_complete.json'),summary=read(run/'clean_summary.json'),results=read(run/'clean_scored_results.json'),
    dispatcher_log=(release/'dispatcher.stdout.log').read_text(errors='replace')[-6000:] if (release/'dispatcher.stdout.log').exists() else '')
for rid,row in (state or {}).get('rows',{}).items():
    out=Path(row['output_root']) if row.get('output_root') else None
    log=Path(row['log']) if row.get('log') else None
    result['rows'].append(dict(row_id=rid,status=row['status'],gpu=row.get('gpu'),process=proc(row.get('pid')),
        resolved=read(out/'resolved_config.json') if out else None,provenance=read(out/'provenance.json') if out else None,
        completion=read(out/'clean_complete.json') if out else None,log_tail=log.read_text(errors='replace').splitlines()[-4:] if log and log.exists() else []))
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(result))
'''


def inspect(output,run=RUN,release=RELEASE):
    d=json.loads(ssh(INSPECT.replace('PROJECT',repr(PROJECT)).replace('RELEASE',repr(release)).replace('RUN',repr(run))))
    output.mkdir(parents=True,exist_ok=True)
    (output/'readback.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(run_id=run,status=d['pipeline']['status'] if d['pipeline'] else None,marker=d['marker'],
        rows=[dict(row_id=r['row_id'],status=r['status'],gpu=r['gpu'],alive=bool(r['process']),completion=bool(r['completion']),error=r['log_tail'] if r['status']=='FAILED' else None) for r in d['rows']]),ensure_ascii=False))


def publish(output,run=RUN,release=RELEASE,spec_ref='experiments/cvs_clean_eval/configs/launch_spec.json',matrix_rows=16):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True,timeout=30).split()[0]
    if remote!=commit:raise ValueError('Remote branch differs from HEAD')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    prefixes=('experiments/cvs_coordinate_clean/','experiments/cvs_coordinate_identity/','experiments/cvs_synchronized_clean/','experiments/cvs_synchronized_identity/','experiments/cvs_equivariant_clean/','experiments/cvs_equivariant_identity/','experiments/cvs_reference_clean/','experiments/cvs_reference_identity/','experiments/cvs_rff_physics/','experiments/cvs_gauge_clean/','experiments/cvs_gauge_identity/','experiments/cvs_observable_clean/','experiments/cvs_observable_identity/','experiments/cvs_rf_operator_clean/','experiments/cvs_rf_operator_identity/','experiments/cvs_simplex_clean/','experiments/cvs_simplex_identity/','experiments/cvs_coherence_clean/','experiments/cvs_coherence_identity/','experiments/cvs_stability_clean/','experiments/cvs_stability_identity/','experiments/cvs_attentive_clean/','experiments/cvs_attentive_identity/','experiments/cvs_interaction_clean/','experiments/cvs_interaction_identity/','experiments/cvs_balanced_clean/','experiments/cvs_balanced_identity/','experiments/cvs_selected_clean/','experiments/cvs_clean_eval/','experiments/cvs_residual_identity/','experiments/cvs_clean_design/','experiments/cvs_identity_ce/',
        'experiments/adv3b02_xuc/code/','baselines/cvcnn_ce/','baselines/common/','code/leo_practical/')
    selected=[n for n in names if (n.startswith(prefixes) or n in {'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'}) and Path(n).suffix in {'.py','.json'}]
    # Directory pathspecs bound the command line on Windows even as the package grows.
    status_paths=[*prefixes,'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py']
    if subprocess.check_output(['git','status','--porcelain','--',*status_paths],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release')
    output.mkdir(parents=True,exist_ok=True);archive=output/(release+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing package;reconcile before retry')
    with tarfile.open(archive,'w:gz') as tar:
        for n in selected:
            blob=subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT);item=tarfile.TarInfo(n);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    c=dict(project=PROJECT,release=release,run=run,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit,spec_ref=spec_ref,matrix_rows=matrix_rows)
    (output/'package.json').write_text(json.dumps(dict(c,files=len(selected)),indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output)
