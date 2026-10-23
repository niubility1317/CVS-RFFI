"""Release committed adapters once; never replace an existing run/output root."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .plan import RUN, SEEDS, METHODS, GPU_BY_SEED, configuration


def ssh(script, connection):
    compile(script, '<remote>', 'exec')
    result = subprocess.run(['ssh', *connection, '-T', 'N607', 'python3 -'],
                            input=script.encode('utf-8'), capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
    return json.loads(result.stdout)


def prepare(context, project, release, commit):
    root = project + '/runs/' + RUN
    blobs, rows, baselines = {}, [], {}
    for seed in SEEDS:
        original = context[str(seed)]
        baselines[str(seed)] = {arm: original[arm]['score']['output_root'] for arm in ('F20', 'F21')}
        for method in METHODS:
            c = configuration(original['F21']['train'], method, seed, project, commit)
            row_id = c['row_id']
            cp = deepcopy_without(original['F21']['predict'], ('original_arm', 'raw_predictions_from', 'strict_supervised_system'))
            cp.update(run_id=RUN, row_id=row_id, source_run_id=RUN, source_row_id=row_id,
                model_seed=seed, source_output=c['output_root'], readouts={'raw': {}},
                source_physics_diagnostics=False, output_root=root + '/' + row_id + '/predictions', code_commit=commit)
            for task in cp['tests'].values():
                task.pop('truth', None)
            from experiments.cric_dura_comparison.evaluation import reject_truth
            reject_truth(cp)
            cs = dict(original['F21']['score'], run_id=RUN, row_id=row_id, model_seed=seed,
                prediction_output=cp['output_root'], output_root=root + '/' + row_id + '/scores', readouts=['raw'])
            paths = {k: release + '/configs/' + row_id + '-' + k + '.json' for k in ('train', 'predict', 'score', 'row')}
            row = dict(row_id=row_id, method=method, model_seed=seed, gpu=GPU_BY_SEED[seed],
                train_config=paths['train'], prediction_config=paths['predict'], score_config=paths['score'], row_config=paths['row'],
                baseline_source_configs={arm: original[arm]['train'] for arm in ('F20', 'F21')})
            rows.append(row)
            for name, value in (('train', c), ('predict', cp), ('score', cs), ('row', row)):
                blobs['configs/' + row_id + '-' + name + '.json'] = value
    rt = dict(run_id=RUN, project=project, release_root=release, run_root=root,
        runtime_file=release + '/runtime.json', code_commit=commit,
        launch_owner='codex-pamf-ssl-research-20261023', rows=rows,
        frozen_sources=context[str(SEEDS[0])]['F21']['train']['frozen_sources'],
        baseline_source=context[str(SEEDS[0])]['F21']['train']['output_root'],
        baseline_scores=baselines, allowed_gpus=[0, 1, 2], max_total_workers=6, max_per_gpu=2,
        python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', independent_CPU_scorers=1,
        stop_rule='Two identical technical failure fingerprints stop new dispatch; healthy jobs and all outputs preserved. No performance-based stop/reorder/rerun.')
    blobs['runtime.json'] = rt
    return rt, blobs


def deepcopy_without(value, keys):
    from copy import deepcopy
    result = deepcopy(value)
    for key in keys:
        result.pop(key, None)
    return result


REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
rt=c['runtime'];release=Path(rt['release_root'])
if Path(rt['run_root']).exists() or release.exists():raise FileExistsError('Reconcile existing submission before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Release transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.pamf_ssl_research.control','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as stream:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo, context_path, output, ssh_config, *, prepare_run=prepare, run_id=RUN):
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()[0] != commit:
        raise ValueError('Remote branch OID differs from HEAD')
    context = json.loads(Path(context_path).read_text(encoding='utf-8'))
    source = context[str(SEEDS[0])]['F21']['train']['output_root']
    project = source.split('/runs/', 1)[0]
    release = project + '/releases/' + run_id
    rt, configurations = prepare_run(context, project, release, commit)
    connection = ['-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    preflight = ssh("from pathlib import Path\nimport json,subprocess\npaths=" + repr([release, rt['run_root']]) +
        "\nprint(json.dumps(dict(paths={p:Path(p).exists() for p in paths},user=subprocess.check_output(['whoami'],text=True).strip()," +
        "gpu=subprocess.check_output(['nvidia-smi','-i','0,1,2','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)," +
        "apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True))))\n", connection)
    (output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n', encoding='utf-8')
    if any(preflight['paths'].values()):
        raise FileExistsError('Existing remote output/release; reconcile before retry')
    if preflight['user'] != project.split('/')[2]:
        raise ValueError('Ordinary account mismatch at preflight')
    names = git('ls-files', 'experiments/pamf_ssl_research', 'experiments/cric_dura_comparison',
                'experiments/phase1_external_dg_aug/__init__.py', 'experiments/phase1_external_dg_aug/augmentation.py').decode().splitlines()
    names = [n for n in names if n.endswith('.py')]
    if git('status', '--porcelain', '--', *names).strip():
        raise ValueError('Uncommitted execution source')
    archive = output / (run_id + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name in names:
            blob = git('show', commit + ':' + name)
            item = tarfile.TarInfo(name); item.size = len(blob); item.mode = 0o644
            tar.addfile(item, io.BytesIO(blob))
        for name, data in configurations.items():
            blob = (json.dumps(data, indent=2) + '\n').encode('utf-8')
            item = tarfile.TarInfo(name); item.size = len(blob); item.mode = 0o644
            tar.addfile(item, io.BytesIO(blob))
    package = dict(runtime=rt, expected_user=project.split('/')[2], archive_remote=project + '/releases/' + archive.name,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output / 'runtime.json').write_text(json.dumps(rt, indent=2) + '\n', encoding='utf-8')
    (output / 'package.json').write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    subprocess.run(['scp', *connection, str(archive), 'N607:' + package['archive_remote']], check=True, timeout=60)
    receipt = ssh(REMOTE.replace('CONFIG', repr(package)), connection)
    (output / 'submit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'context', 'output', 'ssh-config'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    publish(a.repo, a.context, a.output, a.ssh_config)
