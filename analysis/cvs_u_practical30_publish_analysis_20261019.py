"""Immutable CPU-only aggregate analysis release for UPractical E30.

This publishes analysis code only. It does not load IQ/checkpoints or alter training.
Source analysis exits before formal analysis; final analysis requires an explicit launch.
An ambiguous transport outcome must be reconciled before another invocation.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

RUN = '20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02'
TRAINING_RELEASE = 'cvs_FcrossRX_UPractical_E30_20261019_r02'
RELEASE = 'cvs_F_u_practical30_analysis_20261019_r02'
OUTPUT = 'analysis_auto_r01'
FILES = (
    'cvs_u_practical30_analysis_20261019.py',
    'cvs_u_practical30_publish_analysis_20261019.py',
    'cvs_u_practical30_collect_20261019.py',
    'cvs_practical_budget50_analysis_20261018.py',
    'cvs_satellite_dropout_results_20261018.py',
    'cvs_satellite_dropout_logs_20261018.py',
    'cvs_u_practical30_check_20261019.py',
)


def validate_names(release, output):
    if re.fullmatch(r'cvs_F_u_practical30_analysis_20261019_r[0-9]{2}', release) is None:
        raise ValueError('Expected own immutable analysis release basename')
    if re.fullmatch(r'analysis_auto_r[0-9]{2}', output) is None:
        raise ValueError('Expected own immutable analysis output basename')


def analysis_config(runtime, commit, release=RELEASE, output=OUTPUT, phase='source'):
    validate_names(release, output)
    if phase not in ('source', 'final'):
        raise ValueError('Expected explicit source or final analysis phase')
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('Actual Git commit required')
    project = str(PurePosixPath(runtime['project']))
    result = dict(project=project,
                  training_release=project + '/releases/' + TRAINING_RELEASE,
                  release=project + '/releases/' + release,
                  archive=project + '/releases/' + release + '.tar.gz',
                  output=project + '/runs/' + RUN + '/' + output,
                  python=runtime['python'], user=runtime['user'], host=runtime['host'],
                  commit=commit, phase=phase)
    if phase == 'final':
        result['truth'] = project + '/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    result['launch'] = result['output'] + '-' + phase + '-launch'
    return result


def remote(script, runtime):
    result = subprocess.run(
        ['ssh', '-F', runtime['ssh_config'], '-o', 'BatchMode=yes', '-o',
         'ConnectTimeout=10', '-T', 'N607', runtime['python'] + ' -'],
        input=script.encode('utf8'), capture_output=True, timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', errors='replace'))
    return json.loads(result.stdout)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


PREFLIGHT = r'''
import json,os,subprocess
from pathlib import Path
c=CONFIG
if c['phase']!='source': raise ValueError('Transfer is source phase only; final reuses the immutable analysis release')
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:
    raise ValueError('Expected configured ordinary N607 account')
if any(Path(c[k]).exists() for k in ('release','archive','output','launch')):
    raise FileExistsError('Existing analysis artifact; reconcile, never resubmit')
if not (Path(c['project'])/'runs/RUN_VALUE/dispatcher.json').is_file():
    raise ValueError('Registered real training run missing')
if not Path(c['training_release']).is_dir():
    raise ValueError('Original training release missing')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,training_processes_untouched=True,target_truth_read=False,IQ=0,CP=0,NN=0)))
'''

TRANSFER = r'''
import hashlib,json,os,re,subprocess,tarfile
from pathlib import Path
c=CONFIG; release=Path(c['release'])
if c['phase']!='source': raise ValueError('Transfer is source phase only; final reuses the immutable analysis release')
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:
    raise ValueError('Expected configured ordinary N607 account')
if release.exists() or Path(c['output']).exists() or Path(c['launch']).exists():
    raise FileExistsError('Existing analysis mutation; reconcile')
if hashlib.sha256(Path(c['archive']).read_bytes()).hexdigest()!=c['sha256']:
    raise ValueError('Single release archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive']) as source:
    for member in source.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):
            raise ValueError('Invalid archive member')
    source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'); cwd=release/'analysis'
subprocess.run([c['python'],'-m','compileall','-q',str(cwd)],cwd=cwd,env=env,check=True)
result=dict(status='TRANSFER_VERIFIED',commit=c['commit'],archive_SHA256=c['sha256'],remote_compile='PASS',remote_unit_tests='NOT_REPEATED_LOCAL_FOCUSED_CHECKS',IQ=0,CP=0,NN=0,updates=0,target_truth_read=False)
(release/'transfer.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
'''

LAUNCH = r'''
import json,os,subprocess
from pathlib import Path
c=CONFIG; release=Path(c['release']); out=Path(c['output']); launch=Path(c['launch']); cwd=release/'analysis'
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:
    raise ValueError('Expected configured ordinary N607 account')
if launch.exists(): raise FileExistsError('Existing phase launch owner; reconcile, no duplicate')
if c['phase']=='source':
    if out.exists(): raise FileExistsError('Existing analysis output; source cannot overwrite saved artifacts')
    if 'truth' in c: raise ValueError('Source phase must not receive a truth path')
elif c['phase']=='final':
    if not (out/'source_complete.json').is_file(): raise ValueError('Complete saved source analysis required before final launch')
    final_names=('formal_summary.json','formal_confusions.json','formal_metrics.csv','final_report.md','costs.json','complete.json','failure.json','source_failure.json','final_failure.json')
    if any((out/name).exists() for name in final_names): raise FileExistsError('Existing formal/failure artifact; reconcile without overwriting')
else:
    raise ValueError('Expected explicit source or final analysis phase')
if (release/'release_commit.txt').read_text().strip()!=c['commit'] or json.loads((release/'transfer.json').read_text())['status']!='TRANSFER_VERIFIED':
    raise ValueError('Analysis release identity differs')
launch.mkdir(); env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
cmd=[c['python'],'-u',str(cwd/'cvs_u_practical30_analysis_20261019.py'),'--project',c['project'],'--release',c['training_release'],'--output',str(out),'--phase',c['phase'],'--code-commit',c['commit']]
if c['phase']=='final': cmd.extend(['--truth',c['truth']])
with (launch/'stdout.log').open('x') as stdout,(launch/'stderr.log').open('x') as stderr:
    child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
result=dict(status='SUBMITTED',phase=c['phase'],pid=child.pid,cwd=str(cwd),argv=cmd,analysis_code_commit=c['commit'],training_processes_untouched=True,IQ=0,CP=0,NN=0,updates=0)
(launch/'launch.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('transfer', 'launch'), required=True)
    parser.add_argument('--phase', choices=('source', 'final'), default='source')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--analysis-release', default=RELEASE)
    parser.add_argument('--output-name', default=OUTPUT)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text(encoding='utf-8-sig'))
    config = analysis_config(runtime, args.commit, args.analysis_release, args.output_name, args.phase)
    if args.stage == 'transfer' and args.phase != 'source':
        parser.error('Transfer uses --phase source; final launch reuses the immutable release')
    local = args.artifact_root
    local.mkdir(parents=True, exist_ok=True)
    if args.stage == 'transfer':
        preflight = remote(PREFLIGHT.replace('RUN_VALUE', RUN).replace('CONFIG', repr(config)), runtime)
        write_json(local / 'preflight.json', preflight)
        paths = ['analysis/' + name for name in FILES]
        if subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=args.source_root, text=True).strip():
            raise ValueError('Commit actual analysis code before release')
        raw = subprocess.check_output(['git', 'archive', args.commit, *paths], cwd=args.source_root)
        archive = local / (args.analysis_release + '.tar.gz')
        with archive.open('xb') as stream, tarfile.open(fileobj=io.BytesIO(raw)) as source, tarfile.open(fileobj=stream, mode='w:gz') as target:
            for member in source.getmembers():
                if member.isfile(): target.addfile(member, source.extractfile(member))
            payload = (args.commit + '\n').encode('utf8')
            info = tarfile.TarInfo('release_commit.txt'); info.size = len(payload)
            target.addfile(info, io.BytesIO(payload))
        config['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        subprocess.run(['scp', '-F', runtime['ssh_config'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', str(archive), 'N607:' + config['archive']], capture_output=True, check=True, timeout=55)
        result = remote(TRANSFER.replace('CONFIG', repr(config)), runtime)
        write_json(local / 'transfer.json', result)
        print(json.dumps(result))
    else:
        result = remote(LAUNCH.replace('CONFIG', repr(config)), runtime)
        write_json(local / (args.phase + '_launch.json'), result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('cwd', 'argv')}))


if __name__ == '__main__':
    main()
