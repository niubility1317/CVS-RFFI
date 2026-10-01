"""Stage an exclusive independent Margin support analysis; never launch training."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess

from publish_d92_branch_support_probe import FLAGS
from publish_d92_margin_joint_probe import PATHS as SOURCE_PATHS
from run_d92_margin_joint_probe import validate_spec, budget_for_spec, STATUS, SCHEMA, METHOD

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_STATUS = 'COMPLETE_MARGIN_JOINT_PROBE_VERIFIED'
SUMMARY_SCHEMA = 'd92_margin_joint_support_summary_v1'
PATHS = list(dict.fromkeys(SOURCE_PATHS + [
    'tools/analyze_d92_margin_joint_probe.py', 'tools/summarize_d92_margin_joint_probe.py',
    'tools/summarize_d92_affine_joint_probe.py', 'tools/evaluate_d92_affine_joint_probe.py',
    'tools/run_d92_affine_joint_probe.py', 'tools/d92_affine_analysis_math.py',
    'tools/publish_d92_branch_support_probe.py',
    'docs/D92_MARGIN_JOINT_ANALYSIS_ENTRY_20261001.md']))


def commit_identity(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


def completion_check(done, spec):
    rows = len(spec['rows']); expected = budget_for_spec(spec)['total']['exact']
    if done.get('status') != STATUS or done.get('model_rows') != rows or done.get('completed_rows') != rows or done.get('run_id') != spec['run_id']:
        raise ValueError('All declared rows must be complete before independent analysis')
    if done.get('qp_resources') != spec['probe']['qp_resources'] or done.get('workload_complete') is not True:
        raise ValueError('QP resource binding or full workload completion mismatch')
    for key, value in expected.items():
        if done.get(key) != value: raise ValueError('Declared dynamic physical coverage incomplete: ' + key)


def summary_check(summary, spec):
    if (summary.get('status') != SUMMARY_STATUS or summary.get('summary_schema') != SUMMARY_SCHEMA
            or summary.get('schema') != SCHEMA or summary.get('method') != METHOD
            or summary.get('run_id') != spec['run_id'] or summary.get('model_rows') != len(spec['rows'])):
        raise ValueError('Independent summary schema/current run/rows mismatch')
    for key, value in budget_for_spec(spec)['total']['exact'].items():
        if summary['coverage'].get(key) != value: raise ValueError('Independent summary incomplete: ' + key)
    if summary['query_rows_used'] != 0 or summary['source_rows_used'] != 0 or summary['actual_A'] is not None:
        raise ValueError('Forbidden analysis access or invented ground A')
    if summary.get('qp_resources') != spec['probe']['qp_resources'] or summary.get('algorithm') != spec['probe']['algorithm']:
        raise ValueError('Independent summary frozen algorithm/QP resource mismatch')
    if not commit_identity(summary.get('release_commit')):
        raise ValueError('Independent summary actual runtime commit missing')


def execution_check(evidence,spec,cfg):
    summary_check(evidence['summary'],spec)
    execution=evidence['execution']
    if (execution.get('status')!='VERIFIED' or execution.get('commit')!=cfg['commit']
            or execution.get('run_id')!=spec['run_id'] or execution.get('release')!=cfg['release']
            or execution.get('output')!=cfg['output'] or execution.get('qp_resources')!=spec['probe']['qp_resources']
            or execution.get('runtime_commit')!=evidence['summary'].get('release_commit')
            or evidence.get('resolved_config')!=spec):
        raise ValueError('Independent remote execution/resolved configuration mismatch')


REMOTE = r'''
import json, os, subprocess, tarfile
from pathlib import Path
c=CONFIG
run=Path(c['run']);release=Path(c['release']);archive=Path(c['archive']);output=Path(c['output'])
done=json.loads((run/'complete.json').read_text(encoding='utf-8'))
state=json.loads((run/'state.json').read_text(encoding='utf-8'))
startup=json.loads((run/'startup.json').read_text(encoding='utf-8'))
if done.get('status')!=c['status'] or done.get('run_id')!=c['run_id'] or done.get('completed_rows')!=len(c['rows']) or done.get('model_rows')!=len(c['rows']):raise ValueError('Declared rows incomplete')
if set(state)!=set(c['rows']) or any(v.get('status')!=c['status'] for v in state.values()):raise ValueError('Row states incomplete')
if any(done.get(k)!=v for k,v in c['exact'].items()):raise ValueError('Declared physical coverage incomplete')
if done.get('qp_resources')!=c['qp_resources'] or done.get('workload_complete') is not True:raise ValueError('QP resources/full workload mismatch')
if startup.get('spec')!=c['resolved_spec'] or startup.get('commit')!=done.get('commit'):raise ValueError('Actual runtime resolved config mismatch')
if release.exists() or output.exists():raise FileExistsError('Preserve existing analysis evidence; reconcile before retry')
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not (release.parent/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):raise ValueError('Unsafe analysis archive')
 release.mkdir(exist_ok=False)
 tar.extractall(release.parent)
resolved=release/'analysis_resolved_config.json'
with resolved.open('x',encoding='utf-8') as stream:json.dump(c['resolved_spec'],stream,ensure_ascii=False,allow_nan=False)
argv=[c['python'],'-u',str(release/'tools/summarize_d92_margin_joint_probe.py'),'--spec',str(resolved),'--run-root',str(run),'--output',str(output)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (release/'analysis.log').open('x',encoding='utf-8') as stream:
 p=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
 with (release/'analysis_process.json').open('x',encoding='utf-8') as evidence:json.dump(dict(pid=p.pid,argv=argv,cwd=str(release),commit=c['commit'],runtime_commit=done['commit'],run_id=c['run_id'],qp_resources=c['qp_resources']),evidence)
 rc=p.wait()
if rc:raise RuntimeError('Analysis failed; preserve '+str(release/'analysis.log'))
summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
if summary.get('status')!=c['summary_status'] or summary.get('summary_schema')!=c['summary_schema'] or summary.get('method')!=c['method'] or summary.get('schema')!=c['schema'] or summary.get('model_rows')!=len(c['rows']) or summary.get('run_id')!=c['run_id']:raise ValueError('Independent analysis incomplete')
if any(summary['coverage'].get(k)!=v for k,v in c['exact'].items()):raise ValueError('Analysis physical coverage incomplete')
if summary.get('qp_resources')!=c['qp_resources'] or summary.get('algorithm')!=c['resolved_spec']['probe']['algorithm'] or summary.get('release_commit')!=done['commit']:raise ValueError('Summary configuration/runtime commit mismatch')
if summary.get('query_rows_used')!=0 or summary.get('source_rows_used')!=0 or summary.get('actual_A') is not None:raise ValueError('Summary access mismatch')
with (output/'analysis_execution.json').open('x',encoding='utf-8') as stream:json.dump(dict(commit=c['commit'],runtime_commit=done['commit'],argv=argv,status='VERIFIED',release=str(release),pid=p.pid,run_id=c['run_id'],output=str(output),qp_resources=c['qp_resources']),stream)
print(json.dumps(dict(status=summary['status'],coverage=summary['coverage'],output=str(output))))
'''


def remote_config(spec, name, commit):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name): raise ValueError('Unsafe analysis release name')
    if not commit_identity(commit): raise ValueError('Actual analysis source commit required')
    base = PurePosixPath(spec['code']['cwd']).parent; release = str(base / name)
    return dict(commit=commit, run_id=spec['run_id'], rows=[row['row_id'] for row in spec['rows']],
        exact=budget_for_spec(spec)['total']['exact'], status=STATUS, method=METHOD, schema=SCHEMA,
        qp_resources=spec['probe']['qp_resources'],resolved_spec=spec,
        summary_status=SUMMARY_STATUS, summary_schema=SUMMARY_SCHEMA,
        run=spec['execution']['remote_run_root'], release=release, archive=str(base / (name + '.tar')),
        output=spec['execution']['remote_run_root'] + '/results/support_summary', python=spec['code']['environment'],
        spec=str(PurePosixPath(spec['code']['cwd']) / spec['spec_path']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--analysis-release', required=True)
    args = parser.parse_args(); spec = json.loads((ROOT / args.spec).read_text(encoding='utf-8')); validate_spec(spec)
    if args.spec != spec['spec_path']: raise ValueError('Analysis spec path differs from declared runtime spec')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], cwd=ROOT, text=True).split()
    if not remote or remote[0] != commit: raise ValueError('Analysis source commit is not independently pushed')
    paths=list(dict.fromkeys(PATHS+[args.spec]+[co['config_path'] for co in spec['probe']['cohorts'].values()]))
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=ROOT, text=True).strip(): raise ValueError('Uncommitted analysis dependencies/configuration')
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing: raise ValueError('Incomplete analysis source bundle: ' + ', '.join(missing))
    cfg = remote_config(spec, args.analysis_release, commit)
    local = Path('E:/type10-7/automation_reports/CV-SincNet') / spec['run_id'] / 'results/support_summary'
    folder = Path('E:/type10-7/local_artifacts/d92_upgrade_20260928') / args.analysis_release
    if local.exists() or folder.exists(): raise FileExistsError('Preserve existing local analysis evidence')
    probe = ('from pathlib import Path\nimport json\nc=' + repr(cfg) + '\n'
        'assert not any(Path(c[k]).exists() for k in ("release","archive","output"))\n'
        'd=json.loads((Path(c["run"])/"complete.json").read_text(encoding="utf-8"))\n'
        's=json.loads((Path(c["run"])/"state.json").read_text(encoding="utf-8"))\n'
        'assert d.get("status")==c["status"] and d.get("run_id")==c["run_id"] and d.get("completed_rows")==d.get("model_rows")==len(c["rows"])\n'
        'assert set(s)==set(c["rows"]) and all(v.get("status")==c["status"] for v in s.values())\n'
        'assert all(d.get(k)==v for k,v in c["exact"].items())\n'
        'assert d.get("qp_resources")==c["qp_resources"] and d.get("workload_complete") is True\n'
        'startup=json.loads((Path(c["run"])/"startup.json").read_text(encoding="utf-8"))\n'
        'assert startup.get("spec")==c["resolved_spec"] and startup.get("commit")==d.get("commit")\n')
    subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'], input=probe.encode('utf-8'), check=True)
    folder.mkdir(parents=True, exist_ok=False); archive = folder / (args.analysis_release + '.tar')
    subprocess.run(['git', 'archive', '--format=tar', '--prefix=' + args.analysis_release + '/', '--output=' + str(archive), commit, *paths], cwd=ROOT, check=True)
    subprocess.run(['scp', *FLAGS, str(archive), 'N607:' + cfg['archive']], check=True)
    script = REMOTE.replace('CONFIG', repr(cfg)); result = subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'], input=script.encode('utf-8'), capture_output=True)
    (folder / 'analysis.stdout').write_bytes(result.stdout); (folder / 'analysis.stderr').write_bytes(result.stderr); result.check_returncode()
    # Separate remote readback: a command's exit status alone is not completion.
    readback = 'import json\nfrom pathlib import Path\np=Path(' + repr(cfg['output']) + ')\nr=Path('+repr(cfg['release'])+')\nprint(json.dumps(dict(summary=json.loads((p/"summary.json").read_text(encoding="utf-8")),execution=json.loads((p/"analysis_execution.json").read_text(encoding="utf-8")),resolved_config=json.loads((r/"analysis_resolved_config.json").read_text(encoding="utf-8")))))\n'
    observed = subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'], input=readback.encode('utf-8'), capture_output=True, check=True)
    evidence = json.loads(observed.stdout.decode('utf-8')); execution_check(evidence,spec,cfg)
    (folder / 'analysis.readback.json').write_text(json.dumps(evidence, allow_nan=False), encoding='utf-8')
    local.parent.mkdir(parents=True, exist_ok=True); subprocess.run(['scp', *FLAGS, '-r', 'N607:' + cfg['output'], str(local)], check=True)
    downloaded = json.loads((local / 'summary.json').read_text(encoding='utf-8')); summary_check(downloaded, spec)
    if downloaded != evidence['summary']: raise ValueError('Downloaded summary differs from independent remote post-state')
    print(json.dumps(dict(status='VERIFIED', summary=str(local / 'summary.json'))))


if __name__ == '__main__': main()
