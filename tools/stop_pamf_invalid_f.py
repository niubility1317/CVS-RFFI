"""Stop only registered input-invalid F workers, preserving valid worker PIDs.

The existing controller counts terminated workers as technical failures. Retire
that controller, record explicit user-stop statuses in a separate handover
snapshot, and reuse its already-published controller code with identical jobs.
No model configuration, data, checkpoint, prediction or score is modified.
"""
import argparse
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import copy,datetime,json,os,signal,subprocess,time
from pathlib import Path
OPTIONS=__OPTIONS__
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def now():return datetime.datetime.now().astimezone().isoformat()
rt=OPTIONS['runtime'];root=Path(rt['project']);old_release=Path(rt['release_root'])
old_control=Path(rt['control_root']);state_path=old_control/'pipeline_state.json'
assert os.geteuid()!=0,'Administrator execution forbidden'
assert read(rt['runtime_file'])==rt,'Current runtime changed; reconcile before stopping'
for _ in range(8):
    handoff=old_control/'multiprocess_handoff.json'
    if not handoff.exists():break
    next_runtime=Path(read(handoff)['new_runtime'])
    assert next_runtime.is_relative_to(root/'releases'),'Handover left project releases'
    rt=read(next_runtime)
    assert rt['project']==str(root) and rt['main_run']==OPTIONS['runtime']['main_run']
    old_release=Path(rt['release_root']);old_control=Path(rt['control_root'])
else:raise RuntimeError('Too many handovers; inspect current owner')
assert not (old_control/'multiprocess_handoff.json').exists()
state_path=old_control/'pipeline_state.json'
state=read(state_path);jobs=read(rt['jobs']);by_config={j['config']:j for j in jobs}
assert len(by_config)==len(jobs),'Duplicate job configurations'
legal={v['original'] for v in state['F_binding_recovery']['replacements']}
assert len(legal)==69 and legal<=set(state['excluded_job_ids'])
controller=state['pid'];controller_path=Path('/proc')/str(controller)
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        args=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid,start=stat[19],ppid=int(stat[1]),args=args,
                    cwd=os.readlink(p/'cwd'),uid=p.stat().st_uid)
    except (FileNotFoundError,ProcessLookupError):return None
cp=proc(controller)
assert cp and cp['uid']==os.geteuid()
assert cp['args']==[rt['python'],'-u','-m','experiments.cric_dura_comparison.multiprocess_control','--config',rt['runtime_file']]
assert cp['cwd']==str(old_release),'Controller CWD changed'
compile((old_release/'experiments/cric_dura_comparison/multiprocess_control.py').read_text(),'<existing-controller>','exec')
active=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:q=proc(int(p.name))
    except PermissionError:continue
    if not q or '--config' not in q['args']:continue
    config=q['args'][q['args'].index('--config')+1]
    job=by_config.get(config)
    if not job:continue
    assert q['args']==[rt['python'],'-u','-m','experiments.cric_dura_comparison.pipeline','worker','--config',config]
    assert q['cwd']==job.get('worker_release',rt['worker_release']) and q['uid']==os.geteuid()
    q.update(job=job['id'],config=config,kind=job['kind'])
    active.append(q)
targets=[q for q in active if q['job'] in legal]
protected=[q for q in active if q['job'] not in legal]
for q in targets:
    config=read(q['config'])
    assert q['kind']=='F' and config['protocol_id']=='F0'
    assert config['run_id'] in ('20261021-phase1-T1-DURA-F-ablation-m63-r01','20261021-phase1-T1-DURA-comparison-m153-r01')
    assert 'binding' not in q['config'],'Never stop a repaired F worker'
new_id=OPTIONS['new_run_id'];assert '/' not in new_id and '\\' not in new_id
new_release=root/'releases'/new_id;new_control=root/'runs'/new_id
assert not new_release.exists() and not new_control.exists(),'New control output already exists'
plan=dict(observed_at=now(),controller=cp,targets=targets,protected=protected,
          new_run_id=new_id,preserve_all_artifacts=True,scientific_jobs_changed=False,
          controller_code_commit=rt['code_commit'])
if not OPTIONS['execute'] or not targets:
    print(json.dumps(dict(status='PREFLIGHT_PASS' if targets else 'ALREADY_STOPPED',**plan)))
else:
    # Only user-owned controllers and the precisely matched invalid F tree receive signals.
    new_release.mkdir()
    write(new_release/'user_stop_preflight.json',plan)
    write(new_release/'previous_state_before_stop.json',state)
    def same(q):
        current=proc(q['pid'])
        if not current:return False
        assert all(current[k]==q[k] for k in ('start','args','cwd','uid')),'PID identity changed'
        return True
    def terminate(q):
        if same(q):os.kill(q['pid'],signal.SIGTERM)
        deadline=time.monotonic()+8
        while proc(q['pid']) and time.monotonic()<deadline:time.sleep(.1)
        if same(q):os.kill(q['pid'],signal.SIGKILL)
        deadline=time.monotonic()+5
        while proc(q['pid']) and time.monotonic()<deadline:time.sleep(.1)
        assert not proc(q['pid']),'Process did not stop'
    # Reconcile one final state after the sole dispatcher exits; valid workers keep running.
    terminate(cp)
    try:state=read(state_path)
    except (OSError,ValueError):pass
    # Snapshot descendants only after dispatch is retired, before terminating their parent.
    descendants=[];frontier={q['pid'] for q in targets}
    while frontier:
        children=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:q=proc(int(p.name))
            except PermissionError:continue
            if q and q['ppid'] in frontier:
                assert q['uid']==os.geteuid() and q['pid'] not in {v['pid'] for v in protected}
                children.append(q)
        descendants.extend(children);frontier={q['pid'] for q in children}
    for q in targets+list(reversed(descendants)):terminate(q)
    for q in targets:
        item=state['jobs'][q['job']]
        item.update(status='STOPPED_BY_USER',stop_reason='INVALID_INPUT_BINDING; explicit user stop',stopped_at=now())
    # Original queued input-invalid jobs must never be restarted by the next owner.
    for key in legal:
        if state['jobs'][key]['status']=='QUEUED':
            state['jobs'][key].update(status='STOPPED_BY_USER',stop_reason='INVALID_INPUT_BINDING; explicit user stop')
    state['user_stop_invalid_F']=dict(at=now(),jobs=[q['job'] for q in targets],preserve_all_artifacts=True)
    handover_state=new_release/'previous_state_after_user_stop.json';write(handover_state,state)
    new_rt=copy.deepcopy(rt)
    new_rt.update(run_id=new_id,release_root=str(new_release),control_root=str(new_control),
        runtime_file=str(new_release/'runtime.json'),previous_state=str(handover_state),
        previous_runtime=rt['runtime_file'],previous_control_root=str(old_control),
        previous_controller_cwd=str(old_release),source_jobs=rt['jobs'],jobs=rt['jobs'],
        launch_owner='codex-user-stop-invalid-F-20261022',F_source_binding_recovery=False,
        G_teacher_acceleration=False,supervised_extension=False,
        controller_code_release=str(old_release))
    write(new_release/'runtime.json',new_rt)
    command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.multiprocess_control','--config',new_rt['runtime_file']]
    env=dict(os.environ,PYTHONPATH=str(old_release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (new_release/'continuation.log').open('x') as log:
        child=subprocess.Popen(command,cwd=new_release,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(new_release/'submit.json',dict(pid=child.pid,command=command,at=now(),code_commit=rt['code_commit']))
    still_same=[q['pid'] for q in protected if same(q)]
    naturally_finished=[q['pid'] for q in protected if not proc(q['pid'])]
    result=dict(status='STOPPED_AND_HANDOVER_SUBMITTED',at=now(),stopped=[q['job'] for q in targets],
        stopped_pids=[q['pid'] for q in targets],protected_same_pid=still_same,
        protected_naturally_finished=naturally_finished,new_controller_pid=child.pid,
        new_runtime=new_rt,new_control_root=str(new_control),failure_counts_unchanged=state.get('failures'))
    write(new_release/'user_stop_result.json',result)
    print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--ssh-config', required=True)
    p.add_argument('--new-run-id', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Output exists; inspect its result before retrying')
    options = dict(runtime=json.loads(a.runtime.read_text(encoding='utf-8-sig')),
                   new_run_id=a.new_run_id, execute=a.execute)
    code = REMOTE.replace('__OPTIONS__', repr(options))
    compile(code, '<bounded-user-stop>', 'exec')
    r = subprocess.run(['ssh', '-F', a.ssh_config, '-T', '-o', 'BatchMode=yes',
                        '-o', 'ConnectTimeout=10', 'N607', 'python3 -'],
                       input=code.encode('utf-8'), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=55)
    if r.returncode:
        raise SystemExit(r.stderr.decode('utf-8', errors='replace'))
    result = json.loads(r.stdout.decode('utf-8'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result.get(k) for k in ('status','at','observed_at','stopped',
        'stopped_pids','protected_same_pid','protected_naturally_finished','new_controller_pid')}, ensure_ascii=False))
    if not a.execute:
        print(json.dumps(dict(targets=[dict(job=q['job'],pid=q['pid']) for q in result['targets']],
                              protected_count=len(result['protected'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
