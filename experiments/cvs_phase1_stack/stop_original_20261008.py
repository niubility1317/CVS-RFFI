"""Explicit user stop: original136-row stack only; preserve all artifacts."""
from pathlib import Path
import os,json,time,signal,subprocess
EXPECTED = [{'pid': 1234553, 'ppid': 4170278, 'start_ticks': 67492558, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-selected-s2026092701.json']}, {'pid': 1234562, 'ppid': 4170278, 'start_ticks': 67492604, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-full-s2026092701.json']}, {'pid': 1234572, 'ppid': 4170278, 'start_ticks': 67492646, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-native_full-s2026092701.json']}, {'pid': 1234645, 'ppid': 4170278, 'start_ticks': 67492689, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_leo-s2026092701.json']}, {'pid': 1234722, 'ppid': 4170278, 'start_ticks': 67492748, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_mixstyle-s2026092701.json']}, {'pid': 1234867, 'ppid': 4170278, 'start_ticks': 67492823, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_pseudo-s2026092701.json']}, {'pid': 1234943, 'ppid': 4170278, 'start_ticks': 67492877, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_dg-s2026092701.json']}, {'pid': 1234954, 'ppid': 4170278, 'start_ticks': 67492927, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_proto_compact-s2026092701.json']}, {'pid': 1235027, 'ppid': 4170278, 'start_ticks': 67492968, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_open_boundary-s2026092701.json']}, {'pid': 1235109, 'ppid': 4170278, 'start_ticks': 67493048, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_daot-s2026092701.json']}, {'pid': 1235248, 'ppid': 4170278, 'start_ticks': 67493105, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_rc4-s2026092701.json']}, {'pid': 1235263, 'ppid': 4170278, 'start_ticks': 67493153, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-selected-s2026092702.json']}, {'pid': 1263815, 'ppid': 4170278, 'start_ticks': 67856294, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-full-s2026092702.json']}, {'pid': 1275729, 'ppid': 4170278, 'start_ticks': 68001585, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-native_full-s2026092702.json']}, {'pid': 1277202, 'ppid': 4170278, 'start_ticks': 68015950, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_leo-s2026092702.json']}, {'pid': 1336359, 'ppid': 4170278, 'start_ticks': 68759351, 'state': 'R', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.source', '--config', '/home/szu2070436088/2510044040/CV-SincNet/runs/20261006-phase1-reference-stack-manysig-m136-r02/configs/source-r6-no_mixstyle-s2026092702.json']}, {'pid': 4170278, 'ppid': 1, 'start_ticks': 50415368, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.recover']}, {'pid': 4170279, 'ppid': 1, 'start_ticks': 50415368, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_recovery_20261006_r03', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.sixscene_after', '--mode', 'dispatch']}, {'pid': 4176081, 'ppid': 1, 'start_ticks': 50471903, 'state': 'S', 'cwd': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_reference_stack_evaluation_repair_20261006_r01', 'argv': ['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python', '-u', '-m', 'experiments.cvs_phase1_stack.evaluation_recovery']}]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
MAIN=PROJECT/'runs/20261006-phase1-reference-stack-manysig-m136-r02'
CONTROLLERS={4170278,4170279,4176081}
def proc(pid):
    d=Path('/proc')/str(pid)
    try:
        stat=(d/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,ppid=int(stat[1]),start_ticks=int(stat[19]),state=stat[0],uid=d.stat().st_uid,cwd=str((d/'cwd').resolve(strict=True)),argv=[x for x in (d/'cmdline').read_bytes().decode().split('\0') if x])
    except (FileNotFoundError,ProcessLookupError):return None
def same(expected):
    current=proc(expected['pid'])
    if current is None or current['state']=='Z':return None
    if current['uid']!=os.getuid() or any(current[k]!=expected[k] for k in ['pid','start_ticks','cwd','argv']):raise RuntimeError('PID identity changed '+str(expected['pid']))
    return current
def table():
    out={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            value=proc(int(p.name))
            if value:out[value['pid']]=value
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return out
assert subprocess.check_output(['whoami'],text=True).strip()=='szu2070436088'
assert subprocess.check_output(['hostname'],text=True).strip()=='dell-DSS8440'
before=[same(p) for p in EXPECTED]
assert all(before),'Reconcile missing processes before stop'
assert all(p['ppid']==4170278 for p in before if p['pid'] not in CONTROLLERS)
allowed_cwds={p['cwd'] for p in EXPECTED}
request=MAIN/'user_stop_20261008.json'
with request.open('x') as f:json.dump(dict(status='USER_STOP_REQUESTED',requested_at=time.time(),authorization='User: stop original subsequent experiments, including running ones',controllers=sorted(CONTROLLERS),expected=EXPECTED,queue_before=json.loads((MAIN/'queue_state.json').read_text()),preserve_all_artifacts=True),f,indent=2)
events=[]
def send(p,sig):
    if same(p):os.kill(p['pid'],sig);events.append(dict(pid=p['pid'],signal=sig.name,time=time.time()))
for p in EXPECTED:
    if p['pid'] in CONTROLLERS:send(p,signal.SIGSTOP)
# Controllers are frozen before discovering and stopping descendants.
allp=table();owned={p['pid']:p for p in EXPECTED}
while True:
    added=[p for p in allp.values() if p['ppid'] in owned and p['pid'] not in owned]
    if not added:break
    for p in added:
        if p['cwd'] not in allowed_cwds:raise RuntimeError('Unexpected descendant cwd '+str(p))
        owned[p['pid']]=p
for p in owned.values():
    if p['pid'] not in CONTROLLERS:send(p,signal.SIGTERM)
for p in owned.values():
    if p['pid'] in CONTROLLERS:
        send(p,signal.SIGTERM);send(p,signal.SIGCONT)
deadline=time.monotonic()+15
while time.monotonic()<deadline and any(same(p) for p in owned.values()):time.sleep(.5)
for p in owned.values():
    if same(p):send(p,signal.SIGKILL)
time.sleep(1)
remaining=[p for p in owned.values() if same(p)]
matched=[p for p in table().values() if p['state']!='Z' and p['cwd'] in allowed_cwds and 'experiments.cvs_phase1_stack.' in ' '.join(p['argv'])]
result=dict(status='VERIFIED' if not remaining and not matched else 'UNKNOWN',stopped_at=time.time(),owned_pids=list(owned),events=events,remaining=remaining,matched_remaining=matched,preserved_all_artifacts=True)
with (MAIN/'user_stop_20261008_result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
