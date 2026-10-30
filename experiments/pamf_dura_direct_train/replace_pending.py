"""Replace only the still-unstarted, exactly owned R01 CPU queue after approval."""
import json
import os
from pathlib import Path
import signal
import time

OLD_RUN='20261030-phase1-DURA-source-headFeature-s2026092701-r01'
OLD_COMMIT='d79a280d843696d680c82bf63b0138905cf1cb5d'
NEW_RUN='20261030-phase1-DURA-source-headFeature-s2026092701-r02'


def validate_pending(old,new,launch,cwd,command,children):
    if old.name!=OLD_RUN or new.name!=NEW_RUN or new.exists():
        raise ValueError('Exact existing old queue and absent new root required')
    if launch.get('commit')!=OLD_COMMIT or Path(cwd).name!='20261030-DURA-source-headFeature-r01':
        raise ValueError('Original owned queue commit/CWD differs')
    if 'experiments.pamf_dura_direct_train.head_guided run' not in command or OLD_COMMIT not in command:
        raise ValueError('Original owned controller command differs')
    if children or any((old/name).exists() for name in ('submitted-scratch_joint.json','scratch_joint','complete.json','resource_replaced.json')):
        raise ValueError('Original row started or already transitioned; preserve it without replacement')


def replace_pending(root):
    import fcntl
    root=Path(root);old=root/'runs'/OLD_RUN;new=root/'runs'/NEW_RUN
    with (root/'runs/receiver_residual_capacity4.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        launch=json.loads((old/'launch.json').read_text());pid=launch['pid'];proc=Path('/proc')/str(pid)
        if not proc.exists():raise ValueError('Original CPU controller is not live; reconcile rather than replace blindly')
        cwd=str((proc/'cwd').resolve());command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        children=(proc/'task'/str(pid)/'children').read_text().strip()
        validate_pending(old,new,launch,cwd,command,children)
        if (Path(cwd)/'release_commit.txt').read_text().strip()!=OLD_COMMIT:
            raise ValueError('Original release code identity differs')
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+5
        while proc.exists() and time.monotonic()<deadline:
            try:state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:break
            if state=='Z':break
            time.sleep(.1)
        if proc.exists():
            try:state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:state=None
            if state not in (None,'Z'):raise RuntimeError('Original controller termination UNKNOWN; do not launch replacement')
        result=dict(status='OWNED_PENDING_CONTROLLER_STOPPED',old_run=OLD_RUN,new_run=NEW_RUN,
            old_pid=pid,old_commit=OLD_COMMIT,no_old_GPU_worker=True,outputs_preserved=True,
            other_processes_modified=0,reason='Explicit user-approved resource-only queue replacement')
        with (old/'resource_replaced.json').open('x',encoding='utf-8') as output:
            json.dump(result,output,indent=2);output.write('\n')
        return result
