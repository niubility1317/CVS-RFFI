"""Recover only the interrupted post-smoke owner, with every scientific job unchanged."""
from pathlib import Path
import shutil
from experiments.cric_dura_comparison.io import read,write
from .priority import alive
from .plan import RUN

CONTROL='20261025-phase1-PAMF-PeerSupervised-control-r03'
PREVIOUS_COMMIT='7eb8d21752b5fab6f1366e1cb96c2266bd46c2a7'
FINGERPRINT="PermissionError: [Errno 13] Permission denied: '/proc/3632480/environ'"


def prepared(rt,context):
    previous=read(context['recover_prepared_runtime']);source=Path(previous['run_root']);root=Path(rt['run_root'])
    state=read(source/'pipeline_state.json');jobs=read(source/'jobs.json');smoke=read(source/'smoke.json')
    log=(Path(previous['release_root'])/'controller.log').read_text(errors='replace')
    if (rt['run_id']!=CONTROL or previous['run_id']!=RUN or previous['code_commit']!=PREVIOUS_COMMIT
        or state['code_commit']!=PREVIOUS_COMMIT or alive(state['pid']) or state['status']!='SMOKE_RUNNING'
        or FINGERPRINT not in log or (source/'priority_handoff.json').exists()
        or Path(context['interrupted_handoff_root'])!=source):
        raise ValueError('Only the reproduced interrupted post-smoke controller may recover')
    if (smoke['status']!='PASS' or smoke['optimizer_updates'] or smoke['query_reads']
        or not smoke['peer_parameters_frozen'] or not smoke['peer_buffers_unchanged']
        or smoke['U']['seen']!=256 or smoke['U']['peer_views']!=512 or len(jobs)!=27):
        raise ValueError('Unchanged real-checkpoint no-query source smoke required')
    for job in jobs:
        c=read(job['config'])
        if (job['execution_release']!=previous['release_root'] or c['run_id']!=RUN
            or Path(job['completion']).exists() or Path(c['output_root']).exists()):
            raise ValueError('New scientific row already started or changed; refuse duplicate recovery')
    for name in ('jobs.json','sources.json','reference_scores.json','smoke.json'):
        shutil.copyfile(source/name,root/name)
    write(root/'technical_recovery.json',dict(status='PREPARED_SCIENCE_REUSED',previous_runtime=context['recover_prepared_runtime'],
        scientific_run_id=RUN,scientific_root=str(source),scientific_execution_commit=PREVIOUS_COMMIT,
        controller_commit=rt['code_commit'],new_scientific_jobs=27,configs_changed=0,budget_changes=0,
        source_smoke_reused=True,reason='Exit liveness uses proc stat, not environ; no worker killed or retrained'))
    return jobs,str(source)
