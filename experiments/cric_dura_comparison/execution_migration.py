"""Resolve execution-only changes after the old dispatcher is retired.

Use original configs to find/adopt every live process first. A G job that became
active during release preparation keeps that original config and process.
"""
from pathlib import Path
from .io import read,write
from .pipeline import done


def promote_pending_G(rt,jobs,previous,active):
    admitted=[]
    for job in jobs:
        key=job['id'];old=previous['jobs'].get(key,{})
        if job['kind']!='G' or key in active or old.get('status')!='QUEUED' or done(job):
            continue
        config=read(job['config'])
        if config.get('stage')!='G' or any(config.get(k) for k in ('F_checkpoint','reader','truth','target_truth','target_inputs','tests')):
            raise ValueError('Only existing source-only G jobs may change execution')
        if Path(config['output_root']).exists():
            continue
        path=Path(rt['control_root'])/'accelerated_configs'/(job['log_name']+'.json')
        updated=dict(config,code_commit=rt['code_commit'],baseline_code_commit=config['code_commit'],
            execution_variant='frozen_teacher_epoch_cache_v1',training_acceleration=dict(
                teacher_cache='epoch',storage='auto',device_limit_mib=64,max_mib=512,
                frozen_prefix=True,skip_zero_spectral=True,reduce_syncs=True))
        if any(updated[k]!=v for k,v in config.items() if k!='code_commit'):
            raise ValueError('Original G scientific configuration changed')
        write(path,updated)
        original=job['config'];job.update(config=str(path),worker_release=rt['release_root'],execution_commit=rt['code_commit'])
        admitted.append(dict(id=key,original_config=original,config=str(path),output_root=config['output_root'],
                             baseline_commit=config['code_commit'],execution_commit=rt['code_commit']))
    write(Path(rt['control_root'])/'G_acceleration_admission.json',dict(
        status='UNSTARTED_G_EXECUTION_RESOLVED',rows=admitted,healthy_worker_ids=list(active),
        original_job_count=len(jobs),no_extra_scientific_rows=True))
    return admitted
