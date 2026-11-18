"""Preserve exact frozen E50 artifacts and retry incomplete source rows."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_anchor_kl.objective import installed
from experiments.pamf_repl_step_guard import runner as scored
from experiments.pamf_repl_view_dose.runner import main
from . import plan


def verify_frozen(folder, done, expected):
    keys = ('row_id', 'tuning_row', 'tuning_recipe', 'model_seed', 'source_contract',
            'source_cache', 'protocol_id', 'shared_prefix', 'paper_row')
    if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
            or done['target_access'] or done['U_truth_read']
            or done['config']['run_id'] != plan.PREVIOUS_RUN
            or any(done['config'][k] != expected[k] for k in keys)):
        raise ValueError('Reused E50 must remain the exact source-only row')
    if not (folder / 'final_ssdg.pth').is_file():
        raise FileNotFoundError('Frozen weights missing')
    for epoch, calls in ((30, 2220), (50, 4440)):
        segment = folder / f'segment{epoch}'
        state = read(segment / 'repl_state.json')
        if state['calls'] != calls or state['U_truth_read']:
            raise ValueError('Blind observations must cover exact source trajectory')
        if not (segment / 'U_pseudo_observations.jsonl.gz').is_file():
            raise FileNotFoundError('Frozen blind observations missing')


def prepare(rt):
    project, root = Path(rt['project']), Path(rt['run_root'])
    previous = project / 'runs' / plan.PREVIOUS_RUN
    state = read(previous / 'pipeline_state.json')
    if state['status'] != 'PARTIAL_TECHNICAL_FAILURE' or not state['failed_jobs']:
        raise ValueError('Wait for the previous owner to preserve all healthy workers')
    for job_id in state['failed_jobs']:
        log = (previous / 'logs' / (job_id + '.log')).read_text(encoding='utf-8', errors='replace')
        if (not job_id.startswith('train__') or 'CUDA out of memory' not in log
                or '_evaluate_source_val_tail_geometry' not in log):
            raise ValueError('Recovery is restricted to the verified source validation OOM')
    if any(job_id.startswith(('predict__', 'score__', 'u_score__')) and job['status'] == 'COMPLETE'
           for job_id, job in state['jobs'].items()):
        raise ValueError('Reconcile existing complete readouts before recovery')
    old_sources = read(previous / 'sources.json')
    jobs = scored.prepare(rt)
    by_id = {job['id']: job for job in jobs}
    sources = read(root / 'sources.json')
    reused = {}
    for row in plan.candidate_rows():
        row_id = row['row_id']
        jid = 'train__' + row_id
        status = state['jobs'][jid]['status']
        if status in ('FAILED', 'QUEUED'):
            continue
        if status != 'COMPLETE':
            raise ValueError('Unexpected prior source state: ' + row_id)
        folder = Path(old_sources[row_id]['source_output'])
        if folder != previous / 'F' / row_id / 'source':
            raise ValueError('Reused E50 folder is outside the original exact row')
        verify_frozen(folder, read(folder/'complete.json'), read(by_id[jid]['config']))
        reused[jid] = str(folder)
        sources[row_id]['source_output'] = str(folder)
        for kind in ('predict', 'u_score'):
            job = by_id[kind + '__' + row_id]
            config = read(job['config'])
            config['source_output'] = str(folder)
            write(job['config'], config, exclusive=False)
            job['dependencies'] = [str(folder / 'complete.json')]
    jobs = [job for job in jobs if job['id'] not in reused]
    write(root / 'sources.json', sources, exclusive=False)
    write(root / 'jobs.json', jobs, exclusive=False)
    write(root / 'recovered_sources.json', dict(previous=plan.PREVIOUS_RUN, reused=reused))
    return jobs


if __name__ == '__main__':
    main(plan, 'experiments.pamf_repl_anchor_kl_recovery.runner', objective=installed,
         max_total=plan.MAX_GPU_PROCESSES, reserve_mib=12288, prepare_jobs=prepare)
