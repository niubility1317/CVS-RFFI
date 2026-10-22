"""Recover only WRONG_REFERENCE device-construction failures, preserving science."""
from pathlib import Path
from .io import read, write

ORIGINAL = '20261022-phase1-PAMF-supervised-physics-m24-r01'
ORIGINAL_TIMING = '20261022-phase1-PAMF-supervised-physics-timing-r01'
RUN = '20261023-phase1-PAMF-physics-device-recovery-r01'
TIMING = '20261023-phase1-PAMF-supervised-physics-timing-recovery-r01'
CONTROL = '20261023-phase1-PAMF-physics-device-control-r01'
ERROR = 'RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!'
KNOWN_TIMING_ERROR = "ModuleNotFoundError: No module named 'baselines.riei_fd'"


def recovery_needed(jobs, state, active=()):
    by = {j['id']: j for j in jobs}
    excluded = set(state.get('excluded_job_ids', []))
    failures = {k: v for k, v in state['jobs'].items()
                if k not in excluded and v['status'] == 'FAILED'}
    bad = []
    for key, item in failures.items():
        if item.get('failure') == KNOWN_TIMING_ERROR:
            continue
        if item.get('failure') != ERROR or key not in by:
            raise ValueError('Unrelated unresolved failure; reconcile before handover')
        job = by[key]
        c = read(job['config'])
        if (job['kind'] != 'F' or key.split('/')[0] != ORIGINAL
                or c.get('supervised_architecture') != 'WRONG_REFERENCE'
                or key in active):
            raise ValueError('Recovery must only replace failed WRONG_REFERENCE F rows')
        for suffix in ('/predict', '/score'):
            consumer = state['jobs'][key + suffix]
            if key + suffix in active or consumer['status'] not in ('QUEUED', 'BLOCKED_TECHNICAL_DEPENDENCY'):
                raise ValueError('Preserve started downstream consumers')
        bad.append(key)
    if not 1 <= len(bad) <= 3:
        raise ValueError('Expected one to three identified architecture construction failures')
    for job in jobs:
        if job['id'].split('/')[0] != ORIGINAL or job['kind'] not in ('F', 'predict'):
            continue
        if job['id'].split('/')[1].startswith('WRONG_REFERENCE-s') and job['id'] in active:
            raise ValueError('Wait for any original WRONG_REFERENCE worker to finish naturally')
    timing = state['jobs'][ORIGINAL_TIMING]
    if ORIGINAL_TIMING in active or timing['status'] not in ('QUEUED', 'BLOCKED_TECHNICAL_DEPENDENCY'):
        raise ValueError('Do not replace started or measured structure timing')
    return sorted(bad)


def smoke(rt, output):
    from .f_binding_recovery import smoke as actual_checkpoint_smoke
    actual_checkpoint_smoke(rt, Path(output).with_name('actual_source_checkpoint_smoke.json'))
    from .f_data import SourceDataset
    from .f_train import arguments, class_dimension, original_config
    from .f_model import install
    from experiments.cvs_phase1_stack.runtime import installed
    import torch
    jobs = read(rt['source_jobs'])
    recovery_needed(jobs, read(rt['previous_state']))
    job = next(j for j in jobs if j['kind'] == 'F'
               and j['id'].startswith(ORIGINAL + '/WRONG_REFERENCE-'))
    c = read(job['config'])
    source = SourceDataset(c['source_cache'], 'L_s', read(c['source_contract']))
    torch.manual_seed(c['model_seed'])
    with class_dimension(c), installed(original_config(c['model_seed'])) as native:
        a = arguments(c, 'cpu'); a.input_len = 256
        model = install(native.build_baseline_model(a, 'cpu'), c)
    x = source.x[:4]
    y = torch.tensor(source.metadata['y'][:4].tolist())
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, y).backward()
    head = model.id_backbone.encoder.core.id_backbone.cls_head.weight
    if not torch.isfinite(logits).all() or head.grad is None or not torch.isfinite(head.grad).all():
        raise ValueError('Corrected architecture cannot legally train on source L')
    write(output, dict(status='PASS', actual_checkpoint_no_query=True,
        architecture='WRONG_REFERENCE', samples=4, source_role='L_s',
        target_reads=0, U_IQ_reads=0, optimizer_updates=0,
        training_initialization='scratch; no trained F used as initialization',
        phase_rng='unchanged CPU seed2026102208 / FP64; device transfer only'))


def prepare(rt, jobs, previous, active):
    bad = recovery_needed(jobs, previous, active)
    by = {j['id']: j for j in jobs}
    excluded = set(previous.get('excluded_job_ids', []))
    configs = Path(rt['control_root']) / 'device_recovery_configs'
    source_map = {}; completion_map = {}; records = []; specs = []
    for key in bad:
        row = key.split('/')[1]
        root = Path(rt['project']) / 'runs' / RUN / row
        if root.exists():
            raise FileExistsError('Recovery output already exists; reconcile without overwrite')
        for suffix, kind, subfolder in (('', 'F', 'source'), ('/predict', 'predict', 'predictions'), ('/score', 'score', 'scores')):
            old = by[key + suffix]; c = read(old['config'])
            target = root / subfolder
            c.update(run_id=RUN, output_root=str(target))
            if kind == 'F':
                source_map[read(old['config'])['output_root']] = str(target)
            if kind == 'predict':
                c.update(source_run_id=RUN, source_output=str(root / 'source'))
            if kind == 'score':
                c.update(prediction_output=str(root / 'predictions'))
            new_key = RUN + '/' + row + suffix
            completion_map[old['completion']] = str(target / 'complete.json')
            specs.append((old, c, new_key, str(target / 'complete.json')))
        records.append(dict(original=key, replacement=RUN + '/' + row,
            old_source_output=read(by[key]['config'])['output_root'],
            source_output=str(root / 'source'), scores=str(root / 'scores/scores.json'),
            scientific_recipe_unchanged=True, seed_unchanged=True))
    timing_root = Path(rt['project']) / 'runs' / TIMING
    if timing_root.exists():
        raise FileExistsError('Timing recovery output already exists')
    old_timing = by[ORIGINAL_TIMING]; tc = read(old_timing['config'])
    tc.update(run_id=TIMING, output_root=str(timing_root))
    tc['F_mechanisms'] = [dict(m, source_output=source_map.get(m['source_output'], m['source_output']))
                          for m in tc['F_mechanisms']]
    specs.append((old_timing, tc, TIMING, str(timing_root / 'complete.json')))
    # Validate all pending execution-only migrations before mutating any queue.
    promoted = []
    for job in jobs:
        key = job['id']
        if (key in excluded or key in active or key in bad
                or key.split('/')[0] != ORIGINAL or job['kind'] not in ('F', 'predict')
                or previous['jobs'][key]['status'] != 'QUEUED'):
            continue
        c = read(job['config'])
        if not key.split('/')[1].startswith('WRONG_REFERENCE-s'):
            continue
        if Path(c['output_root']).exists():
            raise ValueError('Pending migration has existing outputs; preserve and reconcile')
        promoted.append(job)
    new = []
    for old, c, key, completion in specs:
        c.update(code_commit=rt['code_commit'], frozen_sources=rt['frozen_sources'])
        path = configs / (key.replace('/', '__') + '.json')
        write(path, c)
        job = dict(old, id=key, config=str(path), completion=completion,
                   dependencies=[completion_map.get(p, p) for p in old['dependencies']],
                   log_name=key.replace('/', '__'), worker_release=rt['release_root'],
                   execution_commit=rt['code_commit'])
        new.append(job)
        previous['jobs'][key] = dict(status='QUEUED')
        excluded.add(old['id'])
        old_state = previous['jobs'][old['id']]
        if old_state['status'] != 'FAILED':
            old_state['status'] = 'SKIPPED_TECHNICAL_REPLACED'
        old_state['replacement'] = key
    for job in promoted:
        job.update(worker_release=rt['release_root'], execution_commit=rt['code_commit'])
    previous['excluded_job_ids'] = sorted(excluded)
    previous['failures'] = dict(previous.get('failures', {}))
    previous['failures'].pop(ERROR, None)
    record = dict(status='TECHNICAL_DEVICE_RECOVERY', new_jobs=len(new),
                  failed_F_replacements=len(bad), new_hypotheses=0,
                  new_F_budget_or_seeds=0, new_G_training=0,
                  replacements=records, timing_original=ORIGINAL_TIMING, timing_replacement=TIMING,
                  migrated_unstarted_execution_only=[j['id'] for j in promoted],
                  healthy_PIDs_preserved=list(active), old_outputs_preserved=True)
    previous['physics_device_recovery'] = record
    previous.setdefault('resolved_technical_failures', []).append(dict(
        fingerprint=ERROR, replaced_jobs=bad, replacement_run=RUN, old_outputs_preserved=True))
    write(Path(rt['control_root']) / 'physics_device_recovery.json', record)
    return jobs + new
