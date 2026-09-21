"""Single-owner Linux dispatcher for the authorized scratch CORE90 matrix."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts.core90_cross_response_matrix import build_matrix, load_config, resolve_variant, VARIANTS, V2_VARIANTS

ROLES = dict(wisig_train_rxs='1,3,4,6,8', wisig_test_rxs='0,2,5,7,9,10,11',
             wisig_train_days='1,2,3', wisig_test_days='0')
SCENES = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


def training_command(python, argv):
    return [python, str(CODE/'SSDG/train_ssdg.py'), *argv], CODE


def detached_command(python, argv):
    return [python, str(Path(__file__).resolve()), *[value for value in argv if value != '--detach']], CODE.parent


def source_cell_counts(base, indices, k, role):
    cells = Counter((base.index[i].tx_i, base.index[i].rx_i, base.index[i].day_i, base.index[i].eq_i) for i in indices)
    expected = {(row.tx_i,row.rx_i,row.day_i,row.eq_i) for row in base.index}
    if any(cells[cell] < k for cell in expected):
        raise ValueError(f'{role} cannot provide K={k} records for every source TX/RX/day/condition cell')
    return {':'.join(map(str,cell)): n for cell,n in sorted(cells.items())}


def source_block_feasibility(records, config, *, validation):
    from cvsrffi.cross_response.sampler import CrossBlockSampler
    limit = int(config['source_eval_max_blocks'])
    p,q,k = config['P'],config['Q'],config['K']
    sampler = CrossBlockSampler(records, P=p,Q=q,k_menu=(k,),batch_size=p*q*k,
        seed=config['data_seed'],max_candidates=config['scheduler_candidate_limit'],steps_per_epoch=limit)
    valid = 0
    for _ in range(limit):
        plan = sampler.next_batch()
        for block in plan.blocks:
            block.validate()
            valid += 1
    if valid < limit:
        raise ValueError(f'{"validation" if validation else "training"} source cannot form {limit} complete blocks')
    return dict(candidate_count=len(sampler.candidates), candidate_search_complete=sampler.candidate_search_complete,
                requested_blocks=limit, valid_blocks=valid, metadata_only=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def failure_fingerprint(log_path):
    lines = Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines()
    errors = [line.strip() for line in lines if re.search(r'(?:Error|Exception|AssertionError|CUDA out of memory)\b', line)]
    message = errors[-1] if errors else '\n'.join(lines[-5:])
    message = re.sub(r'0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\b', '#', message)
    return message or 'EMPTY_PROCESS_LOG'


def inspect_completion(directory, returncode):
    directory = Path(directory)
    pred = directory / 'final_predictions'
    required = [directory/'final_ssdg.pth', pred/'prediction_manifest.json', pred/'independent_scores.json']
    required += [pred/f'predictions.{scene}.jsonl' for scene in SCENES]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        return dict(status='FAILED', reason='ARTIFACT_CLOSURE_INCOMPLETE', missing=missing)
    try:
        manifest = json.loads((pred/'prediction_manifest.json').read_text(encoding='utf-8'))
        scores = json.loads((pred/'independent_scores.json').read_text(encoding='utf-8'))
        if manifest.get('state') != 'PREDICTIONS_FIXED' or manifest.get('scenarios') != list(SCENES):
            raise ValueError('four-scene fixed prediction manifest required')
        count = manifest.get('record_count', 0)
        if count <= 0 or manifest.get('rows_written') != count * 4:
            raise ValueError('prediction manifest row counts incomplete')
        for scene in SCENES:
            with (pred/f'predictions.{scene}.jsonl').open(encoding='utf-8') as handle:
                if sum(bool(line.strip()) for line in handle) != count:
                    raise ValueError(f'{scene} prediction row count incomplete')
        if scores.get('status') != 'COMPLETE' or scores.get('truth_last') is not True or scores.get('prediction_rows') != count * 4:
            raise ValueError('independent truth-last scoring incomplete')
        if not (directory/'cross_response_activation.json').is_file():
            return dict(status='FAILED', reason='CROSS_RESPONSE_ACTIVATION_INCOMPLETE', missing=['cross_response_activation.json'])
        activation = json.loads((directory/'cross_response_activation.json').read_text(encoding='utf-8'))
        if activation.get('missing') or activation.get('status') != 'ACTIVE_VERIFIED':
            return dict(status='FAILED', reason='CROSS_RESPONSE_ACTIVATION_INCOMPLETE', activation=activation)
    except (ValueError, OSError) as exc:
        return dict(status='FAILED', reason='ARTIFACT_VALIDATION_FAILED', detail=str(exc))
    return dict(status='COMPLETE' if returncode == 0 else 'FAILED',
                reason='VERIFIED_FINAL_ARTIFACTS' if returncode == 0 else 'NONZERO_EXIT_AFTER_ARTIFACTS')


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {parts[0].strip(): int(parts[1]) for line in raw.splitlines() if (parts := line.split(',')) and len(parts) == 2}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    registered = {int(parts[0]): parts[1].strip() for line in raw.splitlines() if len(parts := line.split(',')) == 2 and parts[0].strip().isdigit()}
    return free, registered


def choose_gpu(devices, active, free, registered, *, minimum, limit, total_limit=2):
    candidates = []
    for gpu in devices:
        owned = [job for job in active if job['gpu'] == gpu]
        reservation = sum(minimum for job in owned if registered.get(job['pid']) != gpu)
        total = sum(value == gpu for value in registered.values()) + sum(
            registered.get(job['pid']) != gpu for job in owned)
        available = free.get(gpu, 0) - reservation
        if len(owned) < limit and total < total_limit and available >= minimum:
            candidates.append((len(owned), -available, gpu))
    return min(candidates)[2] if candidates else None


def source_preflight(args, destination):
    """Only source indexing and source IQ forward; no heldout loader or scoring."""
    import torch
    from torch.utils.data import DataLoader
    from SSDG import train_ssdg as train
    from cvsrffi.cross_response.config import validate_runtime_config
    from cvsrffi.cross_response.integration import CrossResponseRuntime, forward_labeled
    torch.set_num_threads(2)
    config = load_config(args.config)
    row = build_matrix(config_path=args.config, wisig_pkl=str(args.dataset), output_root=destination,
                       roles=ROLES, seeds=[args.seed], variants=['U0'])[0]
    parsed = train.build_arg_parser().parse_args(row['argv'])
    ds = train.load_wisig_compact_pkl(str(args.dataset))
    eq = 'both' if str(parsed.wisig_equalized).lower() == 'both' else int(parsed.wisig_equalized)
    base = train.WiSigCompactDataset(ds, out_len=parsed.wisig_out_len, crop_mode='center', normalize=True,
        equalized=eq, rx_keep=[1,3,4,6,8], day_keep=[1,2,3], domain=parsed.wisig_domain,
        max_samples_per_combo=None if parsed.wisig_max_day123_per_combo <= 0 else parsed.wisig_max_day123_per_combo,
        seed=config['cross_response']['data_seed'], build_index=True)
    labeled, unlabeled, val = train.split_tx_rx_day_1_7_2(base, labeled_ratio=.07, unlabeled_ratio=.63, source_val_ratio=.30)
    counts = {}
    for role, indices in [('L_s', labeled), ('V_s', val)]:
        counts[role] = source_cell_counts(base, indices, config['cross_response']['K'], role)
    parsed.num_classes = len(ds['tx_list'])
    domain_map = train.build_domain_label_map(base)
    domains = max(1, len(domain_map))
    device = torch.device('cpu')
    train.set_seed(args.seed)
    model_args = train.merge_checkpoint_args({'args': {}, 'model': None, 'stats': {}, 'split_info': None}, parsed,
                                            input_len=parsed.wisig_out_len, num_domains=domains)
    model_args = train._apply_model_cli_args(model_args, parsed)
    model = train.build_baseline_model(model_args, device).eval()
    sample = base[labeled[0]][0]
    sample = sample if torch.is_tensor(sample) else torch.tensor(sample.tolist(), dtype=torch.float32)
    with torch.no_grad():
        reference = model(sample.unsqueeze(0).to(device), return_aux=False)
    checkpoint = destination/'scratch_smoke.pth'
    torch.save({'model': model.state_dict(), 'provenance': 'THIS_PREFLIGHT_FRESH_INITIALIZATION_ONLY'}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    # numpy 2 / old torch bridge is deliberately avoided in this smoke boundary.
    with torch.no_grad():
        output = model(sample.unsqueeze(0).to(device), return_aux=False)
    logits = output['tx_logits'] if isinstance(output, dict) else output
    expected_logits = reference['tx_logits'] if isinstance(reference, dict) else reference
    if not torch.isfinite(logits).all() or not torch.equal(logits, expected_logits):
        raise ValueError('source scratch strict-load smoke produced nonfinite output')
    ctx = dict(input_len=parsed.wisig_out_len, domain_label_map=domain_map)
    for name, indices, role in [('train_loader', labeled, 'ssdg_labeled_tx_visible'),
                                ('unlabeled_loader', unlabeled, 'ssdg_unlabeled_tx_hidden'),
                                ('val_loader', val, 'ssdg_source_v_cal')]:
        ctx[name] = DataLoader(train.WiSigSubsetDataset(base, indices, split_source=role),
                               batch_size=parsed.batch_size, num_workers=0)
    parsed.cross_response_variant = 'U4_bilinear'
    parsed.output_dir = str(destination/'scratch_runtime')
    runtime_config = validate_runtime_config(resolve_variant(config, 'U4_bilinear'))
    runtime = CrossResponseRuntime(model, ctx, runtime_config, parsed, device)
    feasibility = dict(train=source_block_feasibility(runtime.records, runtime_config, validation=False),
                       validation=source_block_feasibility(runtime.validation_records, runtime_config, validation=True))
    x, y, d, metadata = next(iter(ctx['train_loader']))
    plan = runtime.begin_batch((d, metadata))
    if not plan.blocks or runtime.joint_open:
        raise ValueError('source preflight requires effective blocks and the unchanged closed source gate')
    out = forward_labeled(model, runtime, plan, x, return_aux=True)
    terms = runtime.losses(model, out, y, plan)
    if any(not torch.isfinite(term) or not term.requires_grad for term in terms[:2]):
        raise ValueError('source response/decision losses must be finite and differentiable')
    zero = out['tx_logits'].sum() * 0.
    # Exercise production gradient routing: raw backward would include the
    # shared domain frontend, which the formal runtime deliberately excludes.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    runtime.backward(model, zero, zero, (terms[0], zero, zero), scaler)
    grad_norms = {}
    for name, parameters in [('head', runtime.parameters()), ('domain', list(model.dom_backbone.parameters()))]:
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError(f'source {name} gradients missing or nonfinite')
        grad_norms[name] = sum(float(grad.square().sum()) for grad in gradients) ** .5
        if grad_norms[name] <= 0:
            raise ValueError(f'source {name} response gradient is zero')
    identity_response = sum(float(p.grad.square().sum()) for p in model.id_backbone.parameters() if p.grad is not None)
    if identity_response != 0:
        raise ValueError('closed source gate leaked response gradient to identity')
    model.zero_grad(set_to_none=True)
    # response_backward consumes the graph; obtain an independent source-only
    # forward for decision differentiability without updating parameters.
    out = forward_labeled(model, runtime, plan, x, return_aux=True)
    terms = runtime.losses(model, out, y, plan)
    terms[1].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('source decision backward nonfinite')
    result = dict(status='VERIFIED', roles=ROLES, model_seed=args.seed, data_seed=config['cross_response']['data_seed'],
                  source_counts=dict(L_s=len(labeled), U_s=len(unlabeled), V_s=len(val)), cells=counts,
                  checkpoint=str(checkpoint), strict_load=True, fresh_initialization=True,
                  strict_forward_exact=True, effective_blocks=len(plan.blocks),
                  response_loss=float(terms[0].detach()), decision_loss=float(terms[1].detach()),
                  response_gradient_norms=grad_norms, identity_response_gradient_squared=identity_response,
                  source_gate_open=runtime.joint_open,
                  source_block_feasibility=feasibility,
                  target_loader_constructed=False, target_iq_forwarded=False,
                  note='compact pickle deserialized as supplied; only source index and source IQ used')
    write_json(destination/'source_preflight.json', result)
    return result


def dispatch(args):
    root = args.project_root.resolve()
    runs = (args.runs_root or root/'runs') / args.run_id
    logs = (args.logs_root or root/'logs') / args.run_id
    if runs.exists() or logs.exists():
        raise FileExistsError('run and log directories must both be new; existing artifacts are preserved')
    runs.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    lock = runs/'dispatcher.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), started=time.time(), run_id=args.run_id), handle)
    state = dict(run_id=args.run_id, status='PREFLIGHT', dispatcher_pid=os.getpid(), roles=ROLES,
                 total_jobs_per_gpu=2, new_jobs_per_gpu=args.new_per_gpu, variants=args.variants,
                 model_seed=args.seed, jobs=[], started=time.time())
    statepath = runs/'pipeline.json'
    write_json(statepath, state)
    try:
        source_preflight(args, runs)
        if args.preflight_only:
            state['status'] = 'PREFLIGHT_COMPLETE'
            write_json(statepath, state)
            return
        free, _ = gpu_snapshot()
        devices = args.devices or list(free)
        if not devices or any(gpu not in free for gpu in devices):
            raise ValueError('devices must be available nvidia-smi GPU UUIDs')
        rows = build_matrix(config_path=args.config, wisig_pkl=str(args.dataset), output_root=runs,
                            roles=ROLES, seeds=[args.seed], variants=args.variants)
        for row in rows:
            variant = row['variant']
            argv = row['argv']
            output = runs/variant
            for flag, value in [('--output_dir', str(output)), ('--run_id', args.run_id), ('--device', 'cuda:0')]:
                if flag in argv:
                    argv[argv.index(flag)+1] = value
                else:
                    argv += [flag, value]
            command, cwd = training_command(args.python, argv)
            state['jobs'].append(dict(variant=variant, status='PENDING', output=str(output),
                                      log=str(logs/f'{variant}.log'), argv=command, cwd=str(cwd)))
        state['status'] = 'RUNNING'
        write_json(statepath, state)
        active = {}
        fingerprints = Counter()
        while active or any(job['status'] == 'PENDING' for job in state['jobs']):
            for variant, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                job = next(job for job in state['jobs'] if job['variant'] == variant)
                job.update(inspect_completion(job['output'], code), returncode=code, ended=time.time())
                del active[variant]
                if job['status'] == 'FAILED' and not (Path(job['output'])/'final_predictions/prediction_manifest.json').exists():
                    fingerprint = failure_fingerprint(job['log'])
                    job['failure_fingerprint'] = fingerprint
                    fingerprints[fingerprint] += 1
                    if fingerprints[fingerprint] >= 2:
                        state['queue_stopped_reason'] = 'REPEATED_PRE_PREDICTION_FAILURE'
                        for pending in state['jobs']:
                            if pending['status'] == 'PENDING':
                                pending.update(status='FAILED', reason='NOT_LAUNCHED_REPEATED_FAILURE')
            if not state.get('queue_stopped_reason'):
                free, registered = gpu_snapshot()
                for job in state['jobs']:
                    if job['status'] != 'PENDING':
                        continue
                    owned = [entry for entry in state['jobs'] if entry['status'] == 'RUNNING']
                    gpu = choose_gpu(devices, owned, free, registered, minimum=args.min_free_mib, limit=args.new_per_gpu)
                    if gpu is None:
                        break
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                    with Path(job['log']).open('xb') as handle:
                        process = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env, stdout=handle,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status='RUNNING', gpu=gpu, pid=process.pid, started=time.time())
                    active[job['variant']] = process
                    write_json(statepath, state)
            write_json(statepath, state)
            if active or any(job['status'] == 'PENDING' for job in state['jobs']):
                time.sleep(args.poll_seconds)
        state['status'] = 'COMPLETE' if all(job['status'] == 'COMPLETE' for job in state['jobs']) else 'FAILED'
        state['ended'] = time.time()
        write_json(statepath, state)
    except BaseException as exc:
        state.update(status='FAILED', dispatcher_error=f'{type(exc).__name__}: {exc}', ended=time.time())
        write_json(statepath, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--devices', nargs='+')
    p.add_argument('--seed', type=int, default=392005)
    p.add_argument('--variants', choices=V2_VARIANTS, nargs='+', default=list(VARIANTS))
    p.add_argument('--min-free-mib', type=int, default=6500)
    p.add_argument('--new-per-gpu', type=int, choices=[1,2], default=2)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--logs-root', type=Path)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('run-id must be a single safe directory name')
    if args.seed != 392005 or load_config(args.config)['cross_response']['data_seed'] != 392005:
        raise ValueError('authorized model/data seed is 392005')
    from cvsrffi.cross_response.config import validate_runtime_config
    if len(args.variants) != len(set(args.variants)):
        raise ValueError('duplicate variants would share output paths')
    for variant in args.variants:
        validate_runtime_config(resolve_variant(load_config(args.config), variant))
    if args.min_free_mib < 6500 or args.poll_seconds <= 0:
        raise ValueError('minimum free memory must be >=6500 MiB and polling positive')
    if args.detach:
        if os.name != 'posix':
            raise RuntimeError('detached dispatcher is Linux-only')
        logfile = (args.logs_root or args.project_root/'logs')/f'{args.run_id}.dispatcher.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open('xb') as handle:
            argv, cwd = detached_command(args.python, sys.argv[1:])
            process = subprocess.Popen(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(status='DISPATCHED', pid=process.pid, log=str(logfile))))
    else:
        dispatch(args)


if __name__ == '__main__':
    main()
