"""Single-owner dependency queue for the complete non-PAMF comparison suite."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .data import read, write
from .plan import SEEDS, VIEWS, make_plan, validate_plan


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(tmp, path)


def environment(rt, gpu='', legacy=False):
    paths = [rt['release_root'], str(Path(rt['release_root']) / 'code')]
    if not legacy: paths.append(rt['numpy_deps'])
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def process_alive(pid, fragment=None):
    path = Path('/proc') / str(pid)
    try:
        status = (path / 'status').read_text()
        if re.search(r'^State:\s+Z', status, flags=re.M): return False
        command = (path / 'cmdline').read_bytes().decode().replace('\0', ' ')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return bool(command) and (fragment is None or fragment in command)


def parent_state(rt):
    state = read(Path(rt['parent_run_root']) / 'pipeline_state.json')
    alive = process_alive(rt['parent_pid'], rt['parent_release_root'])
    if alive: return 'LIVE', state
    if state['status'] == 'ANALYZED' and len(state['rows']) == 24 and all(
        r['status'] == 'ANALYZED' for r in state['rows'].values()):
        return 'DONE', state
    return 'NEEDS_RECOVERY', state


def source_config(rt, row, handles):
    h = handles[row['protocol_id']]
    meta = read(Path(h['source_cache']) / 'metadata.json')
    return dict(row, source_cache=h['source_cache'],
        source_contract_path=h['source_contract'], classes=meta['classes'],
        num_classes=len(meta['classes']), num_domains=len(meta['source_contract']['source_rxs']),
        ibrff_model_root=rt['ibrff_model_root'], sdg_model_root=rt['sdg_model_root'],
        output_root=str(Path(rt['run_root']) / row['row_id'] / 'source'))


def worker(c):
    rt = c['runtime']; row = c['row']; h = c['handles']
    root = Path(rt['run_root']) / row['row_id']; root.mkdir(parents=True, exist_ok=False)
    legacy = row['budget_kind'] == 'frozen_eval'
    if legacy:
        source_output = str(Path(rt['legacy_run_root']) /
            ('%s-s%d' % (row['family'], row['model_seed'])) / 'source')
    else:
        sc = source_config(rt, row, {row['protocol_id']: h})
        write(root / 'source_config.json', sc)
        subprocess.run([rt['python'], '-u', '-m', 'experiments.phase1_external_dg.source',
            '--config', str(root / 'source_config.json')],
            env=environment(rt, os.environ['CUDA_VISIBLE_DEVICES']), check=True)
        source_output = sc['output_root']
    tests = {name: dict(views_root=test['views_root'], kind=test['kind'])
             for name, test in h['tests'].items()}
    pc = dict(row_id=row['row_id'], model_seed=row['model_seed'],
        source_output=source_output, expected_source_contract=h['source_contract'],
        output_root=str(root / 'prediction'), tests=tests)
    write(root / 'prediction_config.json', pc)
    inference_python = rt['legacy_python'] if legacy else rt['python']
    subprocess.run([inference_python, '-u', '-m', 'experiments.phase1_external_dg.evaluation',
        'predict', '--config', str(root / 'prediction_config.json')],
        env=environment(rt, os.environ['CUDA_VISIBLE_DEVICES'], legacy=legacy), check=True)
    classes = read(Path(h['source_cache']) / 'metadata.json')['classes']
    sc = dict(row_id=row['row_id'], model_seed=row['model_seed'], classes=classes,
        prediction_output=pc['output_root'], output_root=str(root / 'scores'), tests=h['tests'])
    write(root / 'score_config.json', sc)
    subprocess.run([rt['python'], '-u', '-m', 'experiments.phase1_external_dg.evaluation',
        'score', '--config', str(root / 'score_config.json')], env=environment(rt), check=True)
    write(root / 'complete.json', dict(status='ANALYZED', row_id=row['row_id'],
        model_seed=row['model_seed'], trained=not legacy, tests=list(tests),
        views=list(row.get('views', VIEWS)), default_head=True))
    print('ROW_ANALYZED', row['row_id'], flush=True)


def smoke(c):
    """Real L_s gradients and strict checkpoint roundtrips; never gets target paths."""
    import torch
    import torch.nn.functional as F
    from . import source
    from .models import FishrVariance, source_coral, balanced_indices
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    results = []
    for row in c['rows']:
        meta = source.check_training(row)
        source.configure_native(row, meta); source.native.seed(row['model_seed'])
        net = source.build(row)
        batch = next(iter(source.native.array_loader(row, 'L_s', 12)))
        optimizer = torch.optim.Adam(net.parameters(), lr=row['lr'])
        objective = 'default_identity_CE'
        if row['family'] == 'riei' and row['variant'] == 'full':
            optimizer = torch.optim.SGD(list(net.ec.parameters()) + list(net.rc.parameters()), lr=row['lr'])
            fed = torch.optim.SGD(net.fed.parameters(), lr=row['lr'])
            source.native.ordinary_step(net, row, batch, optimizer, fed)
            objective = 'registered_RIEI_alternation'
        elif row['family'] == 'drift' and row['variant'] == 'full':
            source.native.ordinary_step(net, row, batch, optimizer)
            objective = 'registered_DRIFT_sum_L2'
        else:
            x, y, rx = batch
            if row['family'] == 'resnet18' and row['route'] in ('fishr', 'coral', 'mixstyle', 'mixstyle_id'):
                loader = source.native.array_loader(row, 'L_s')
                all_x, labels, receivers = loader.dataset.tensors
                ix = balanced_indices(labels, receivers, row['num_classes'], row['num_domains'],
                    150, torch.Generator().manual_seed(row['model_seed']))
                x, y, rx = all_x[ix], labels[ix], receivers[ix]
                net.context = (y, rx)
            if row['family'] == 'ibrff' and row['variant'] == 'full':
                from utils.hsic import hsic_normalized_cca
                z, logits = net(x); loss = F.cross_entropy(logits, y) + .04 * hsic_normalized_cca(z.flatten(1), x.flatten(1))
                objective = 'CE_plus_author_HSIC'
            elif row['family'] == 'resnet18':
                z, logits = net(x); loss = F.cross_entropy(logits, y)
                if row['route'] == 'fishr':
                    loss = loss + 1000 * FishrVariance(row['num_domains'])(z, logits, y, rx)
                    objective = 'active_Fishr_default_head_gradient'
                elif row['route'] == 'coral':
                    loss = loss + source_coral(z, rx, row['num_domains']); objective = 'active_source_CORAL'
            else:
                loss = F.cross_entropy(source.forward(net, row, x), y)
            optimizer.zero_grad(); loss.backward(); source.native.finite(net, loss); optimizer.step()
            if row['family'] == 'resnet18': net.context = None
        x = batch[0]; net.eval()
        path = root / (row['row_id'] + '.pt')
        source.native.save(path, net, row, meta['source_contract'], 0, 0, 0.)
        payload = source.native.load(path, row, meta['source_contract'])
        copy = source.build(row); copy.load_state_dict(payload['state_dict'], strict=True); copy.eval()
        with torch.no_grad():
            a, b = source.forward(net, row, x), source.forward(copy, row, x)
        if not torch.equal(a, b) or not torch.isfinite(a).all():
            raise ValueError('Real L_s default-head checkpoint roundtrip differs')
        results.append(dict(row_id=row['row_id'], status='PASS', source_samples=len(x),
            classes=row['num_classes'], source_RX=row['num_domains'], objective=objective,
            query_access=False, used_as_training_initialization=False))
        del net, copy, optimizer, payload
        gc.collect()
    write(root / 'result.json', dict(status='PASS', rows=results, query_access=False,
        note='SDG checkpoint smoke covers its identity path; original full augmentation/objectives are retained in the training function.'))


def legacy_smoke(c):
    import torch
    from .evaluation import legacy_model
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); results = []
    for row in c['rows']:
        net, meta = legacy_model(row, read(row['expected_source_contract']), torch.device('cpu'))
        results.append(dict(row_id=row['row_id'], status='PASS', strict_scratch_provenance=True,
            original_joint_head=True, query_access=False))
        del net; gc.collect()
    write(root / 'result.json', dict(status='PASS', rows=results, query_access=False))


def smoke_configs(rt, plan, handles):
    selected = []; seen = set(); old = []
    for row in plan['rows']:
        if row['budget_kind'] == 'frozen_eval':
            old.append(dict(row_id=row['row_id'], model_seed=row['model_seed'],
                source_output=str(Path(rt['legacy_run_root']) /
                    ('%s-s%d' % (row['family'], row['model_seed'])) / 'source'),
                expected_source_contract=handles['F0']['source_contract']))
            continue
        if row['model_seed'] != SEEDS[0]: continue
        c = source_config(rt, row, handles)
        key = (c['family'], c['variant'], c.get('route'), c['num_classes'], c['num_domains'])
        if key in seen: continue
        seen.add(key); selected.append(c)
    return (dict(rows=selected, output_root=str(Path(rt['run_root']) / 'smoke')),
            dict(rows=old, output_root=str(Path(rt['run_root']) / 'legacy_smoke')))


def run_stage(rt, state, name, command, *, legacy=False):
    logfile = Path(rt['log_root']) / (name + '.log')
    with logfile.open('x') as log:
        child = subprocess.Popen(command, cwd=rt['release_root'], env=environment(rt, legacy=legacy),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state.update(status=name.upper(), stage_pid=child.pid, stage_command=command, stage_log=str(logfile))
        atomic(Path(rt['run_root']) / 'pipeline_state.json', state)
        code = child.wait()
    state.pop('stage_pid', None)
    if code:
        state.update(status='FAILED_STAGE', failed_stage=name, returncode=code)
        atomic(Path(rt['run_root']) / 'pipeline_state.json', state)
        raise RuntimeError('Technical stage failure: ' + name)


def failure_fingerprint(path, row_id):
    lines = Path(path).read_text(errors='replace').splitlines()
    for line in reversed(lines):
        if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):', line):
            return line.replace(row_id, '<row>')[:500]
    return 'worker_exit_without_completion'


def dispatch(rt):
    plan = validate_plan(read(rt['matrix']))
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    Path(rt['log_root']).mkdir(parents=True, exist_ok=False)
    state = dict(status='STARTING', run_id=rt['run_id'], pid=os.getpid(),
        cwd=os.getcwd(), argv=sys.argv, allowed_gpus=[0, 1], rows={},
        expected_new_training=162, expected_frozen_eval=6)
    write(root / 'dispatcher.json', state)
    run_stage(rt, state, 'prepare_data', [rt['python'], '-u', '-m',
        'experiments.phase1_external_dg.driver', 'prepare', '--config', rt['runtime_file']])
    handles = read(root / 'data_handles.json')
    own, old = smoke_configs(rt, plan, handles)
    write(root / 'smoke_config.json', own); write(root / 'legacy_smoke_config.json', old)
    run_stage(rt, state, 'smoke', [rt['python'], '-u', '-m',
        'experiments.phase1_external_dg.driver', 'smoke', '--config', str(root / 'smoke_config.json')])
    run_stage(rt, state, 'legacy_smoke', [rt['legacy_python'], '-u', '-m',
        'experiments.phase1_external_dg.driver', 'legacy-smoke', '--config', str(root / 'legacy_smoke_config.json')], legacy=True)
    while True:
        verdict, parent = parent_state(rt)
        if verdict == 'DONE': break
        state.update(status='WAITING_PARENT' if verdict == 'LIVE' else 'PARENT_NEEDS_RECOVERY',
            parent_pid=rt['parent_pid'], parent_process_alive=verdict == 'LIVE',
            parent_status=parent['status'], parent_analyzed=sum(x['status'] == 'ANALYZED' for x in parent['rows'].values()))
        atomic(root / 'pipeline_state.json', state)
        if verdict == 'NEEDS_RECOVERY': return
        time.sleep(10)
    pending = list(plan['rows']); active = {}; failures = Counter()
    state['status'] = 'RUNNING'; state['parent_complete'] = True
    while pending or active:
        for gpu, item in list(active.items()):
            code = item['process'].poll()
            if code is None: continue
            rid = item['row']['row_id']; result = root / rid / 'complete.json'
            row_state = state['rows'][rid]
            row_state.update(returncode=code, status='ANALYZED' if code == 0 and result.is_file() else 'FAILED')
            item['log'].close()
            if row_state['status'] == 'FAILED':
                fingerprint = failure_fingerprint(row_state['log'], rid)
                row_state['failure'] = fingerprint; failures[fingerprint] += 1
            del active[gpu]
        if pending and failures and max(failures.values()) >= 2:
            state['unstarted_rows'] = [x['row_id'] for x in pending]; pending.clear()
            state['stop_reason'] = 'two_matching_technical_failures'
        for gpu in (0, 1):
            if gpu in active or not pending: continue
            current = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if current: continue
            row = pending.pop(0); rid = row['row_id']
            configuration = dict(runtime=rt, row=row, handles=handles[row['protocol_id']])
            path = root / (rid + '.json'); write(path, configuration)
            logfile = Path(rt['log_root']) / (rid + '.log'); log = logfile.open('x')
            command = [rt['python'], '-u', '-m', 'experiments.phase1_external_dg.driver',
                'worker', '--config', str(path)]
            child = subprocess.Popen(command, cwd=rt['release_root'], env=environment(rt, gpu),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = dict(process=child, row=row, log=log)
            state['rows'][rid] = dict(status='RUNNING', pid=child.pid, gpu=gpu,
                argv=command, cwd=rt['release_root'], log=str(logfile))
        atomic(root / 'pipeline_state.json', state)
        if pending or active: time.sleep(10)
    state['status'] = 'PARTIAL' if failures or state.get('unstarted_rows') else 'ANALYZED'
    atomic(root / 'pipeline_state.json', state)
    run_stage(rt, state, 'analysis', [rt['python'], '-u', '-m',
        'experiments.phase1_external_dg.analysis', '--config', rt['runtime_file']])
    # analysis updates completion from actual row artifacts, not this stage label.


def prepare(rt):
    from .data import prepare as build_data
    build_data(rt, read(rt['matrix']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'worker', 'prepare', 'smoke', 'legacy-smoke'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    globals()[args.action.replace('-', '_')](read(args.config))
