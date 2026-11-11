"""Frozen m1 source-L/V calibration and source-day branch certificate probe."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_paper_completion.model import load_reader
from experiments.pamf_physics_branch.geometry import bounded_echo
from experiments.pamf_physics_branch.objective import echo_parameters
from .recovery import branch_certificate

RUN = '20261111-phase1-pamf-single-rx-source-v-probe-r01'


def get_features(reader, x):
    core, sequence, reference = reader.core.id_backbone, reader.sequence_pool, reader.response_projection
    with capture_branches(core, sequence, reference) as features:
        fused = reader.features(x)
    return features, reader.classify_features(fused).softmax(1)


def evaluate(source, row, contract, cache, device):
    reader, cfg = load_reader(source, contract, row, device)
    if cfg['protocol_id'] != 'm1-natural' or cfg['source_rxs'] != [1] or cfg['source_days'] != [1, 2, 3]:
        raise ValueError('Only the declared single-RX source data contract is allowed')
    labeled, validation = (SourceDataset(cache, role, contract) for role in ('L_s', 'V'))
    anchor = {}
    with torch.no_grad():
        for start in range(0, len(labeled), 128):
            features, _ = get_features(reader, labeled.x[start:start + 128].to(device))
            for key, value in features.items():
                anchor.setdefault(key, []).append(value)
        anchor = {key: torch.cat(value) for key, value in anchor.items()}
        l_y = torch.tensor(labeled.metadata['y'].tolist(), device=device, dtype=torch.long)
        l_day = torch.tensor(labeled.metadata['day_i'].tolist(), device=device, dtype=torch.long)
        metrics = {}
        for start in range(0, len(validation), 128):
            end = min(start + 128, len(validation))
            x = validation.x[start:end].to(device)
            y = torch.tensor(validation.metadata['y'][start:end].tolist(), device=device, dtype=torch.long)
            day = torch.tensor(validation.metadata['day_i'][start:end].tolist(), device=device, dtype=torch.long)
            coeff, delay = echo_parameters(len(x), x.device, x.dtype, 202611200 + start)
            altered = bounded_echo(x, coeff, delay)
            raw, probs = get_features(reader, x)
            echo, _ = get_features(reader, altered)
            confidence, pseudo = probs.max(1)
            all_raw = {key: torch.cat((anchor[key], raw[key])) for key in anchor}
            all_echo = {key: torch.cat((anchor[key], echo[key])) for key in anchor}
            branch, stable = branch_certificate(all_raw, all_echo, l_y, l_day, day, pseudo,
                                                len(contract['classes']), 4)
            accepted = confidence >= .95
            for dimension, values in (('overall', torch.zeros_like(y)), ('day', day), ('transmitter', y)):
                for value in values.unique().tolist():
                    mask = values == value
                    key = (dimension, int(value))
                    result = metrics.setdefault(key, dict(count=0, correct=0, accepted=0, accepted_wrong=0,
                                                          branch_selected=0, branch_wrong=0,
                                                          stable_selected=0, stable_wrong=0))
                    wrong = pseudo != y
                    for name, selection in (('accepted', accepted), ('branch', accepted & branch),
                                            ('stable', accepted & stable)):
                        result['accepted' if name == 'accepted' else name + '_selected'] += int((mask & selection).sum())
                        result[name + '_wrong'] += int((mask & selection & wrong).sum())
                    result['count'] += int(mask.sum())
                    result['correct'] += int((mask & ~wrong).sum())
    if metrics['overall', 0]['count'] != contract['counts']['V']:
        raise ValueError('Full held-out source V required')
    return [dict(dimension=dim, stratum=stratum, **value) for (dim, stratum), value in sorted(metrics.items())]


def run(project, output, device):
    project, output = Path(project), Path(output)
    runs = project / 'runs'
    paper = read(runs / '20261027-phase1-pamf-paper-completion-m119-r03' / 'sources.json')
    new_run = runs / '20261111-phase1-pamf-tri-single-rx-day-m6-r01'
    new = read(new_run / 'sources.json')
    results = []
    for seed in (2026092701, 2026092702, 2026092703):
        tri_row = f'source_rx_1-factorial-ssl-full-permuted-s{seed}'
        pl_row = f'source_rx_1-factorial-pl-full-ordered-s{seed}'
        sources = {'PL': paper[pl_row]['source_output'], 'Tri': paper[tri_row]['source_output'],
                   'day': new[f'source_rx_1-tri-day-s{seed}']['source_output'],
                   'day_physics': new[f'source_rx_1-tri-day_physics-s{seed}']['source_output']}
        for name, source in sources.items():
            done = read(Path(source) / 'complete.json')
            cfg = done['config']
            contract = read(cfg['source_contract'])
            if done['target_access'] or done['U_truth_read'] or contract['source_rxs'] != [1]:
                raise ValueError('Frozen matching source-only model required')
            started = time.monotonic()
            result = evaluate(source, cfg['paper_row'], contract, cfg['source_cache'], device)
            results.append(dict(arm=name, seed=seed, source=source, results=result, seconds=time.monotonic() - started))
            print('SOURCE_V_PROBE', name, seed, flush=True)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'results.json', dict(status='SOURCE_V_PROBE_COMPLETE', source_L_V_only=True,
          U_reads=0, target_reads=0, optimizer_updates=0, confidence_threshold=.95, required_branch_votes=4,
          physical_proxy='bounded echo, diagnostic only', full_L_anchor_bank=True, results=results))


def dispatch(rt):
    from experiments.pamf_physics_branch.runner import admits
    root = Path(rt['run_root'])
    if rt['run_id'] != RUN:
        raise ValueError('Unique source-only diagnostic owner required')
    root.mkdir(parents=True, exist_ok=False)
    state = dict(status='WAITING_CAPACITY', pid=os.getpid(), source_L_V_only=True,
                 U_reads=0, target_reads=0)
    write(root / 'state.json', state)
    gpu = None
    while gpu is None:
        gpu = next((g for g in rt['allowed_gpus'] if admits(g, {}, 6000)), None)
        if gpu is None:
            time.sleep(15)
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [rt['python'], '-u', '-m', 'experiments.pamf_physics_tri.single_rx_source_probe', 'worker',
               '--project', rt['project'], '--output', str(root / 'result')]
    with (root / 'probe.log').open('x') as log:
        child = subprocess.Popen(command, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING', worker_pid=child.pid, gpu=gpu, command=command, cwd=rt['release_root'])
        write(root / 'state.json', state, exclusive=False)
        code = child.wait()
    result = root / 'result' / 'results.json'
    ok = code == 0 and result.exists() and read(result)['status'] == 'SOURCE_V_PROBE_COMPLETE'
    state.update(status='COMPLETE' if ok else 'FAILED', exit_code=code)
    write(root / 'state.json', state, exclusive=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--config')
    p.add_argument('--project')
    p.add_argument('--output')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.action == 'dispatch':
        dispatch(read(a.config))
    else:
        run(a.project, a.output, a.device)
