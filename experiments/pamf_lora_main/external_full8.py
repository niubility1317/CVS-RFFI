"""Train and blind-predict the four clean-only, full-preamble LoRa comparisons."""
import argparse
from inspect import signature
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from .clean_aperture import clean_iq
from .data import CLASSES, SOURCE, SPLIT_SEED, TARGET
from .external_full8_models import (CHIRPS, EDGE_GAIN, SAMPLES, PacketEdges, SDGPacketViews, build, identity,
                                    optimizers, train_batch)
from .full_packet_data import LIMIT
from .full_packet_views import native, source_contract, target_contract


RUN = '20261125-phase1-lora-external-full8-m4-r01'
RECOVERY_RUN = '20261125-phase1-lora-external-full8-clean-predict-m4-r02'
TRAIN_COMMIT = 'ef6f890b9aeeeffa2d223158e2b5a9c38ef98bf6'
SEED = 2026092702
BUDGETS = {'riei': (200, 64, .0001, 1), 'drift': (200, 256, .0001, 1),
           'ibrff': (100, 128, .001, 2), 'sdg': (400, 32, .001, 1)}
METHODS = tuple(BUDGETS)
SOURCE_FIELDS = ('run_id', 'run_root', 'code_commit', 'source_root', 'clean_cache',
                 'frozen_sources', 'external_releases', 'ibrff_model_root', 'sdg_model_root')
VALIDATION_BATCH = 64


def source_runtime(rt):
    if rt['run_id'] != RUN or not set(rt) <= set(SOURCE_FIELDS) | {'family'}:
        raise ValueError('External source worker refuses target and inherited model handles')
    source = Path(rt['source_root'])
    contract = source_contract(source)
    cache = Path(rt['clean_cache'])
    if read(cache/'complete.json') != dict(status='CLEAN_SOURCE_READY', samples=SAMPLES,
                                          source_protocol=contract['protocol'], chirps=list(CHIRPS)):
        raise ValueError('Registered all-chirp clean source cache required')
    return source, contract, cache


def recovery_training_runtime(rt):
    root = Path(rt['run_root'])
    if rt['run_id'] != RECOVERY_RUN or root.name != RECOVERY_RUN or root.parent.name != 'runs':
        raise ValueError('Wrong immutable prediction recovery root')
    project = root.parent.parent
    old_root = root.with_name(RUN)
    previous = read(project/'releases'/f'{RUN}-release'/'runtime.json')
    if (previous['run_id'] != RUN or previous['code_commit'] != TRAIN_COMMIT
            or Path(previous['run_root']) != old_root):
        raise ValueError('Unexpected external Full8 training provenance')
    for field in (*SOURCE_FIELDS[3:], 'native_target_root'):
        if previous[field] != rt[field]:
            raise ValueError(f'Prediction recovery data or model source differs: {field}')
    return {k: previous[k] for k in SOURCE_FIELDS}


def freeze_record(folder):
    try:
        record = (folder/'complete.json').read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    # The original writer appends its newline after closing the JSON object.
    return json.loads(record) if record.endswith('\n') else None


def frozen_checkpoint(folder, c, family, frozen):
    path = folder/f"stage{c['stages']}.pt"
    if (not frozen or frozen['status'] != 'SOURCE_FROZEN' or frozen['method'] != family
            or frozen['seed'] != SEED or frozen['checkpoint'] != path.name
            or frozen['stage_epochs'] != {str(k): c['epochs'] for k in range(1, c['stages'] + 1)}
            or any(frozen[k] for k in ('target_reads', 'satellite_batches', 'U_truth_reads'))):
        raise ValueError('Source checkpoint must be frozen before clean prediction')
    return path


def configuration(rt, family):
    epochs, batch, lr, stages = BUDGETS[family]
    return dict(run_id=RUN, family=family, code_commit=rt['code_commit'], model_seed=SEED,
                split_seed=SPLIT_SEED, epochs=epochs, batch_size=batch, lr=lr, stages=stages,
                num_classes=len(CLASSES), num_domains=len(SOURCE), initialization='scratch',
                samples_per_chirp=SAMPLES, chirps=list(CHIRPS), edge_gain=EDGE_GAIN,
                ibrff_model_root=rt['ibrff_model_root'], sdg_model_root=rt['sdg_model_root'],
                U_s_use=False, satellite_train_augmentation=False)


def role_dataset(source, contract, cache, role, take=None):
    if role not in ('L_s', 'V'):
        raise ValueError('External Full8 consumes only authorized labeled source roles')
    count = contract['counts'][role]
    x = np.load(cache/f'{role}.npy', mmap_mode='r', allow_pickle=False)
    with np.load(source/f'{role}_metadata.npz', allow_pickle=False) as m:
        if m['ids'].tolist() != contract['role_ids'][role] or x.shape != (count, 8, 2, SAMPLES):
            raise ValueError(f'Unmatched physical packet role: {role}')
        if x.dtype != np.float32 or m['y'].shape != (count,) or m['rx_i'].shape != (count,):
            raise ValueError(f'Unmatched physical packet schema: {role}')
        return torch.utils.data.TensorDataset(torch.from_numpy(x[:take].copy()),
                                              torch.tensor(m['y'][:take], dtype=torch.long),
                                              torch.tensor(m['rx_i'][:take], dtype=torch.long))


def native_tools(c, device):
    if c['family'] == 'sdg':
        from experiments.phase1_dg_nativebudget.pipeline import author_sdg
        author = author_sdg(c)
        return author, author.SupConLoss().to(device), None
    if c['family'] == 'ibrff':
        from experiments.practical_full_2x2.author_imports import author_hsic
        return None, None, author_hsic(c)
    return None, None, None


def checkpoint(path, net, c, contract, stage, epoch, validation):
    torch.save(dict(state_dict=net.state_dict(), config=c, source_contract=contract, stage=stage,
                    epoch=epoch, validation=validation, satellite_batches=0, U_truth_reads=0,
                    target_reads=0, scratch=True, default_head_joint_training=True), path)


def restore(path, c, contract, stage, device):
    options = {'weights_only': True} if 'weights_only' in signature(torch.load).parameters else {}
    payload = torch.load(path, map_location=device, **options)
    if (payload['config'] != c or payload['source_contract'] != contract or payload['stage'] != stage
            or not payload['scratch'] or not payload['default_head_joint_training']
            or any(payload[k] for k in ('satellite_batches', 'target_reads', 'U_truth_reads'))):
        raise ValueError('External Full8 frozen checkpoint provenance differs')
    return payload


@torch.no_grad()
def validate(net, family, loader, device):
    net.eval()
    y_all, rx_all, scores = [], [], []
    for x, y, rx in loader:
        q = identity(net, family, x.to(device))
        if q.shape != (len(y), len(CLASSES)) or not torch.isfinite(q).all():
            raise FloatingPointError('Malformed external source logits')
        y_all.append(y)
        rx_all.append(rx)
        scores.append(q.cpu())
    y, rx, q = torch.cat(y_all), torch.cat(rx_all), torch.cat(scores)
    hits = q.argmax(1) == y
    return dict(n=len(y), correct=int(hits.sum()), accuracy=float(hits.float().mean()),
                nll=float(F.cross_entropy(q, y)),
                by_rx={name: float(hits[rx == i].float().mean()) for i, name in enumerate(SOURCE)},
                by_tx={str(tx): float(hits[y == i].float().mean()) for i, tx in enumerate(CLASSES)})


def smoke(rt, family=None, device='cpu', full_batch=False):
    from experiments.phase1_dg_nativebudget.pipeline import seed
    source, contract, cache = source_runtime(rt)
    take = max(v[1] for v in BUDGETS.values()) if full_batch else 128
    dataset = role_dataset(source, contract, cache, 'L_s', take=take)
    x, y, rx = dataset.tensors
    records = {}
    for name in (family,) if family else METHODS:
        c = configuration(rt, name)
        order = list(range(c['batch_size'])) if full_batch else list(range(16))
        seed(SEED)
        net = build(c).to(device)
        author, contrast, hsic = native_tools(c, device)
        if author:
            views = SDGPacketViews(dataset, author, SEED)
            xx = torch.stack([views[i][0] for i in order])
        else:
            xx = x[order]
        opt, fed = optimizers(net, c)
        projection = [(m.project[1].weight.detach().clone(), m.project[1].weight)
                      for m in net.modules() if isinstance(m, PacketEdges)]
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        terms = train_batch(net, c, xx.to(device), y[order].to(device), rx[order].to(device),
                            opt, fed, c['stages'], contrast, hsic)
        peak = torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else None
        if not projection or any(torch.equal(before, after) for before, after in projection):
            raise ValueError(f'{name} did not train both its native objective and its packet-edge projection')
        with tempfile.TemporaryDirectory(dir=rt['run_root']) as scratch:
            path = Path(scratch)/'source.pt'
            checkpoint(path, net, c, contract, c['stages'], 1, None)
            payload = restore(path, c, contract, c['stages'], device)
            reader = build(c).to(device).eval()
            reader.load_state_dict(payload['state_dict'], strict=True)
            with torch.no_grad():
                q = identity(reader, name, x[order].to(device))
        if q.shape != (len(order), len(CLASSES)) or not torch.isfinite(q).all():
            raise ValueError(f'{name} physical source smoke failed')
        records[name] = dict(input=list(xx.shape), logits=list(q.shape), terms=terms,
                             parameters=sum(p.numel() for p in net.parameters()),
                             peak_gpu_bytes=peak)
        print('EXTERNAL_FULL8_SOURCE_SMOKE_PASS', name, records[name], flush=True)
    if family is None:
        write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', methods=records, target_reads=0))
    return records


def train(rt, family):
    from experiments.phase1_dg_nativebudget.pipeline import seed
    source, contract, cache = source_runtime(rt)
    c = configuration(rt, family)
    root = Path(rt['run_root'])/'F'/family
    root.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0'
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    labeled = role_dataset(source, contract, cache, 'L_s')
    valid = torch.utils.data.DataLoader(role_dataset(source, contract, cache, 'V'), batch_size=VALIDATION_BATCH)
    train_clean = torch.utils.data.DataLoader(labeled, batch_size=VALIDATION_BATCH)
    author, contrast, hsic = native_tools(c, device)
    write(root/'resolved_config.json', c)
    stage_epochs = {}
    for stage in range(1, c['stages'] + 1):
        seed(SEED)
        net = build(c).to(device)
        if stage == 2:
            parent = restore(root/'stage1.pt', c, contract, 1, device)
            net.load_state_dict(parent['state_dict'], strict=True)
            del parent
        opt, fed = optimizers(net, c)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.1, patience=5, min_lr=1e-6)
                     if family == 'ibrff' else None)
        train_data = SDGPacketViews(labeled, author, SEED) if author else labeled
        loader = torch.utils.data.DataLoader(train_data, batch_size=c['batch_size'], shuffle=True,
                                             generator=torch.Generator().manual_seed(SEED))
        best = None
        with (root/f'epochs-stage{stage}.jsonl').open('x', encoding='utf8') as log:
            for epoch in range(1, c['epochs'] + 1):
                terms = [train_batch(net, c, x.to(device), y.to(device), rx.to(device), opt, fed,
                                     stage, contrast, hsic) for x, y, rx in loader]
                val = validate(net, family, valid, device)
                if scheduler and epoch >= 25:
                    scheduler.step(val['nll'])
                rank = (-val['nll'],) if stage != c['stages'] else (val['correct'], -val['nll'])
                if best is None or rank > best:
                    checkpoint(root/f'stage{stage}.pt', net, c, contract, stage, epoch, val)
                    best = rank
                record = dict(stage=stage, epoch=epoch, V=val, lr=opt.param_groups[0]['lr'],
                              **{k: mean(t[k] for t in terms) for k in terms[0]})
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                print('EXTERNAL_FULL8_EPOCH', family, stage, epoch, val['correct'], val['nll'], flush=True)
        stage_epochs[str(stage)] = c['epochs']
    path = root/f"stage{c['stages']}.pt"
    chosen = restore(path, c, contract, c['stages'], device)
    net.load_state_dict(chosen['state_dict'], strict=True)
    train_result = validate(net, family, train_clean, device)
    write(root/'complete.json', dict(status='SOURCE_FROZEN', method=family, seed=SEED, checkpoint=path.name,
          frozen=dict(chosen['validation'], epoch=chosen['epoch'], stage=chosen['stage']), L=train_result,
          stage_epochs=stage_epochs, target_reads=0, U_truth_reads=0, satellite_batches=0,
          peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - started))


@torch.no_grad()
def predict(rt, family):
    recovering = rt['run_id'] == RECOVERY_RUN
    source_rt = (recovery_training_runtime(rt) if recovering
                 else {k: v for k, v in rt.items() if k != 'native_target_root'})
    source, contract, _ = source_runtime(source_rt)
    c = configuration(source_rt, family)
    root = Path(rt['run_root'])
    folder = Path(source_rt['run_root'])/'F'/family
    frozen = freeze_record(folder)
    checkpoint_path = frozen_checkpoint(folder, c, family, frozen)
    device = 'cuda:0'
    model = build(c).to(device).eval()
    chosen = restore(checkpoint_path, c, contract, c['stages'], device)
    if chosen['epoch'] != frozen['frozen']['epoch']:
        raise ValueError('Source checkpoint does not match the selected epoch')
    model.load_state_dict(chosen['state_dict'], strict=True)
    if recovering:
        raw = native(source, 'L_s', contract['counts']['L_s'])[:8]
        iq = clean_iq(raw, SAMPLES, CHIRPS)
        q = identity(model, family, torch.from_numpy(iq).to(device))
        if q.shape != (len(raw), len(CLASSES)) or not torch.isfinite(q).all():
            raise ValueError('Recovered external source checkpoint failed no-query smoke')
        write(root/'source_smoke'/f'{family}.json',
              dict(status='PASS', target_reads=0, source_run=RUN, code_commit=TRAIN_COMMIT,
                   checkpoint=str(checkpoint_path), epoch=chosen['epoch'], source_count=len(raw)))
    target = Path(rt['native_target_root'])
    target_contract(target)
    out = root/'predictions'/family
    out.mkdir(parents=True, exist_ok=False)
    for rx in TARGET:
        x = native(target, rx, LIMIT)
        pred = []
        for i in range(0, len(x), VALIDATION_BATCH):
            iq = clean_iq(x[i:i + VALIDATION_BATCH], SAMPLES, CHIRPS)
            logits = identity(model, family, torch.from_numpy(iq).to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'Nonfinite clean prediction for {family}/{rx}')
            pred.extend(logits.argmax(1).tolist())
        np.save(out/f'{rx}__clean__direct.npy', np.asarray(pred, dtype=np.int16), allow_pickle=False)
        print('EXTERNAL_FULL8_CLEAN', family, rx, flush=True)
    write(out/'complete.json', dict(status='PREDICTIONS_FROZEN', method=family, seed=SEED,
          views=['clean'], receivers=list(TARGET), per_receiver=LIMIT, target_truth_read=False))


def launch(rt, kind, gpu=None, family=None):
    root = Path(rt['run_root'])
    config = {k: rt[k] for k in SOURCE_FIELDS}
    if family:
        config['family'] = family
    if kind == 'predict':
        config['native_target_root'] = rt['native_target_root']
    tag = f'{kind}-{family or "all"}'
    cfg = root/'configs'/f'{tag}.json'
    write(cfg, config)
    path = root/'logs'/f'{tag}.log'
    path.parent.mkdir(exist_ok=True)
    cmd = [rt['python'], '-u', '-m', 'experiments.pamf_lora_main.external_full8', kind, str(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with path.open('x', encoding='utf8') as log:
        child = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
    print('EXTERNAL_FULL8_LAUNCHED', kind, family, child.pid, gpu, cfg, flush=True)
    return child


def dispatch(rt):
    from .runner import available
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    try:
        if rt['run_id'] == RECOVERY_RUN:
            return dispatch_recovery(rt, state)
        write(state, dict(status='SOURCE_SMOKE'))
        if launch(rt, 'smoke').wait():
            raise RuntimeError('External Full8 source smoke failed')
        write(state, dict(status='RUNNING', methods=list(METHODS)), exclusive=False)
        # The native dual-view SDG batch has the highest measured source memory.
        pending = [('train', name) for name in ('sdg', 'riei', 'drift', 'ibrff')]
        active = {}
        while pending or active:
            for gpu, (child, kind, family) in list(active.items()):
                done = child.poll()
                if done is None:
                    continue
                if done:
                    raise RuntimeError(f'{family} {kind} failed ({done})')
                if kind == 'train':
                    pending.insert(0, ('predict', family))
                del active[gpu]
            if pending:
                candidates = [gpu for gpu in rt['allowed_gpus'] if gpu not in active]
                free = available(candidates, rt['max_compute_processes'], rt['min_free_mib'])
                sdg_free = (available(free, rt['max_compute_processes'], rt['sdg_min_free_mib'])
                            if ('train', 'sdg') in pending else ())
                for gpu in free:
                    job = next(((kind, name) for kind, name in pending
                                if (kind, name) != ('train', 'sdg') or gpu in sdg_free), None)
                    if job is not None:
                        kind, family = job
                        pending.remove(job)
                        active[gpu] = launch(rt, kind, gpu, family), kind, family
            if pending or active:
                time.sleep(15)
        write(state, dict(status='PREDICTIONS_FROZEN', names=list(METHODS), target_truth_read=False), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def dispatch_recovery(rt, state):
    from .runner import available
    source_rt = recovery_training_runtime(rt)
    source_runtime(source_rt)
    old_root = Path(source_rt['run_root'])
    if read(old_root/'state.json')['status'] != 'FAILED' or any((old_root/'predictions').glob('*/complete.json')):
        raise ValueError('Recovery requires the failed original controller and no completed predictions')
    write(state, dict(status='WAITING_SOURCE_FREEZE', source_run=RUN, methods=list(METHODS)))
    pending = list(METHODS)
    active = {}
    while pending or active:
        for gpu, (child, family) in list(active.items()):
            done = child.poll()
            if done is None:
                continue
            if done:
                raise RuntimeError(f'{family} recovery prediction failed ({done})')
            del active[gpu]
        ready = []
        for family in pending:
            folder = old_root/'F'/family
            frozen = freeze_record(folder)
            if frozen is not None:
                frozen_checkpoint(folder, configuration(source_rt, family), family, frozen)
                ready.append(family)
        if ready:
            candidates = [gpu for gpu in rt['allowed_gpus'] if gpu not in active]
            for gpu, family in zip(available(candidates, rt['max_compute_processes'], rt['min_free_mib']), ready):
                active[gpu] = launch(rt, 'predict', gpu, family), family
                pending.remove(family)
        if pending or active:
            time.sleep(15)
    write(state, dict(status='PREDICTIONS_FROZEN', names=list(METHODS), source_run=RUN,
                      target_truth_read=False), exclusive=False)


def worker(rt, kind):
    from experiments.practical_full_2x2.runner import activate
    activate(rt)
    if kind == 'smoke':
        return smoke(rt)
    return {'train': train, 'predict': predict}[kind](rt, rt['family'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('dispatch', 'smoke', 'train', 'predict'))
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    runtime = read(args.runtime)
    if args.kind == 'dispatch':
        dispatch(runtime)
    else:
        worker(runtime, args.kind)
