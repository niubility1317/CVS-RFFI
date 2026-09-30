"""Explicit ISSL repair variant; pinned author code and original acceptance stay intact.

Preserves the author's ResNet, logits contrastive space, loss weights, temperatures,
data repetition, crop ratio, optimizers and stage ordering. No query truth is used.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from acceptance import Logger, batch_tensor, batches, grad_norm, predict, write_json
from runtime import verify_source, load_module, expand_head, load_diagnostic_input, data_contract_path


def distillation_loss(student, teacher, temperature=2):
    """Author temperature CE value, with student graph and detached teacher."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite KD temperature required')
    if student.ndim != 2 or student.shape != teacher.shape:
        raise ValueError('KD requires matching N,C logits')
    return -(F.softmax(teacher.detach()/temperature, dim=1) *
             F.log_softmax(student/temperature, dim=1)).sum(dim=1).mean()


def contrastive_loss(q, k, queue, temperature=5):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite contrastive temperature required')
    if q.ndim != 2 or q.shape != k.shape or queue.ndim != 2 or queue.shape[1] != q.shape[1] or not len(queue):
        raise ValueError('Contrastive shape or empty queue')
    q = F.normalize(q, dim=1, eps=1e-12)
    k = F.normalize(k.detach(), dim=1, eps=1e-12)
    queue = F.normalize(queue.detach(), dim=1, eps=1e-12)
    positive = (q*k).sum(dim=1, keepdim=True)
    negatives = q @ queue.T
    target = torch.zeros(len(q), dtype=torch.long, device=q.device)
    return F.cross_entropy(torch.cat((positive, negatives), dim=1)/temperature, target)


def freeze_teacher(model):
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters(): parameter.grad = None
    return model


@torch.no_grad()
def momentum_update(key, student, momentum=.99):
    if not 0 <= momentum < 1: raise ValueError('Momentum must be in [0,1)')
    for key_parameter, query_parameter in zip(key.parameters(), student.parameters()):
        key_parameter.mul_(momentum).add_(query_parameter, alpha=1-momentum)
    # Keep eval-mode key BN statistics consistent with the EMA encoder.
    for key_buffer, query_buffer in zip(key.buffers(), student.buffers()):
        if key_buffer.is_floating_point(): key_buffer.mul_(momentum).add_(query_buffer, alpha=1-momentum)
        else: key_buffer.copy_(query_buffer)
    key.eval()


def append_queue(queue, keys, capacity=1000):
    if capacity < 1: raise ValueError('Positive queue capacity required')
    return F.normalize(torch.cat((queue.detach(), keys.detach()), dim=0)[-capacity:], dim=1, eps=1e-12)


def ssl_step(model, key, optimizer, qx, kx, queue, old_classes=3, capacity=1000,
             temperature_kd=20, temperature_contrastive=5, kd_weight=.5, momentum=.99):
    model.train()
    key.eval()
    optimizer.zero_grad(set_to_none=True)
    q, _ = model(qx)
    with torch.no_grad():
        k, _ = key(kx)
        teacher_logits, _ = key(qx)
    old_loss = distillation_loss(q[:, :old_classes], teacher_logits[:, :old_classes], temperature_kd)
    new_loss = contrastive_loss(q, k, queue, temperature_contrastive)
    kd_gradient = torch.autograd.grad(old_loss, q, retain_graph=True)[0]
    loss = kd_weight*old_loss + (1-kd_weight)*new_loss
    loss.backward()
    gradient = grad_norm(model)
    new_head_gradient = float(model.fc.weight.grad[old_classes:].norm())
    if not torch.isfinite(loss) or not math.isfinite(gradient): raise FloatingPointError('ssl')
    optimizer.step()
    queue = append_queue(queue, k, capacity)
    momentum_update(key, model, momentum)
    stats = dict(loss=float(loss.detach()), kd=float(old_loss.detach()), contrastive=float(new_loss.detach()),
                 kd_weight=kd_weight, contrastive_weight=1-kd_weight, grad_norm=gradient,
                 kd_logit_grad_norm=float(kd_gradient.norm()), new_head_grad_norm=new_head_gradient,
                 kd_student_gradient=True, zero_grad=True, teacher_frozen=True,
                 negative_queue_layout='transpose', queue_normalized=True, queue_size=len(queue),
                 momentum=momentum, temperature_contrastive=temperature_contrastive,
                 temperature_kd=temperature_kd, lr=optimizer.param_groups[0]['lr'],
                 source_validation=None)
    return queue, stats


def training_accuracy(model, x, y, device):
    # Only fitted training/support labels; never query or validation truth.
    predictions = predict(model, x, device)
    result = {'train_accuracy': float(np.mean(predictions == y))}
    for label in np.unique(y): result[f'train_accuracy_class_{label}'] = float(np.mean(predictions[y == label] == label))
    for name, mask in [('old', y < 3), ('new', y >= 3)]:
        result[f'train_accuracy_{name}'] = float(np.mean(predictions[mask] == y[mask])) if mask.any() else None
    return result


def train_supervised(model, x, y, epochs, lr, stage, device, log, teacher=None, crop=None):
    if teacher is not None: freeze_teacher(teacher)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, .5)
    for epoch in range(epochs):
        start = time.perf_counter()
        records = []
        for step, indices in enumerate(batches(len(x), 128)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = batch_tensor(x[indices], device, crop=crop)
            target = torch.tensor(y[indices], device=device)
            logits, _ = model(batch)
            ce = F.cross_entropy(logits, target)
            kd = None
            kd_gradient = None
            if teacher is not None:
                with torch.no_grad(): teacher_logits, _ = teacher(batch)
                kd = distillation_loss(logits[:, :3], teacher_logits, 2)
                kd_gradient = float(torch.autograd.grad(kd, logits, retain_graph=True)[0].norm())
            loss = ce if kd is None else .5*ce + .5*kd
            loss.backward()
            gradient = grad_norm(model)
            if not torch.isfinite(loss) or not math.isfinite(gradient): raise FloatingPointError(stage)
            new_head_gradient = float(model.fc.weight.grad[3:].norm()) if model.fc.out_features > 3 else None
            optimizer.step()
            record = dict(loss=float(loss.detach()), ce=float(ce.detach()), kd=None if kd is None else float(kd.detach()),
                          ce_weight=1. if kd is None else .5, kd_weight=None if kd is None else .5,
                          kd_logit_grad_norm=kd_gradient, new_head_grad_norm=new_head_gradient,
                          grad_norm=gradient, lr=optimizer.param_groups[0]['lr'], zero_grad=True,
                          teacher_frozen=None if teacher is None else True,
                          source_validation=None, validation_note='Training labels only; no query feedback')
            records.append(record)
            log.log(stage=stage, epoch=epoch, step=step, **record)
        if not records: raise ValueError('No supervised steps')
        accuracy = training_accuracy(model, x, y, device)
        log.log(stage=stage, epoch=epoch, summary=True, loss=float(np.mean([r['loss'] for r in records])),
                ce=float(np.mean([r['ce'] for r in records])),
                kd=None if teacher is None else float(np.mean([r['kd'] for r in records])),
                grad_norm=float(np.mean([r['grad_norm'] for r in records])),
                lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-start, **accuracy)
        scheduler.step()


def validate_config(config):
    if config.get('implementation') != 'issl_fixed_v1': raise ValueError('Expected explicit repair variant')
    for key in ('base_epochs', 'ssl_epochs', 'downstream_epochs', 'incremental_epochs'):
        if type(config.get(key)) is not int or config[key] < 1: raise ValueError(f'Positive integer {key} required')
    for key in ('base_lr', 'ssl_lr', 'downstream_lr', 'incremental_lr'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0: raise ValueError(f'Invalid {key}')
    if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Integer seed in [0,2**32) required')
    for key in ('data', 'output', 'issl_source', 'run_id'):
        if not config.get(key): raise ValueError(f'Missing {key}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    validate_config(config)
    source = Path(config['issl_source'])
    verify_source(source, 'ISSL')
    x, y, query, query_ids = load_diagnostic_input(config['data'])
    contract = data_contract_path(config['data'])
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    resolved = config | dict(device=str(device), torch=torch.__version__, pid=os.getpid(), cwd=str(Path.cwd()),
                             gpu_name=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                             checkpoint='scratch', data_contract_ref=str(contract), train_samples=len(x),
                             query_samples=len(query), old_classes=3, new_classes=3, crop=218,
                             claim_scope='ISSL repaired WiSig diagnostic; no original-paper fidelity claim')
    write_json(out/'resolved_config.json', resolved)
    print(json.dumps(resolved), flush=True)
    log = Logger(out)
    start = time.perf_counter()
    module = load_module(source/'resnet.py', 'issl_fixed_pinned_resnet')
    model = module.resnet18(num_classes=3, drop_out=.1).to(device)
    old_mask = y < 3
    train_supervised(model, x[old_mask], y[old_mask], config['base_epochs'], config['base_lr'], 'base', device, log)
    base = copy.deepcopy(model)
    torch.save(dict(state_dict=base.state_dict(), initialization='scratch', implementation='issl_fixed_v1',
                    training_ids_ref=str(contract)), out/'base.pt')
    # Real freshly generated checkpoint, training IQ only; isolate verification RNG.
    with torch.random.fork_rng(devices=[0] if device.type == 'cuda' else []):
        smoke = module.resnet18(num_classes=3, drop_out=.1).to(device)
        smoke.load_state_dict(torch.load(out/'base.pt', weights_only=True, map_location=device)['state_dict'], strict=True)
        smoke.eval(); base.eval()
        with torch.no_grad():
            inputs = batch_tensor(x[old_mask][:2], device)
            smoke_logits, _ = smoke(inputs)
            reference_logits, _ = base(inputs)
        if not torch.equal(smoke_logits, reference_logits): raise AssertionError('Training-only checkpoint smoke mismatch')
    write_json(out/'checkpoint_smoke.json', dict(status='VERIFIED', checkpoint='base.pt', query_access=False,
                                               training_samples=2, strict_reload=True, logits_identical=True))
    del smoke
    a = predict(base, query, device)
    model, _ = expand_head(copy.deepcopy(base), 3)
    key = freeze_teacher(copy.deepcopy(model))
    ssl_x = np.concatenate((x[old_mask], x))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['ssl_lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, .8)
    size = random.randint(1, min(1000, len(ssl_x)))
    indices = np.random.permutation(len(ssl_x))[:size]
    with torch.no_grad(): queue, _ = key(batch_tensor(ssl_x[indices], device, crop=218))
    queue = F.normalize(queue.detach(), dim=1, eps=1e-12)
    initial_head = model.fc.weight.detach().clone()
    for epoch in range(config['ssl_epochs']):
        epoch_start = time.perf_counter()
        records = []
        for step, indices in enumerate(batches(len(ssl_x), 64)):
            qx = batch_tensor(ssl_x[indices], device, crop=218)
            kx = batch_tensor(ssl_x[indices], device, crop=218)
            queue, stats = ssl_step(model, key, optimizer, qx, kx, queue)
            records.append(stats)
            log.log(stage='ssl', epoch=epoch, step=step, **stats)
        if not records: raise ValueError('No SSL steps')
        log.log(stage='ssl', epoch=epoch, summary=True, seconds=time.perf_counter()-epoch_start,
                **{field: float(np.mean([r[field] for r in records])) for field in ('loss', 'kd', 'contrastive', 'grad_norm', 'kd_logit_grad_norm', 'new_head_grad_norm')},
                lr=optimizer.param_groups[0]['lr'], queue_size=len(queue), source_validation=None)
        scheduler.step()
    ssl_new_head_updated = not torch.equal(model.fc.weight[3:], initial_head[3:])
    torch.save(dict(state_dict=model.state_dict(), parent='base.pt', stage='ssl', implementation='issl_fixed_v1'), out/'ssl.pt')
    train_supervised(model, x, y, config['downstream_epochs'], config['downstream_lr'], 'downstream_transfer', device, log, crop=218)
    c = predict(model, query, device)
    torch.save(dict(state_dict=model.state_dict(), parent='ssl.pt', stage='downstream', implementation='issl_fixed_v1'), out/'downstream.pt')
    baseline, teacher = expand_head(copy.deepcopy(base), 3)
    teacher_before = copy.deepcopy(teacher.state_dict())
    train_supervised(baseline, x, y, config['incremental_epochs'], config['incremental_lr'], 'incremental_baseline', device, log, teacher=teacher)
    teacher_state_unchanged = all(torch.equal(value, teacher_before[name]) for name, value in teacher.state_dict().items())
    if not teacher_state_unchanged: raise AssertionError('Frozen incremental teacher changed')
    cb = predict(baseline, query, device)
    torch.save(dict(state_dict=baseline.state_dict(), parent='base.pt', stage='incremental_baseline', implementation='issl_fixed_v1'), out/'incremental_baseline.pt')
    with (out/'predictions.npz').open('xb') as handle:
        np.savez_compressed(handle, ids=query_ids, A=a, C=c, C_incremental_baseline=cb)
    log.finish()
    epoch_rows = [row for row in log.rows if row.get('summary')]
    with (out/'epochs.jsonl').open('x', encoding='utf-8') as handle:
        for row in epoch_rows: handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    with (out/'epochs.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({field for row in epoch_rows for field in row}), lineterminator='\n')
        writer.writeheader(); writer.writerows(epoch_rows)
    acceptance = dict(status='VERIFIED', method='issl', implementation='issl_fixed_v1',
                      seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                      steps=sum(not row.get('summary', False) for row in log.rows), query_truth_used_for_training=False,
                      kd_student_gradient=True, ssl_new_head_updated=ssl_new_head_updated, teacher_frozen=True,
                      incremental_teacher_state_unchanged=teacher_state_unchanged,
                      old_only_adaptation_B=None, B_note='Original ISSL workflow has no old-only adaptation stage')
    write_json(out/'acceptance.json', acceptance)
    print(json.dumps(acceptance), flush=True)


if __name__ == '__main__': main()
