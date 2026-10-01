"""Scratch CE-only training; exact CVCNN source roles and optimization budget."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from baselines.common.practical_source import build_contract_split, PracticalResidualAugment
from baselines.common.cvs_data import make_cvs_loader


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')


def validate_config(c):
    if any(c.get(k) for k in ('checkpoint', 'resume', 'teacher', 'initial_checkpoint',
        'target_inputs', 'target_truth', 'p1_truth')):
        raise ValueError('Scratch source training rejects checkpoint and target inputs')
    required = {'method': 'cvs_identity_ce', 'epochs': 200, 'batch_size': 128,
        'lr': .0002, 'lr_min': .000001, 'weight_decay': .0001,
        'augmentation_seed': 2027, 'receiver_seed': 2027, 'split_seed': 392005,
        'drop_last': False, 'sat_ce_start': 80, 'lambda_sat_cls': .68,
        'domain_backbone': False, 'mixstyle': False, 'extra_losses': [],
        'selection': 'fixed_last_epoch', 'branch_ablation': 'no_dac'}
    for key, value in required.items():
        if c.get(key) != value:
            raise ValueError('Matched recipe mismatch: ' + key)
    return c


def source_args(c):
    return SimpleNamespace(source_contract=c['source_contract'], use_source_ssl_split=True,
        wisig_pkl=c['dataset'], output_dir=c['output_root'], wisig_train_rxs='1,3,4,6,8',
        wisig_train_days='1,2,3', wisig_test_rxs='0,2,5,7,9,10,11',
        wisig_labeled_ratio=.07, wisig_unlabeled_ratio=.63, wisig_source_val_ratio=.30,
        wisig_split_seed=392005, wisig_equalized=1, wisig_out_len=256,
        wisig_rms_normalize=True, wisig_domain='rx_day')


def loss_for_batch(model, x, y, batch, augment, epoch):
    if epoch >= 80:
        satellite = augment(x, metadata=batch['meta'])
        clean_logits, sat_logits = model(torch.cat((x, satellite))).split(len(x))
        ce = F.cross_entropy(clean_logits, y)
        sat = F.cross_entropy(sat_logits, y)
        return ce + .68 * sat, ce, sat
    ce = F.cross_entropy(model(x), y)
    return ce, ce, None


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    count = correct = 0
    total = 0.
    for batch in loader:
        y = batch['label'].to(device)
        scores = model(batch['iq'].to(device))
        total += float(F.cross_entropy(scores, y, reduction='sum'))
        correct += int((scores.argmax(1) == y).sum())
        count += len(y)
    return dict(source_val_count=count, source_val_accuracy=correct/count,
                source_val_ce=total/count)


def checkpoint_smoke(model, path, device):
    torch.save({'model': model.state_dict(), 'scratch_only': True}, path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    with torch.no_grad():
        scores = model(torch.zeros(2, 2, 256, device=device))
    if scores.shape != (2, 6) or not torch.isfinite(scores).all():
        raise ValueError('Initial real checkpoint no-query smoke failed')
    print('SMOKE PASS initial checkpoint roundtrip; query_read=false', flush=True)


def train(c):
    validate_config(c)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    seed = int(c['model_seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    device = torch.device(c.get('device', 'cuda:0'))
    model = IdentityOnlyCVS().to(device)
    checkpoint_smoke(model, out/'initial_smoke.pt', device)
    split = build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts'] != {'L_s': 6300, 'U_s': 56700, 'V': 27000}:
        raise ValueError('Source-only split/count mismatch')
    train_loader = make_cvs_loader(split.train, batch_size=128, shuffle=True,
        num_workers=0, device=device, drop_last=False)
    val_loader = make_cvs_loader(split.val, batch_size=256, shuffle=False,
        num_workers=0, device=device, drop_last=False)
    initial = dict(status='SCRATCH', scratch_only=True, checkpoint=None, ancestors=[],
        checkpoint_sources=[], target_access=False, target_contact=False,
        model_seed=seed, physical_roles='EXACT_MATCH', selection='fixed_last_epoch')
    write(out/'initialization.json', initial)
    resolved = dict(c, pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL_CHECK',
        torch_version=torch.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        source_counts=split.split_info['counts'], steps_per_epoch=len(train_loader),
        precision='float32', classifier='native cosine scale30, no label margin',
        U_s_use='unused', augmentation='PracticalResidualAugment; E80 onward only',
        gradient_clipping=None, optimizer='AdamW+CosineAnnealingLR', target_access=False)
    write(out/'resolved_config.json', resolved)
    write(out/'startup.json', resolved)
    print('RESOLVED_CONFIG '+json.dumps(resolved), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    augment = PracticalResidualAugment(fs_hz=25e6, seed=2027, receiver_seed=2027)
    steps_total = 0
    started = time.perf_counter()
    with (out/'step_metrics.jsonl').open('x', encoding='utf-8') as steps_file, \
         (out/'epoch_metrics.jsonl').open('x', encoding='utf-8') as epochs_file, \
         (out/'epoch_metrics.csv').open('x', encoding='utf-8', newline='') as csvfile:
        writer = None
        for epoch in range(1, 201):
            tic = time.perf_counter(); model.train(); augment.set_epoch(epoch)
            sums = dict(total_loss=0., clean_ce=0., satellite_ce=0., gradient_norm=0.)
            n = exposure = 0
            lr = optimizer.param_groups[0]['lr']
            for batch in train_loader:
                x, y = batch['iq'].to(device), batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, ce, sat = loss_for_batch(model, x, y, batch, augment, epoch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                # Observe gradients without clipping or changing the CVCNN update.
                norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm()
                    for p in model.parameters() if p.grad is not None]))
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                optimizer.step(); n += 1; steps_total += 1; exposure += len(y)
                record = dict(epoch=epoch, step=steps_total, total_loss=float(loss.detach()),
                    clean_ce=float(ce.detach()), satellite_ce=float(sat.detach()) if sat is not None else None,
                    clean_weight=1., satellite_weight=.68 if sat is not None else 0.,
                    gradient_norm=float(norm), learning_rate=lr, source_samples=len(y),
                    satellite_samples=len(y) if sat is not None else 0,
                    domain_backbone_active=False, extra_losses_active=False, pseudo_labels_active=False)
                steps_file.write(json.dumps(record, allow_nan=False)+'\n')
                for key in sums:
                    if record[key] is not None: sums[key] += record[key]
            if n != 50 or exposure != 6300: raise ValueError('CVCNN optimizer exposure mismatch')
            scheduler.step()
            metrics = dict(epoch=epoch, optimizer_steps=n, optimizer_steps_total=steps_total,
                source_sample_exposure=exposure, satellite_sample_exposure=exposure if epoch>=80 else 0,
                **{key: value/n for key, value in sums.items()}, learning_rate=lr,
                satellite_weight=.68 if epoch>=80 else 0.,
                **validate(model, val_loader, device), elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                domain_backbone_active=False, pseudo_labels_active=False, extra_losses_active=False)
            if epoch < 80: metrics['satellite_ce'] = None
            line = json.dumps(metrics, allow_nan=False)
            print('EPOCH '+line, flush=True)
            epochs_file.write(line+'\n'); epochs_file.flush(); steps_file.flush()
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=list(metrics)); writer.writeheader()
            writer.writerow(metrics); csvfile.flush()
    contract = json.loads((out/'source_contract.json').read_text(encoding='utf-8'))
    torch.save(dict(model=model.state_dict(), method='cvs_identity_ce', epoch=200,
        classes=contract['classes'], num_classes=6, source_contract=contract,
        initialization=initial, selection='fixed_last_epoch', config=resolved), out/'last.pt')
    write(out/'completion.json', dict(status='SOURCE_TRAINED', epoch=200, steps=steps_total,
        checkpoint=str(out/'last.pt'), elapsed_seconds=time.perf_counter()-started,
        target_access=False, target_evaluated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding='utf-8')))
