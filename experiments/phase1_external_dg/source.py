"""Scratch source training for the non-PAMF comparison matrix."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time

from .data import read, write
from .plan import SEEDS, make_plan
from . import models

# Reuse the committed, reviewed native training objectives. This module only
# changes registered data cardinalities in its own short-lived source process.
from experiments.phase1_dg_nativebudget import pipeline as native

_NATIVE_CHECK = native.safe_row
_PLAN = make_plan()
_ROWS = {r['row_id']: r for r in _PLAN['rows'] if r['budget_kind'] != 'frozen_eval'}


def check_training(c):
    forbidden = ('truth', 'target_truth', 'target_inputs', 'views_root', 'tests',
                 'dataset', 'resume', 'teacher', 'checkpoint', 'initial_checkpoint')
    if any(c.get(k) for k in forbidden):
        raise ValueError('Source process rejects target/dataset/inherited model inputs')
    registered = _ROWS.get(c.get('row_id'))
    if registered is None or any(c.get(key) != value for key, value in registered.items()):
        raise ValueError('Source row differs from the registered comparison recipe')
    if (c['model_seed'] not in SEEDS or c.get('initialization') != 'scratch' or
        c.get('U_s_use') is not False or c.get('satellite_train_augmentation') is not False or
        c.get('default_head_joint_training') is not True or c.get('separate_head_training') is not False):
        raise ValueError('Training scope or default-head contract mismatch')
    meta = read(Path(c['source_cache']) / 'metadata.json')
    contract = read(c['source_contract_path'])
    protocol = _PLAN['protocols'][c['protocol_id']]
    if (meta['source_contract'] != contract or meta['U_s_exported'] or
        meta['target_constructed'] or meta['classes'] != c['classes'] or
        len(contract['source_rxs']) != c['num_domains'] or len(c['classes']) != c['num_classes'] or
        contract['source_rxs'] != protocol['source_rxs'] or contract['source_days'] != protocol['source_days'] or
        c['num_classes'] != (6 if protocol['dataset'] == 'ManySig' else 10)):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if c['family'] in ('riei', 'drift', 'ibrff', 'sdg'):
        _NATIVE_CHECK(c)
    elif c['family'] == 'resnet18':
        if (c['lr'], c['weight_decay'], c['batch_size']) not in (
            (.0002, .0001, 150), (.0002, .0001, 180), (.0002, .0001, 128)):
            raise ValueError('Changed ResNet comparison budget')
        if c['group'] == 'generic':
            if c['updates'] != 5000 or c['batch_size'] != 150 or c['route'] not in (
                'erm', 'mixstyle', 'mixstyle_id', 'fishr', 'coral'):
                raise ValueError('Unregistered generic mechanism')
        elif c['group'].startswith('count-'):
            if c['updates'] != 10000 or c['batch_size'] != 180 or c['route'] != 'erm':
                raise ValueError('Unregistered source-count budget')
        elif c['group'] in ('label01', 'label05'):
            if c['epochs'] != 200 or c['batch_size'] != 128 or c['route'] != 'erm':
                raise ValueError('Changed label-efficiency control')
        else:
            raise ValueError('Unknown ResNet training group')
    else:
        raise ValueError('PAMF and frozen-only architectures cannot enter training')
    return meta


def build(c):
    if c['family'] == 'riei':
        from baselines.riei_fd.model import RIEIModel
        return RIEIModel(c['num_classes'], c['num_domains'], fed_variant='short_stem1d')
    if c['family'] == 'drift':
        from baselines.drift.model import DRIFTModel
        return DRIFTModel(c['num_classes'], c['num_domains'], domain_discriminator_layers=2)
    if c['family'] == 'ibrff':
        sys.path.insert(0, c['ibrff_model_root'])
        from backbones.FAN import FAN
        return FAN(seq_len=256, in_chans=2, patch_size=32, embed_dim=128,
            mlp_ratio=2., num_classes=c['num_classes'], bottleneck_type='DSCNN',
            use_SAF=True, use_FAF=True)
    if c['family'] == 'sdg':
        author = native.author_sdg(c)
        native.seed(c['model_seed'])
        return author.MACNN(in_channels=2, channels=8, num_classes=c['num_classes'])
    if c['family'] == 'resnet18':
        return models.RoutedResNet(c['num_classes'], c['route'])
    raise ValueError('Non-PAMF training family required')


def forward(net, c, x):
    if c['family'] == 'riei':
        return net.ec(net.fed(x)[:, :net.emitter_feature_dim])
    if c['family'] == 'drift':
        return net.tx_classifier(net.encoder(x)[:, :net.split_dim])
    return net(x)[1]


def configure_native(c, meta):
    native.COUNTS = meta['counts']
    native.CLASSES = meta['classes']
    native.safe_row = check_training
    native.build = build
    native.forward = forward


def train_resnet(c, meta):
    import torch
    import torch.nn.functional as F
    native.seed(c['model_seed'])
    device = torch.device('cuda:0')
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    net = build(c).to(device)
    train_loader = native.array_loader(c, 'L_s')
    val_loader = native.array_loader(c, 'V', 256)
    x_data, labels, receivers = train_loader.dataset.tensors
    balanced = c['balanced_sampling']
    total = c['updates'] if c['budget_kind'] == 'updates' else c['epochs'] * len(train_loader)
    if not total: raise ValueError('Empty training budget')
    generator = torch.Generator().manual_seed(c['model_seed'])
    state = models.FishrVariance(c['num_domains'], c['fishr_ema']) if c['route'] == 'fishr' else None
    optimizer = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    mix_pairs = [0, 0]
    write(out / 'resolved_config.json', dict(c, source_counts=meta['counts'],
        pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        target_access=False, actual_total_updates=total, default_head_joint_training=True,
        separate_head_training=False, total_parameters=sum(p.numel() for p in net.parameters()),
        mechanism=dict(route=c['route'], mixstyle_layers=['features.5', 'features.7']
            if c['route'].startswith('mixstyle') else [], identity_protected=c['route'] == 'mixstyle_id',
            fishr_gradient_parameters=['classifier.weight', 'classifier.bias']
            if c['route'] == 'fishr' else [], source_domain_grouping='receiver')))
    iterator = iter(train_loader)
    started = time.monotonic(); exposure = 0; accumulated = []
    log = (out / 'steps.jsonl').open('x', encoding='utf-8')
    try:
        for index in range(total):
            reset = c.get('optimizer_reset_step')
            if reset is not None and index == reset:
                optimizer = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
            # All routes retain the same global LR trajectory across a reset.
            progress = index / total if c['budget_kind'] == 'updates' else (index // len(train_loader)) / c['epochs']
            lr = c['lr_min'] + (c['lr'] - c['lr_min']) * .5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups: group['lr'] = lr
            if balanced:
                ix = models.balanced_indices(labels, receivers, c['num_classes'],
                    c['num_domains'], c['batch_size'], generator)
                x, y, rx = x_data[ix], labels[ix], receivers[ix]
            else:
                try: x, y, rx = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader); x, y, rx = next(iterator)
            x, y, rx = x.to(device), y.to(device), rx.to(device)
            net.train(); net.context = (y, rx)
            z, logits = net(x); ce = F.cross_entropy(logits, y)
            mix_pairs = [a + b for a, b in zip(mix_pairs, net.mixstyle_counts)]
            penalty = ce.new_zeros(()); weight = 0.
            if c['route'] == 'fishr':
                penalty = state(z, logits, y, rx)
                weight = c['fishr_lambda'] if index >= c['fishr_warmup'] else 0.
            elif c['route'] == 'coral':
                penalty = models.source_coral(z, rx, c['num_domains']); weight = c['coral_lambda']
            loss = ce + weight * penalty
            optimizer.zero_grad(); loss.backward(); native.finite(net, loss); optimizer.step()
            net.context = None
            exposure += len(y)
            accumulated.append((float(loss.detach()), float(ce.detach()), float(penalty.detach())))
            boundary = ((index + 1) % (100 if balanced else len(train_loader)) == 0 or index + 1 == total)
            if boundary:
                val_loss, val_acc = native.validation(net, c, val_loader, device)
                record = dict(update=index + 1, total_updates=total, learning_rate=lr,
                    loss=sum(x[0] for x in accumulated) / len(accumulated),
                    ce=sum(x[1] for x in accumulated) / len(accumulated),
                    penalty=sum(x[2] for x in accumulated) / len(accumulated),
                    penalty_weight=weight, physical_source_exposure=exposure,
                    mixstyle_pairs_total=list(mix_pairs),
                    source_val_loss=val_loss, source_val_accuracy=val_acc,
                    seconds=time.monotonic() - started, target_access=False)
                import json
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                print('SOURCE_UPDATE', json.dumps(record), flush=True); accumulated.clear()
    finally:
        log.close()
    native.save(out / 'last.pt', net, c, meta['source_contract'],
        c['epochs'], 1, val_loss)
    write(out / 'freeze.json', dict(status='SOURCE_TRAINED_FROZEN',
        checkpoint=str(out / 'last.pt'), config=c, source_contract=meta['source_contract'],
        selection='fixed_final_budget', target_access=False, target_score_used=False,
        updates=total, physical_source_exposure=exposure, mixstyle_pairs_total=mix_pairs,
        seconds=time.monotonic() - started))
    print('SOURCE_TRAINED_FROZEN', c['row_id'], total, flush=True)


def train(c):
    import torch
    meta = check_training(c)
    configure_native(c, meta)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if c['family'] == 'resnet18':
        train_resnet(c, meta)
    else:
        native.train(c)
    write(Path(c['output_root']) / 'training_cost.json', dict(
        wall_seconds=time.monotonic() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_counts=meta['counts'], target_access=False,
        note='Training process cost; excludes independent data preparation and prediction.'))


def frozen_model(source_output, expected_contract, device):
    import torch
    frozen = read(Path(source_output) / 'freeze.json')
    c = frozen['config']
    meta = check_training(c)
    if (frozen['status'] != 'SOURCE_TRAINED_FROZEN' or frozen['target_access'] is not False or
        frozen['target_score_used'] or frozen['source_contract'] != expected_contract):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    configure_native(c, meta)
    payload = native.load(frozen['checkpoint'], c, expected_contract, device)
    if payload.get('default_head_joint_training') is not True or payload.get('separate_head_training') is not False:
        raise ValueError('Default classification head contract failed')
    net = build(c).to(device); net.load_state_dict(payload['state_dict'], strict=True); net.eval()
    with torch.no_grad(): value = forward(net, c, torch.zeros(2, 2, 256, device=device))
    if value.shape != (2, c['num_classes']) or not torch.isfinite(value).all():
        raise ValueError('Real checkpoint no-query smoke failed')
    return net, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    train(read(parser.parse_args().config))
