"""Scratch joint training: native DG + common concat CE + optional plain PL."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.phase1_dg_nativebudget import pipeline as native
from experiments.phase1_external_dg.models import FishrVariance, balanced_indices
from .augmentation import DefaultConcat
from .data import read, write
from .models import build, identity as forward
from .objectives import step
from .plan import make_plan

_PLAN = make_plan()
_ROWS = {r['row_id']: r for r in _PLAN['rows']}


def save_checkpoint(path,net,c,contract,epoch,stage,val_loss,sat_seen,sat_applied,u_seen,u_accepted):
    torch.save(dict(state_dict=net.state_dict(),source_contract=contract,config=c,
        model_seed=c['model_seed'],epoch=epoch,stage=stage,source_val_loss=val_loss,
        scratch_only=True,ancestors=[],default_head_joint_training=True,
        separate_head_training=False,target_access=False,
        training_telemetry=dict(satellite_examples=sat_seen,satellite_applied_examples=sat_applied,
            pl_seen=u_seen,pl_accepted=u_accepted)),path)


def check_training(c):
    forbidden = ('truth', 'target_truth', 'target_inputs', 'views_root', 'tests',
                 'dataset', 'resume', 'teacher_checkpoint', 'checkpoint', 'initial_checkpoint')
    if any(c.get(k) for k in forbidden):
        raise ValueError('Source process rejects target/dataset/inherited model inputs')
    registered = _ROWS.get(c.get('row_id'))
    if registered is None or any(c.get(k) != v for k, v in registered.items()):
        raise ValueError('Source row differs from the registered augmented recipe')
    meta = read(Path(c['source_cache']) / 'metadata.json')
    contract = read(c['source_contract_path'])
    p = _PLAN['protocols'][c['protocol_id']]
    if (meta['source_contract'] != contract or meta['target_constructed'] or
        meta['classes'] != c['classes'] or c['num_classes'] != len(meta['classes']) or
        contract['source_rxs'] != p['source_rxs'] or contract['source_days'] != p['source_days'] or
        c['num_domains'] != len(p['source_rxs']) or
        c['num_classes'] != (10 if p['dataset'] == 'ManyRx' else 6)):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if c['U_s_use']:
        if (not meta['U_s_exported'] or meta.get('U_s_labels_exported') is not False or
            contract.get('U_s_use') is not True or contract.get('U_s_labels_hidden') is not True):
            raise ValueError('PL requires a source-only, TX-label-free U cache')
        with np.load(Path(c['source_cache']) / 'U_s_index.npz', allow_pickle=False) as index:
            if set(index.files) != {'ids', 'rx'} or len(index['ids']) != meta['counts']['U_s']:
                raise ValueError('U cache contains forbidden labels or wrong role')
    elif meta['U_s_exported']:
        raise ValueError('Supervised control must not receive an unlabeled cache')
    if c['family'] in ('riei', 'drift', 'ibrff', 'sdg'):
        native.safe_row(c)
    return meta


def array_loader(c, role, batch_size=None):
    meta = read(Path(c['source_cache']) / 'metadata.json')
    roots = meta.get('role_cache_roots', {})
    path = Path(roots.get(role, c['source_cache']))
    if role == 'U_s' and not c['U_s_use']:
        raise ValueError('Supervised row cannot iterate U')
    x = torch.from_numpy(np.load(path / (role + '_x.npy'), allow_pickle=False).copy())
    with np.load(path / (role + '_index.npz'), allow_pickle=False) as ix:
        if role == 'U_s':
            if set(ix.files) != {'ids', 'rx'}: raise ValueError('U truth must remain hidden')
            tensors = (x, torch.from_numpy(ix['rx'].copy()).long())
        else:
            tensors = (x, torch.from_numpy(ix['y'].copy()).long(), torch.from_numpy(ix['rx'].copy()).long())
    generator = torch.Generator().manual_seed(c['model_seed'] + (991 if role == 'U_s' else 0))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),
        batch_size=batch_size or c['batch_size'], shuffle=role != 'V',
        generator=generator, num_workers=0, drop_last=role == 'U_s')


class SDGDataset(torch.utils.data.Dataset):
    def __init__(self, loader, author):
        self.x, self.y, self.rx = loader.dataset.tensors
        self.augmented = author.AugDataset(self.x.numpy(), self.y.numpy(), author.transforms.ToTensor(), [2], [1])

    def __len__(self): return len(self.y)

    def __getitem__(self, index):
        views, label = self.augmented[index]
        if int(label) != int(self.y[index]): raise ValueError('Author source label changed')
        return self.x[index], self.y[index], self.rx[index], views


def train(c):
    meta = check_training(c)
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); all_stages = []; total_sat = total_applied = total_u = total_accepted = 0
    write(out / 'resolved_config.json', dict(c, source_counts=meta['counts'], pid=os.getpid(),
        cwd=os.getcwd(), python=sys.executable, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        target_access=False, pseudo_label_truth_access=False,
        satellite_implementation='BaselineOriginSatViewAugment + sat_channel default LEO_WEAK',
        satellite_start_adaptation='first_batch; Core90 relative scene/probability schedule',
        satellite_loss='identity CE only; .68; no consistency',
        PL_implementation='current-model eval/no_grad confidence>=.95; masked hard CE / whole U batch' if c['U_s_use'] else None))
    for stage in range(1, c['stages'] + 1):
        native.seed(c['model_seed']); net = build(c).to(device)
        folder = out / ('stage%d' % stage); folder.mkdir()
        if stage == 2:
            parent = native.load(out / 'stage1/best.pt', c, meta['source_contract'], device)
            if parent['stage'] != 1: raise ValueError('Wrong same-run IB stage1 parent')
            net.load_state_dict(parent['state_dict'], strict=True)
        full = c['variant'] == 'full'
        fed = None
        if c['family'] == 'riei':
            params = list(net.ec.parameters()) + (list(net.rc.parameters()) if full else list(net.fed.parameters()))
            optimizer = torch.optim.SGD(params, lr=c['lr'], momentum=0.)
            if full: fed = torch.optim.SGD(net.fed.parameters(), lr=c['lr'], momentum=0.)
        elif c['family'] in ('resnet18', 'resnet50'):
            optimizer = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=c['lr'], weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1,
            patience=5, min_lr=1e-6) if c['family']=='ibrff' else None
        loader = array_loader(c, 'L_s'); val = array_loader(c, 'V', 256)
        author = None
        if c['family']=='sdg' and full:
            author = native.author_sdg(c)
            loader = torch.utils.data.DataLoader(SDGDataset(loader, author), batch_size=c['batch_size'],
                shuffle=True, generator=torch.Generator().manual_seed(c['model_seed']), num_workers=0)
        unlabeled_loader = array_loader(c, 'U_s') if c['U_s_use'] else None
        u_iterator = iter(unlabeled_loader) if unlabeled_loader else None
        balanced = c.get('balanced_sampling', False)
        data = array_loader(c, 'L_s').dataset.tensors if balanced else None
        sampling = torch.Generator().manual_seed(c['model_seed'])
        max_updates = c['updates'] if c['budget_kind']=='updates' else c['epochs'] * len(loader)
        report_every = 100 if c['budget_kind']=='updates' else len(loader)
        iterator = iter(loader); augment = DefaultConcat(c['augmentation_seed'])
        fishr = FishrVariance(c['num_domains'], c['fishr_ema']) if c.get('route')=='fishr' else None
        best = float('inf'); stale = 0; accum = []; stage_started = time.monotonic()
        stage_exposure = stage_sat = stage_applied = stage_u = stage_accepted = optimizer_steps = 0
        rx_coverage = {str(k): [0, 0] for k in range(c['num_domains'])}
        epoch = 0
        with (folder / 'epochs.jsonl').open('x', encoding='utf-8') as log:
            for index in range(max_updates):
                if c.get('optimizer_reset_step') is not None and index == c['optimizer_reset_step']:
                    optimizer = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
                progress = index/max_updates if c['budget_kind']=='updates' else (index//len(loader))/c['epochs']
                if c['family'] in ('resnet18', 'resnet50'):
                    lr = c['lr_min'] + (c['lr']-c['lr_min'])*.5*(1+math.cos(math.pi*progress))
                    for g in optimizer.param_groups: g['lr'] = lr
                if balanced:
                    xx, yy, rr = data
                    indices = balanced_indices(yy, rr, c['num_classes'], c['num_domains'], c['batch_size'], sampling)
                    batch = (xx[indices], yy[indices], rr[indices])
                else:
                    try: batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader); batch = next(iterator)
                x, y, rx = [v.to(device) for v in batch[:3]]
                views = [v.to(device) for v in batch[3]] if author else None
                u = u_rx = None
                if unlabeled_loader:
                    try: u, u_rx = next(u_iterator)
                    except StopIteration:
                        u_iterator=iter(unlabeled_loader); u, u_rx=next(u_iterator)
                    u, u_rx = u[:len(y)].to(device), u_rx[:len(y)]
                rec = step(net, c, x, y, rx, optimizer, augment, progress, index,
                    stage=stage, fed_optimizer=fed, unlabeled=u, native_views=views,
                    author=author, fishr=fishr, unlabeled_receivers=u_rx)
                accum.append(rec); stage_exposure += len(y); stage_sat += rec['satellite_samples']
                stage_applied += rec['satellite_applied_samples']; stage_u += rec['pl_seen']
                stage_accepted += rec['pl_accepted']; optimizer_steps += rec['optimizer_steps']
                for domain, counts in rec['pl_receiver_counts'].items():
                    rx_coverage[domain] = [a+b for a,b in zip(rx_coverage[domain], counts)]
                boundary = (index+1)%report_every==0 or index+1==max_updates
                if not boundary: continue
                epoch += 1
                val_loss, val_accuracy = native.validation(net, c, val, device)
                if scheduler: scheduler.step(val_loss)
                if val_loss < best:
                    best=val_loss; stale=0
                    if c['family'] in ('ibrff','sdg'):
                        save_checkpoint(folder/'best.pt',net,c,meta['source_contract'],epoch,stage,val_loss,
                            total_sat+stage_sat,total_applied+stage_applied,total_u+stage_u,total_accepted+stage_accepted)
                else: stale += 1
                record = dict(stage=stage, epoch=epoch, update=index+1, total_updates=max_updates,
                    interval=epoch, budget_kind=c['budget_kind'],
                    loss=sum(r['loss'] for r in accum)/len(accum),
                    identity_ce=sum(r['identity_ce'] for r in accum)/len(accum),
                    satellite_ce=sum(r['satellite_ce'] for r in accum)/len(accum),
                    native_penalty=sum(r['native_penalty'] for r in accum)/len(accum),
                    pl_loss=sum(r['pl_loss'] for r in accum)/len(accum),
                    source_val_loss=val_loss, source_val_accuracy=val_accuracy,
                    lr=optimizer.param_groups[0]['lr'], updates=index+1, optimizer_steps_total=optimizer_steps,
                    physical_source_exposure_total=stage_exposure, satellite_samples_total=stage_sat,
                    satellite_applied_samples_total=stage_applied, pl_seen_total=stage_u,
                    pl_accepted_total=stage_accepted, pl_acceptance=stage_accepted/stage_u if stage_u else 0.,
                    pl_receiver_coverage=rx_coverage,
                    satellite_scenarios=sorted({r['satellite_scenario'] for r in accum}),
                    satellite_weight=.68, concat_sat_ce_only=True,
                    U_s_use=c['U_s_use'], pseudo_label_truth_access=False,
                    seconds=time.monotonic()-stage_started, target_access=False)
                if c['budget_kind']=='updates': record['epoch'] = None
                log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                print('EPOCH '+json.dumps(record), flush=True); accum.clear()
                if c['family']=='ibrff' and stale>=10: break
        save_checkpoint(folder/'last.pt',net,c,meta['source_contract'],
            epoch if c['budget_kind']!='updates' else None,stage,val_loss,
            total_sat+stage_sat,total_applied+stage_applied,total_u+stage_u,total_accepted+stage_accepted)
        all_stages.append(dict(stage=stage, epochs=epoch if c['budget_kind']!='updates' else None,
            log_intervals=epoch, updates=index+1, optimizer_steps=optimizer_steps,
            seconds=time.monotonic()-stage_started, labeled_exposure=stage_exposure,
            satellite_examples=stage_sat, satellite_applied_examples=stage_applied,
            pl_seen=stage_u, pl_accepted=stage_accepted))
        total_sat += stage_sat; total_applied += stage_applied; total_u += stage_u; total_accepted += stage_accepted
        del net, optimizer; torch.cuda.empty_cache()
    if not total_sat or not total_applied or (c['U_s_use'] and not total_u):
        raise ValueError('Required concat/PL input branch never actually executed')
    chosen = out/('stage%d'%c['stages'])/('best.pt' if c['family'] in ('ibrff','sdg') else 'last.pt')
    selection = 'source_validation_nll' if c['family'] in ('ibrff','sdg') else 'fixed_final_budget'
    write(out/'freeze.json', dict(status='SOURCE_TRAINED_FROZEN', checkpoint=str(chosen), config=c,
        source_contract=meta['source_contract'], stages=all_stages, selection=selection,
        target_access=False, target_score_used=False, default_head=True,
        satellite_examples=total_sat, satellite_applied_examples=total_applied,
        pl_seen=total_u, pl_accepted=total_accepted,
        seconds=time.monotonic()-started))
    write(out/'training_cost.json', dict(wall_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), target_access=False))
    print('SOURCE_TRAINED_FROZEN', c['row_id'], flush=True)


def frozen_model(output, expected, device):
    frozen=read(Path(output)/'freeze.json');c=frozen['config'];check_training(c)
    if (frozen['status']!='SOURCE_TRAINED_FROZEN' or frozen['source_contract']!=expected or
        frozen['target_access'] or frozen['target_score_used'] or not frozen['satellite_applied_examples']):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    payload=native.load(frozen['checkpoint'],c,expected,device)
    if payload.get('default_head_joint_training') is not True or payload.get('separate_head_training') is not False:
        raise ValueError('Original joint default head required')
    actual=payload.get('training_telemetry',{})
    if not actual.get('satellite_applied_examples') or actual.get('pl_seen',0)==0 and c['U_s_use']:
        raise ValueError('Selected checkpoint lacks actual required augmentation/PL training')
    net=build(c);net.load_state_dict(payload['state_dict'],strict=True);net.to(device).eval()
    return net,c


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    train(read(parser.parse_args().config))
