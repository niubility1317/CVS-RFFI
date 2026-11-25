"""Compare native W24 inputs on a source receiver excluded from fitting."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .clean_aperture import clean_iq, scale_native_band_tail
from .comparison import seed_all
from .data import SOURCE
from .full_packet_data import COUNTS
from .full_packet_views import native, source_contract
from .full_supervised import batches
from .runner import activate
from .tri_clean95 import (
    NATIVE_BAND_ROWS, NATIVE_JITTER_ROWS, NATIVE_RAW_ROWS, NATIVE_WIDE_ROWS,
    clean_model, clean_validation, labels, labeled_training_views, scratch_repl_lr_factor,
    supervised_clean_terms,
)
from .tri_tuning import SEED


HELD = SOURCE.index('rtl_1')
EPOCHS = 48
STEPS = 70
ROWS = {
    'native24': NATIVE_WIDE_ROWS['phase_native24'],
    'native24_jitter': NATIVE_JITTER_ROWS['phase_native24_jitter'],
}
RAW_HEAD_ROWS = {'native24_raw': NATIVE_RAW_ROWS['phase_native24_raw']}
BAND_TAIL_ROWS = {'native24_band': NATIVE_BAND_ROWS['phase_native24_band']}
CENTER_ROWS = {'native24_center_half': dict(ROWS['native24'], center_tail_gain=.5)}


def source_runtime(rt):
    if set(rt) != {'run_root', 'source_root', 'frozen_sources'}:
        raise ValueError('The diagnostic accepts only a source-only runtime')
    source = Path(rt['source_root'])
    return Path(rt['run_root']), source, source_contract(source)


def fitting_indices(receivers):
    keep = np.flatnonzero(receivers != HELD)
    counts = np.bincount(receivers[keep], minlength=len(SOURCE))
    expected = np.full(len(SOURCE), 560)
    expected[HELD] = 0
    if not np.array_equal(counts, expected):
        raise ValueError('The fit requires exactly four source receivers')
    return keep


def extract(packets, indices):
    row = ROWS['native24']
    return np.concatenate([
        clean_iq(packets[indices[i:i+64]], row['samples'], row['chirps'],
                 native_edges=True, edge_points=row['edge_points'])
        for i in range(0, len(indices), 64)
    ])


def model(root, contract, row, device):
    seed_all(SEED)
    net = clean_model({'scratch': True}, root, contract, row, device)
    seed_all(SEED)
    return net


def diagnostic_augmentations(packets, row):
    gain = row.get('center_tail_gain')
    if gain is None:
        return labeled_training_views(packets, row)
    def view(raw):
        return clean_iq(raw, row['samples'], row['chirps'], native_edges=True,
                        edge_points=row['edge_points'])
    windows = []
    for start in range(0, len(packets), 64):
        raw = packets[start:start+64]
        clean = view(raw)
        adjusted = view(scale_native_band_tail(raw, gain))
        adjusted[..., row['samples']:] = clean[..., row['samples']:]
        windows.append(adjusted)
    return (np.concatenate(windows),)


def smoke(rt, *, raw_head=False, band_tail=False, center_attenuation=False):
    root, source, contract = source_runtime(rt)
    activate(rt)
    y, rx = labels(source, contract, 'L_s')
    subset = fitting_indices(rx)
    subset = subset[next(batches(y[subset], rx[subset], np.random.default_rng(721)))[:8]]
    raw = native(source, 'L_s', COUNTS['L_s'])
    if raw_head:
        row = RAW_HEAD_ROWS['native24_raw']
        data = tensor(extract(raw,subset))
    elif center_attenuation:
        row = CENTER_ROWS['native24_center_half']
        original = extract(raw, subset)
        adjusted, = diagnostic_augmentations(raw[subset], row)
        n = row['samples']
        if (not np.array_equal(adjusted[..., n:], original[..., n:]) or
                np.array_equal(adjusted[..., :n], original[..., :n])):
            raise ValueError('Center attenuation must preserve native edges and change the center')
        data = tensor(adjusted)
    else:
        row = BAND_TAIL_ROWS['native24_band'] if band_tail else ROWS['native24_jitter']
        before, after = diagnostic_augmentations(raw[subset],row)
        data = tensor(np.concatenate((before[:4], after[4:]),axis=0))
    fitted = model(root,contract,row,'cpu')
    with torch.enable_grad():
        loss,_,raw_loss = supervised_clean_terms(fitted,data,tensor(y[subset]),tensor(rx[subset]),
                                                 1.,0.,None,row.get('warm_raw_ce',0.))
        loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(fitted.parameters(), 5., error_if_nonfinite=True))
    if grad <= 0 or np.any(rx[subset] == HELD):
        raise ValueError('The source-only real diagnostic smoke failed')
    root.mkdir(parents=True, exist_ok=False)
    write(root/'smoke.json', dict(status='PASS', input=list(data.shape[1:]),
                                  held_rx=SOURCE[HELD], V_reads=0, U_reads=0,
                                  target_reads=0, gradient=grad,
                                  raw_head_ce=float(raw_loss.detach()) if raw_loss is not None else None))


def fit_row(root, contract, name, row, x, y, rx, yn, rn, augmentations, device):
    out = root/'F'/name
    out.mkdir(parents=True, exist_ok=False)
    f = model(out, contract, row, device)
    opt = torch.optim.AdamW(f.parameters(), lr=3e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, partial(scratch_repl_lr_factor, warm=96*STEPS, decay_end=112*STEPS))
    rng = np.random.default_rng(721)
    draw = torch.Generator(device=device).manual_seed(SEED+11)
    steps = 0
    raw_weight = row.get('warm_raw_ce',0.)
    record_band_count = name in BAND_TAIL_ROWS or name in CENTER_ROWS
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1, EPOCHS+1):
            total = torch.zeros((),device=device)
            raw_total = torch.zeros((),device=device)
            altered = torch.zeros((),device=device,dtype=torch.int64)
            f.train()
            for b in batches(yn, rn, rng):
                i = tensor(b,device)
                labeled = x[i]
                if augmentations:
                    q = torch.rand(len(b),device=device,generator=draw)[:,None,None,None]
                    changed = q < .125
                    labeled = torch.where(changed, augmentations[0][i], labeled)
                    if len(augmentations) == 2:
                        high = q > .875
                        labeled = torch.where(high, augmentations[1][i], labeled)
                        changed = changed | high
                    if record_band_count:
                        altered += changed.sum()
                opt.zero_grad(set_to_none=True)
                loss,_,raw_ce = supervised_clean_terms(f,labeled,y[i],rx[i],1.,0.,None,raw_weight)
                if raw_ce is not None:
                    raw_total += raw_ce.detach()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                sched.step()
                steps += 1
                total += loss.detach()
            rec = dict(epoch=epoch, step=steps, source_L_loss=float(total/STEPS))
            if raw_weight:
                rec['raw_head_ce'] = float(raw_total/STEPS)
            if record_band_count:
                rec['source_l_band_scaled_examples'] = int(altered)
            log.write(json.dumps(rec,allow_nan=False)+chr(10))
            log.flush()
            print(name,epoch,steps,rec['source_L_loss'],flush=True)
    if steps != STEPS*EPOCHS:
        raise ValueError('Unexpected diagnostic fitting step count')
    torch.save(f.state_dict(),out/'model.pt')
    write(out/'frozen.json',dict(status='SOURCE_DIAGNOSTIC_FROZEN', seed=SEED, epochs=EPOCHS,
                                held_rx=SOURCE[HELD], n_L=len(y), U_reads=0, V_reads_during_fit=0,
                                target_reads=0, satellite_batches=0))


def fit(rt, *, raw_head=False, band_tail=False, center_attenuation=False):
    rows = RAW_HEAD_ROWS if raw_head else ROWS
    augmentation_row = ROWS['native24_jitter']
    if band_tail:
        rows = BAND_TAIL_ROWS
        augmentation_row = rows['native24_band']
    if center_attenuation:
        rows = CENTER_ROWS
        augmentation_row = rows['native24_center_half']
    root, source, contract = source_runtime(rt)
    activate(rt)
    device = 'cuda:0'
    yn,rn = labels(source,contract,'L_s')
    keep = fitting_indices(rn)
    root.mkdir(parents=True,exist_ok=False)
    raw = native(source,'L_s',COUNTS['L_s'])
    x = tensor(extract(raw,keep),device)
    augmentations = () if raw_head else tuple(
        tensor(v,device) for v in diagnostic_augmentations(raw[keep],augmentation_row))
    yn,rn = yn[keep],rn[keep]
    y,rx = tensor(yn,device),tensor(rn,device)
    for name, row in rows.items():
        row_augmentations = augmentations if row is augmentation_row else ()
        fit_row(root,contract,name,row,x,y,rx,yn,rn,row_augmentations,device)
    del x,augmentations,row_augmentations
    vy,vr = labels(source,contract,'V')
    v = tensor(extract(native(source,'V',COUNTS['V']),np.arange(len(vy))),device)
    v_y,v_rx = tensor(vy,device),tensor(vr,device)
    summary = {}
    for name, row in rows.items():
        f = model(root,contract,row,device)
        f.load_state_dict(torch.load(root/'F'/name/'model.pt',map_location=device,weights_only=True),strict=True)
        score = clean_validation(f,v,v_y,v_rx)
        n_held = int((vr==HELD).sum())
        held = round(score['by_rx'][SOURCE[HELD]]*n_held)
        summary[name] = dict(held_correct=held,held_n=n_held,seen_correct=score['correct']-held,
                             seen_n=score['n']-n_held,source_V=score)
    write(root/'source_holdout_scores.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',
          source_protocol=contract['protocol'], held_rx=SOURCE[HELD], seed=SEED, n_L=2240,
          target_reads=0, U_reads=0, satellite_batches=0, rows=summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('smoke','fit'))
    parser.add_argument('runtime',type=Path)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--raw-head',action='store_true')
    profile.add_argument('--band-tail',action='store_true')
    profile.add_argument('--center-attenuation',action='store_true')
    args = parser.parse_args()
    {'smoke':smoke,'fit':fit}[args.action](read(args.runtime),raw_head=args.raw_head,
                                         band_tail=args.band_tail,center_attenuation=args.center_attenuation)
