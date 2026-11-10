"""Read-only source-V probe of frozen Tri-SIG-TIME SSL; no target or fitting."""
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

RUN = '20261110-diagnostic-pamf-period-sinc-sourcev-s2026092701-r01'
ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
PARENT = ROOT / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02'
ROW = 'main_split-repl-shuffled_time-3view-s2026092701'
HISTORIC_RELEASE = ROOT / 'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extract(reader, x):
    """Capture actual shared bank without recomputing the model forward."""
    core = reader.core.id_backbone
    original = core._sinc_on_iq
    capture = {}
    def record(self, actual):
        if capture:
            raise ValueError('Shared Sinc executed more than once')
        result = original(actual)
        capture['input'], capture['sinc'] = actual, result
        return result
    core._sinc_on_iq = MethodType(record, core)
    try:
        joint = reader.features(x)
        logits = reader.classify_features(joint)
    finally:
        core._sinc_on_iq = original
    if set(capture) != {'input', 'sinc'} or capture['sinc'].shape != (len(x), 48, 256):
        raise ValueError('Expected actual 24-channel shared IQ Sinc')
    ref = reader.response.components(x)
    z = torch.complex(x[:, 0], x[:, 1]); w = z[:, 80:160]
    cross = (w[:, 20:] * w[:, :-20].conj()).sum(-1)
    omega = torch.atan2(cross.imag, cross.real + 1e-12) / 20
    n = torch.arange(80, dtype=x.dtype, device=x.device)
    rotation = torch.complex(torch.cos(-omega[:, None]*n), torch.sin(-omega[:, None]*n))
    cycles = (w * rotation).reshape(-1, 4, 20)
    mean = cycles.mean(1)
    unit_scale = torch.sqrt(mean.abs().square().mean(-1) + 1e-12)
    unit_mean = mean / unit_scale[:, None]
    bank = torch.complex(reader.response.bank_real.to(x.dtype), reader.response.bank_imag.to(x.dtype))
    indices = (ref['selected_delay_samples'] * 16).round().long()
    reference = bank[indices]
    gain = (unit_mean * reference.conj()).mean(-1)
    bins = reader.response.tone_bins
    ideal = torch.fft.fft(reference, dim=-1)[:, bins]
    observed = torch.fft.fft(cycles / unit_scale[:, None, None], dim=-1)[:, :, bins]
    transfer = observed * ideal[:, None, :].conj() / ideal[:, None, :].abs().square()
    transfer = transfer * gain[:, None, None].conj() / (gain.abs().square()+1e-6)[:, None, None]
    if not torch.allclose(transfer.mean(1), ref['relative_transfer'], atol=5e-4, rtol=5e-4):
        raise ValueError('Cycle response does not reproduce original mean transfer')
    period_delta = transfer - transfer.mean(1, keepdim=True)
    shared = capture['sinc'].reshape(len(x), 2, 24, 256)
    full = torch.complex(shared[:, 0, :, 80:160], shared[:, 1, :, 80:160])
    phase = gain.conj() / (gain.abs()+1e-6)
    aligned = full * rotation[:, None, :] * phase[:, None, None]
    aligned = aligned.reshape(len(x), 24, 4, 20)
    sinc_delta = aligned - aligned.mean(2, keepdim=True)
    cropped = core.sinc.forward_iq_pair(capture['input'][:, :, 80:160]).reshape(len(x), 2, 24, 80)
    full_ri = shared[:, :, :, 80:160]
    diff = (full_ri - cropped).square().sum((1, 2))
    power = full_ri.square().sum((1, 2))
    def vector(a):
        return torch.cat((a.real.flatten(1), a.imag.flatten(1)), 1).float()
    values = dict(joint=joint.float(), reference=reader.response(x).float(),
                  period_delta=vector(period_delta), sinc_window=vector(aligned),
                  sinc_delta=vector(sinc_delta))
    if any(not bool(torch.isfinite(v).all()) for v in values.values()) or not bool(torch.isfinite(diff).all()):
        raise ValueError('Nonfinite feature or boundary output')
    return values, logits, diff, power, ref['quality'], ref['coherence']


def relation_indices(y, rx, day, ordinal):
    lookup = {(int(y[i]), int(rx[i]), int(day[i]), int(ordinal[i])): i for i in range(len(y))}
    if len(lookup) != len(y):
        raise ValueError('Duplicate source cell ordinal')
    triples = []
    for i in range(len(y)):
        for other_rx in (1, 3, 4, 6, 8):
            if other_rx == int(rx[i]):
                continue
            p = lookup[(int(y[i]), other_rx, int(day[i]), int(ordinal[i]))]
            for other_tx in range(6):
                if other_tx != int(y[i]):
                    triples.append((i, p, lookup[(other_tx, other_rx, int(day[i]), int(ordinal[i]))]))
    return torch.tensor(triples, dtype=torch.long)


def pair_scores(vec, idx):
    v = F.normalize(vec.double(), dim=1, eps=1e-12)
    gaps, positive, negative = [], [], []
    for chunk in idx.split(256):
        a, p, n = v[chunk[:, 0]], v[chunk[:, 1]], v[chunk[:, 2]]
        cp, cn = (a*p).sum(-1), (a*n).sum(-1)
        positive.append(cp); negative.append(cn); gaps.append(cp-cn)
    return torch.cat(positive), torch.cat(negative), torch.cat(gaps)


def quantiles(v):
    q = torch.quantile(v.double(), torch.tensor([.1, .5, .9], dtype=torch.double))
    return dict(mean=float(v.double().mean()), p10=float(q[0]), median=float(q[1]), p90=float(q[2]))


def main():
    out = ROOT / 'runs' / RUN
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = read(PARENT / 'configs' / ('train__' + ROW + '.json'))
    if cfg['model_seed'] != 2026092701 or cfg['source_rxs'] != [1,3,4,6,8] or cfg['source_days'] != [1,2,3]:
        raise ValueError('Unexpected frozen source contract')
    sys.path.insert(0, str(HISTORIC_RELEASE))
    from experiments.pamf_repl_completion.runner import activate
    activate(cfg)
    from experiments.pamf_repl_view_shuffle.source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    contract = read(cfg['source_contract'])
    reader, source = load_reader(cfg['output_root'], contract, 'cpu')
    reader.eval().requires_grad_(False)
    # Actual checkpoint no-query smoke precedes the first source read.
    t = torch.arange(256, dtype=torch.float32)
    smoke = torch.stack((torch.cos(t/7), torch.sin(t/7)), 0).unsqueeze(0)
    with torch.no_grad():
        smoke_out = extract(reader, smoke)
    if smoke_out[0]['joint'].shape != (1,160) or smoke_out[1].shape != (1,6):
        raise ValueError('Actual checkpoint smoke failed')
    data = SourceDataset(cfg['source_cache'], 'V', contract)
    meta = data.metadata
    ys, rxs, days = (np.asarray(meta[k]) for k in ('y','rx_i','day_i'))
    parts = []
    for yy in range(6):
        for rx in (1,3,4,6,8):
            for day in (1,2,3):
                ids = np.flatnonzero((ys==yy) & (rxs==rx) & (days==day))
                if len(ids) < 20:
                    raise ValueError('Insufficient balanced source V cell')
                parts.extend(ids[:20].tolist())
    selected = np.asarray(parts, dtype=np.int64)
    if len(selected) != 1800 or len(set(selected.tolist())) != 1800:
        raise ValueError('Incorrect selected V physical samples')
    y, rx, day = (torch.as_tensor(a[selected].copy(), dtype=torch.long) for a in (ys,rxs,days))
    ordinal = torch.arange(20).repeat(90)
    outputs = {k: [] for k in ('joint','reference','period_delta','sinc_window','sinc_delta')}
    diffs, powers, quality, coherence, errors = [], [], [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for begin in range(0, len(selected), 64):
            chosen = torch.as_tensor(selected[begin:begin+64], dtype=torch.long)
            x = data.x.index_select(0, chosen).cpu()
            feats, logits, d, p, q, coh = extract(reader, x)
            for k in outputs:
                outputs[k].append(feats[k].cpu())
            diffs.append(d.cpu()); powers.append(p.cpu()); quality.append(q.cpu()); coherence.append(coh.cpu())
            errors.append((logits.argmax(1).cpu() != y[begin:begin+len(x)]))
            if begin % 320 == 0: print('SOURCE_V',min(begin+len(x),len(selected)),flush=True)
    outputs = {k:torch.cat(v) for k,v in outputs.items()}
    idx = relation_indices(y, rx, day, ordinal)
    if idx.shape != (36000,3): raise ValueError('Unexpected fixed source relation count')
    score = {k:pair_scores(v,idx) for k,v in outputs.items()}
    base = score['joint'][2]
    results = {}
    for key,(pos,neg,gap) in score.items():
        results[key] = dict(positive=quantiles(pos), negative=quantiles(neg), gap=quantiles(gap),
            gap_nonpositive_fraction=float((gap<=0).double().mean()),
            by_anchor_rx={str(r):dict(count=int((rx[idx[:,0]]==r).sum()),
                gap_mean=float(gap[rx[idx[:,0]]==r].mean())) for r in (1,3,4,6,8)},
            by_anchor_day={str(t):float(gap[day[idx[:,0]]==t].mean()) for t in (1,2,3)},
            by_anchor_tx={str(t):float(gap[y[idx[:,0]]==t].mean()) for t in range(6)})
        if key != 'joint':
            results[key]['joint_nonpositive_candidate_positive'] = float(((base<=0)&(gap>0)).double().mean())
            results[key]['joint_positive_candidate_nonpositive'] = float(((base>0)&(gap<=0)).double().mean())
            results[key]['joint_gap_correlation'] = float(torch.corrcoef(torch.stack((base,gap)))[0,1])
    d, p = torch.cat(diffs).double(), torch.cat(powers).double()
    error = torch.cat(errors)
    frac = d.sum(1) / p.sum(1).clamp_min(1e-20)
    per_pos = d.sum(0) / p.sum(0).clamp_min(1e-20)
    summary = dict(status='SOURCE_ONLY_DIAGNOSTIC_COMPLETE', run_id=RUN,
        parent_run='20261027-phase1-pamf-repl-shuffled-views-m16-r02', model_row=ROW,
        checkpoint_source_only=True, model_epoch=50, seed=2026092701,
        role='V', population=27000, source_iq_loaded=27000, selected=1800, per_cell=20,
        selection='first 20 by source cache order in every TX/RX/day cell; no selection by score or quality',
        rxs=[1,3,4,6,8], days=[1,2,3], tx_count=6, relations=len(idx),
        relation='same TX/day/ordinal across RX vs different TX at reference RX/day/ordinal; distinct records, not synchronized events',
        reference='same frozen delay/phase/gain and 12 tones; four cycles minus their own complex mean',
        sinc='actual shared 24-filter IQ bank; full 256 then [80:160], phase-aligned; window and four-cycle centered delta',
        metrics=results, source_default_head_errors=int(error.sum()),
        quality=quantiles(torch.cat(quality)), coherence=quantiles(torch.cat(coherence)),
        sinc_boundary=dict(relative_squared_difference=quantiles(frac),
            pooled_relative_squared_difference=float(d.sum()/p.sum().clamp_min(1e-20)),
            relative_difference_per_position=per_pos.tolist(),
            interior_max_absolute_squared_error=float(d[:,39:41].max())),
        no_training=True, optimizer_updates=0, target_access=False, U_reads=0,
        limitations='Single frozen seed, selected source V, descriptive fixed relation; no feature fit, target result or causal TX attribution.',
        seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status=summary['status'],run_id=RUN,selected=1800,relations=len(idx),target_access=False,optimizer_updates=0))+'\n',encoding='utf-8')
    print('COMPLETED',json.dumps({'run_id':RUN,'selected':1800,'relations':len(idx)}),flush=True)


if __name__ == '__main__':
    main()
