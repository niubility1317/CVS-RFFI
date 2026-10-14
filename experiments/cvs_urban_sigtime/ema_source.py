"""Calibrate existing clean EMA targets on legal held-out source V only."""
import time
from pathlib import Path

import torch

from . import design as d
from .model import own_model

THRESHOLDS = (.92, .95, .97)
WEAK_STD = .01
WEAK_SEED = 2026101501
OUTPUT = 'source_ema_vcal'


def summarize(prob, labels, rx, weak_prob):
    if prob.shape != weak_prob.shape or prob.shape != (len(labels), 6):
        raise ValueError('Default-head probability shape differs')
    if (labels < 0).any() or (labels >= 6).any():
        raise ValueError('Only legal labeled source V may be scored')
    if not torch.isfinite(prob).all() or not torch.isfinite(weak_prob).all():
        raise ValueError('Nonfinite frozen teacher probability')
    confidence, prediction = prob.max(1)
    weak_confidence, weak_prediction = weak_prob.max(1)
    correct = prediction == labels
    agreement = prediction == weak_prediction
    entries = []
    for threshold in THRESHOLDS:
        for mode in ('clean_confidence', 'same_sample_weak_agreement'):
            selected = confidence >= threshold
            if mode == 'same_sample_weak_agreement':
                selected = selected & agreement & (weak_confidence >= threshold)
            groups = []
            for receiver in sorted(rx.unique().tolist()):
                for tx in range(6):
                    group = (rx == receiver) & (labels == tx)
                    mask = group & selected
                    count = int(mask.sum())
                    groups.append(dict(RX=receiver, TX=tx, count=int(group.sum()),
                        selected=count, precision=float(correct[mask].float().mean()) if count else None))
            count = int(selected.sum())
            entries.append(dict(threshold=threshold, mode=mode, count=len(labels),
                selected=count, coverage=count / len(labels),
                precision=float(correct[selected].float().mean()) if count else None,
                wrong_soft_mass=float((1-prob[selected].gather(1, labels[selected, None])).mean()) if count else None,
                RX_TX=groups))
    return dict(count=len(labels), accuracy=float(correct.float().mean()),
        confidence_mean=float(confidence.mean()),
        weak_agreement=float(agreement.float().mean()), criteria=entries)


def run():
    out = d.BASE / OUTPUT
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as nd
    _, U, V, contract = load_source(nd.config(nd.rows()[0]))
    if U is not None or len(V['ids']) != 27000 or set(V['ids']) != set(contract['role_ids']['V']):
        raise ValueError('Read-only source V role differs; U must stay unopened')
    generator = torch.Generator().manual_seed(WEAK_SEED)
    weak = V['x'] + WEAK_STD * torch.randn(V['x'].shape, generator=generator)
    started = time.perf_counter()
    results = {}
    for seed in d.SEEDS:
        c = d.config(next(r for r in d.rows() if r['arm'] == 'C4' and r['model_seed'] == seed))
        model = own_model(c, 'cpu')
        checkpoint = torch.load(Path(c['output_root'])/'final_ssdg.pth', map_location='cpu', weights_only=False)
        if checkpoint['ema_model'] is None or not checkpoint['args']['use_ema_teacher']:
            raise ValueError('The existing own EMA provider is unavailable')
        # own_model has already checked scratch origin, complete source contract,
        # exact E30 budget and final-only freeze; teacher is that checkpoint's EMA.
        model.load_state_dict(checkpoint['ema_model'], strict=True)
        model.eval().requires_grad_(False)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        clean_prob, weak_prob = [], []
        with torch.no_grad():
            for start in range(0,27000,256):
                x = V['x'][start:start+256]
                full = model(x, y_tx=None, grl_lambda=1., return_aux=True, domain_labels=None)['tx_logits']
                if start == 0:
                    deployed = model.forward_identity_only(x)['tx_logits']
                    if not torch.equal(full, deployed):
                        raise ValueError('Actual teacher and default identity route differ')
                    print('SMOKE PASS real own EMA checkpoint; source V only; query_read=false', flush=True)
                clean_prob.append(full.softmax(1))
                weak_prob.append(model(weak[start:start+256], y_tx=None, grl_lambda=1.,
                    return_aux=True, domain_labels=None)['tx_logits'].softmax(1))
        if any(not torch.equal(v, model.state_dict()[k]) for k,v in before.items()):
            raise ValueError('Frozen EMA calibration changed model state')
        results[c['row_id']] = dict(summarize(torch.cat(clean_prob), V['y'], V['rx'], torch.cat(weak_prob)),
            state_unchanged=True, provider='own final E30 EMA, not final source student',
            checkpoint_sources=[], query_read=False, U_truth_read=False,
            forward_packets=54000+256, head_fit_updates=0)
        d.write(out/'progress.json',dict(rows=list(results),source_only=True,query_read=False))
        print('SOURCE EMA '+c['row_id']+' complete',flush=True)
        del model, checkpoint, before
    value = dict(status='SOURCE_EMA_CALIBRATION_COMPLETE', source_only=True, query_read=False,
        U_truth_read=False, head_fit_updates=0, added_training_updates=0, rows=results,
        weak_view=dict(kind='clean IQ plus light independent Gaussian noise',std_per_IQ_coordinate=WEAK_STD,
            seed=WEAK_SEED,scope='fixed source V calibration only; no change to formal practical IQ'),
        seconds=time.perf_counter()-started, commit=(d.ROOT/'release_commit.txt').read_text().strip(),
        limitations=['Source V precision does not prove hidden U or target precision.',
            'Confidence mean does not equal confidence-threshold coverage.',
            'No threshold or next model has been selected from target scores.'])
    d.write(out/'results.json',value)
    print('SOURCE_EMA_CALIBRATION_COMPLETE',flush=True)


if __name__ == '__main__':
    run()
