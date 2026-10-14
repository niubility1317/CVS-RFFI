import gzip
import json
import statistics as st
import torch
from . import design as d
from experiments.cvs_source_layer_order.probe import gradient_stats


def recount(row, *, design=None):
    d = design or globals()['d']
    c = d.config(row); p = d.BASE / row['row_id'] / 'source'
    observed = d.read(p / 'receiver_risk_training_observation.json')
    if observed['status'] != 'VERIFIED' or observed['steps'] != 11100 or observed['U_packets'] != 963900:
        raise ValueError('Receiver-risk completed exposure differs')
    mode = c['receiver_risk']['mode']; logq = torch.zeros(5, dtype=torch.float64)
    curves = []; packets = steps = 0; grouped = {}
    with gzip.open(p / 'receiver_risk_steps.jsonl.gz', 'rt', encoding='utf8') as stream, \
         gzip.open(p / 'step_metrics.jsonl.gz', 'rt', encoding='utf8') as native:
        for steps, line in enumerate(stream, 1):
            r = json.loads(line); n = json.loads(next(native))
            if r['step'] != steps or r['epoch'] != (steps - 1) // 222 + 1 or n['step'] != steps or r['mode'] != mode:
                raise ValueError('Receiver-risk/native step identity differs')
            counts = torch.tensor(r['group_counts'], dtype=torch.float64)
            means = torch.tensor(r['group_mean_CE'], dtype=torch.float64)
            if counts.shape != (5,) or means.shape != (5,) or int(counts.sum()) != r['source_packets']:
                raise ValueError('Actual L receiver counts differ')
            if not torch.isfinite(means).all() or (means < 0).any() or (counts < 0).any():
                raise ValueError('Invalid saved receiver risk')
            if mode == 'DRO':
                logq += .01 * means
                logq -= torch.logsumexp(logq, 0)
            q = logq.softmax(0)
            applied = q * (counts > 0) if mode == 'DRO' else counts if mode == 'ERM' else (counts > 0).double()
            applied /= applied.sum()
            value = float(applied @ means)
            if not torch.allclose(q, torch.tensor(r['group_weights'], dtype=torch.float64), atol=1e-7, rtol=1e-6):
                raise ValueError('Saved adversarial group state differs')
            if not torch.allclose(applied, torch.tensor(r['applied_weights'], dtype=torch.float64), atol=1e-7, rtol=1e-6):
                raise ValueError('Applied receiver risk weights differ')
            raw = float((counts * means).sum() / counts.sum())
            if abs(value - r['selected_CE']) > 3e-6 or abs(raw - r['original_CE']) > 3e-6:
                raise ValueError('Actual receiver-risk objective differs')
            if abs(n['metrics']['train/loss_tx_labeled'] - r['selected_CE']) > 3e-6:
                raise ValueError('Native optimizer consumed a different identity loss')
            grouped.setdefault(r['epoch'], []).append(r); packets += r['source_packets']
        if next(native, None) is not None or steps != 11100:
            raise ValueError('Receiver risk/native step stream incomplete')
    for epoch, rr in grouped.items():
        if len(rr) != 222:
            raise ValueError('Receiver-risk epoch incomplete')
        counts = [sum(r['group_counts'][k] for r in rr) for k in range(5)]
        if not all(counts):
            raise ValueError('Source RX not observed during fixed epoch')
        means = [sum(r['group_counts'][k] * r['group_mean_CE'][k] for r in rr) / counts[k] for k in range(5)]
        curves.append(dict(epoch=epoch, group_counts=counts, group_CE=means,
            unweighted_CE=sum(n * m for n, m in zip(counts, means)) / sum(counts),
            worst_RX_CE=max(means), mean_max_group_weight=st.mean(max(r['group_weights']) for r in rr)))
    records = d.read(p / 'receiver_risk_gradient_records.json')['batch_records']
    layers = 0
    for epoch in d.DIAGNOSTIC_EPOCHS:
        raw = torch.load(p / f'receiver_risk_gradients_e{epoch}.pt', map_location='cpu', weights_only=False)
        expected = [r for r in records if r['epoch'] == epoch]
        if len(raw) != 13 or len(expected) != 13:
            raise ValueError('Actual risk derivative period incomplete')
        for j, (a, b) in enumerate(zip(raw, expected), 1):
            if a['step'] != (epoch - 1) * 222 + j * 16 or a['step'] != b['step']:
                raise ValueError('Actual risk derivative identity differs')
            for k, name in enumerate([*d.PARAMETERS, 'existing_default_W']):
                if b['layers'][k]['parameter'] != name:
                    raise ValueError('Actual risk derivative object differs')
                for key in ('selected', 'delta'):
                    if gradient_stats(a['vectors']['original'][k], a['vectors'][key][k]) != b['layers'][k][key]:
                        raise ValueError('Saved actual risk derivative recount differs')
                if mode == 'ERM' and b['layers'][k]['delta']['auxiliary_norm'] != 0:
                    raise ValueError('ERM control changed original gradient')
                layers += 1
    if layers != 208 or len(records) != 52:
        raise ValueError('Receiver-risk derivative budget differs')
    last = curves[-5:]
    return dict(status='VERIFIED', row_id=row['row_id'], arm=row['arm'], steps=steps,
        actual_L_packet_observations=packets, curves=curves,
        last5_unweighted_CE=st.mean(r['unweighted_CE'] for r in last),
        last5_worst_RX_CE=max(st.mean(r['group_CE'][k] for r in last) for k in range(5)),
        last5_mean_max_group_weight=st.mean(r['mean_max_group_weight'] for r in last),
        native_U_objective_unchanged=observed['U_loss_returned_unchanged'],
        body_signal=all(any(r['epoch'] == ep and r['layers'][k]['delta']['auxiliary_norm'] > 1e-12
                            for r in records) for ep in (17, 33, 50) for k in range(3)),
        gradient_layer_records=layers, saved_vectors=layers * 3, extra_model_forwards=0,
        head_fit_updates=0, query_read=False)
