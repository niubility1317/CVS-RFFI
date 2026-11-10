"""Frozen source L/V physical probe; no U, target, fitting, or checkpoint selection."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_branch_training.relation import capture_branches
from .geometry import bounded_echo, tangent
from .objective import echo_parameters


def identity_basis(features, labels, classes=6):
    """Descriptive L-only class-centroid subspace, not a causal TX decomposition."""
    import torch
    if features.ndim != 2 or labels.shape != features.shape[:1] or classes < 2:
        raise ValueError('Aligned L feature/class vectors required')
    means = []
    for label in range(classes):
        selected = features[labels == label]
        if not len(selected):
            raise ValueError('Every source class required for descriptive basis')
        means.append(selected.mean(0))
    matrix = torch.stack(means)
    matrix = matrix - matrix.mean(0, keepdim=True)
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(int((singular > singular.max() * 1e-6).sum()), classes - 1) if float(singular.max()) > 0 else 0
    return right[:rank].T.contiguous(), rank


def direction_overlap(directions, basis):
    """Per-sample fraction of direction energy in an L-derived subspace."""
    if directions.ndim != 2 or basis.ndim != 2 or directions.shape[1] != basis.shape[0]:
        raise ValueError('Aligned branch directions and class basis required')
    denominator = directions.square().sum(1)
    numerator = (directions @ basis).square().sum(1)
    return numerator / denominator.clamp_min(1e-12), denominator > 1e-12


def as_view(raw, meta, indices, scene, role):
    """Original T1 received-IQ As proxy with a fixed source-only seed."""
    from leo_practical.channel import Config, stable_seed, receiver_for_session, normalize_input
    from leo_practical.residual import ResidualChannel
    cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                 mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
    result = []
    keys = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
    for wave, index in zip(raw, indices):
        pid = tuple(int(meta[k][index]) for k in keys)
        if pid[0] != 1:
            raise ValueError('Original T1 As requires the equalized source contract')
        session = 'rx%d:day%d:eq1' % (pid[1], pid[2])
        hw = receiver_for_session(cfg, 20261016423, session)
        signal, _ = normalize_input(wave[0] + 1j * wave[1])
        signal, detail = ResidualChannel(cfg, stable_seed(202611201, 'PTBR_source_probe', role, scene, pid), hw).process(signal)
        if not np.isfinite(signal).all() or detail['channel_equalization_applied']:
            raise ValueError('Source As proxy must remain finite and no-EQ')
        result.append(np.stack((signal.real, signal.imag)))
    return np.ascontiguousarray(result, dtype=np.float32)


def run(c, device='cuda:0'):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_paper_completion.model import load_reader
    if any(c.get(k) for k in ('truth', 'target_truth', 'tests', 'views_root', 'target_inputs')):
        raise ValueError('Physical diagnostic is source-only')
    contract = read(c['source_contract'])
    reader, _ = load_reader(c['source_output'], contract, c['paper_row'], device)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    totals, basis, basis_report = {}, {}, {}
    started = time.monotonic()
    for role in ('L_s', 'V'):
        data = SourceDataset(c['source_cache'], role, contract)
        meta = data.metadata
        groups, l_vectors, l_directions, l_labels, overlaps = {}, {}, {}, [], {}
        for begin in range(0, len(data), 128):
            end = min(begin + 128, len(data))
            x = data.x[begin:end].to(device)
            coeff, delay = echo_parameters(len(x), x.device, x.dtype, 202611200 + begin)
            alternate = bounded_echo(x, coeff, delay)
            core, sequence, reference = reader.core.id_backbone, reader.sequence_pool, reader.response_projection
            with torch.no_grad():
                with capture_branches(core, sequence, reference) as b0:
                    f0 = reader.features(x)
                with capture_branches(core, sequence, reference) as b1:
                    f1 = reader.features(alternate)
                p0 = reader.classify_features(f0).softmax(1)
                p1 = reader.classify_features(f1).softmax(1)
                if not bool(torch.isfinite(p0).all()) or not bool(torch.isfinite(p1).all()):
                    raise FloatingPointError('Nonfinite source diagnostic classifier output')
                confidence, raw = p0.max(1)
                confidence1, echo = p1.max(1)
                r0, r1 = reader.response.components(x), reader.response.components(alternate)
                changed_delay = r0['selected_delay_samples'] != r1['selected_delay_samples']
                measured = {name: tangent(b0[name], b1[name]) for name in b0}
                drift = {name: value[1] for name, value in measured.items()}
            y = torch.tensor(meta['y'][begin:end].tolist(), dtype=torch.long, device=device)
            rx = torch.tensor(meta['rx_i'][begin:end].tolist(), dtype=torch.long, device=device)
            day = torch.tensor(meta['day_i'][begin:end].tolist(), dtype=torch.long, device=device)
            if role == 'L_s':
                l_labels.append(y.cpu())
                for name in b0:
                    l_vectors.setdefault(name, []).append(torch.nn.functional.normalize(b0[name], dim=1).cpu())
                    l_directions.setdefault(name, []).append(measured[name][0].cpu())
            else:
                for name in b0:
                    fraction, valid = direction_overlap(measured[name][0], basis[name])
                    item = overlaps.setdefault(name, dict(sum=0., nonzero=0))
                    item['sum'] += float(fraction[valid].sum())
                    item['nonzero'] += int(valid.sum())
            strata = [('overall', 'ALL', torch.ones(len(x), dtype=torch.bool, device=device))]
            for name, values in (('receiver', rx), ('day', day), ('transmitter', y)):
                strata.extend((name, str(int(k)), values == k) for k in values.unique().tolist())
            for dimension, key, mask in strata:
                entry = groups.setdefault((dimension, key), dict(count=0, raw_correct=0, echo_correct=0,
                    prediction_flips=0, reference_delay_changes=0, raw_accepted=0, accepted_wrong=0,
                    channel_agreement_accepted=0, channel_agreement_wrong=0,
                    branch_drift_sum={name: 0. for name in drift}))
                accepted = confidence >= .95
                transferred = accepted & (confidence1 >= .95) & (raw == echo)
                entry['count'] += int(mask.sum())
                entry['raw_correct'] += int(((raw == y) & mask).sum())
                entry['echo_correct'] += int(((echo == y) & mask).sum())
                entry['prediction_flips'] += int(((raw != echo) & mask).sum())
                entry['reference_delay_changes'] += int((changed_delay & mask).sum())
                entry['raw_accepted'] += int((accepted & mask).sum())
                entry['accepted_wrong'] += int((accepted & (raw != y) & mask).sum())
                entry['channel_agreement_accepted'] += int((transferred & mask).sum())
                entry['channel_agreement_wrong'] += int((transferred & (raw != y) & mask).sum())
                for name, values in drift.items():
                    entry['branch_drift_sum'][name] += float(values[mask].sum())
        if groups['overall', 'ALL']['count'] != contract['counts'][role]:
            raise ValueError('Full source role coverage required')
        if role == 'L_s':
            all_labels = torch.cat(l_labels)
            for name in l_vectors:
                vectors, directions = torch.cat(l_vectors[name]), torch.cat(l_directions[name])
                vectors_basis, rank = identity_basis(vectors, all_labels, classes=len(contract['classes']))
                fraction, valid = direction_overlap(directions, vectors_basis)
                basis[name] = vectors_basis.to(device)
                basis_report[name] = dict(rank=rank, L_nonzero=int(valid.sum()),
                    L_overlap_mean=float(fraction[valid].mean()) if bool(valid.any()) else None)
        else:
            for name, item in overlaps.items():
                basis_report[name].update(V_nonzero=item['nonzero'],
                    V_overlap_mean=item['sum'] / item['nonzero'] if item['nonzero'] else None)
        results = [dict(dimension=dimension, stratum=key, **value,
                        branch_drift_mean={k: v / value['count'] for k, v in value['branch_drift_sum'].items()})
                   for (dimension, key), value in sorted(groups.items())]
        cells = sorted(set(zip(meta['y'].tolist(), meta['rx_i'].tolist(), meta['day_i'].tolist())))
        indices = np.asarray([int(i) for y, rx, day in cells
                    for i in np.flatnonzero((meta['y'] == y) & (meta['rx_i'] == rx) & (meta['day_i'] == day))[:8]], dtype=np.int64)
        if len(cells) != 90 or len(indices) != 720:
            raise ValueError('Fixed source TX/RX/day stratified As subset differs')
        scenes = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
        as_results = {}
        for scene in scenes:
            count = correct = raw_correct = flips = 0
            for begin in range(0, len(indices), 64):
                part = indices[begin:begin + 64]
                x = data.x.index_select(0, tensor(part)).to(device)
                raw = np.asarray(data.x.index_select(0, tensor(part)).tolist(), dtype=np.float64)
                transformed = tensor(as_view(raw, meta, part, scene, role), device)
                y = torch.tensor(meta['y'][part].tolist(), dtype=torch.long, device=device)
                with torch.no_grad():
                    logits0 = reader.classify_features(reader.features(x))
                    logits1 = reader.classify_features(reader.features(transformed))
                    if not bool(torch.isfinite(logits0).all()) or not bool(torch.isfinite(logits1).all()):
                        raise FloatingPointError('Nonfinite source As diagnostic logits')
                    p0, p1 = logits0.argmax(1), logits1.argmax(1)
                count += len(part); correct += int((p1 == y).sum())
                raw_correct += int((p0 == y).sum()); flips += int((p0 != p1).sum())
            as_results[scene] = dict(count=count, correct=correct, raw_correct=raw_correct,
                                     prediction_flips=flips, sample_rule='first 8 per source TX/RX/day')
        totals[role] = dict(bounded_echo=results, original_as_stratified=as_results)
        print('PHYSICAL_SOURCE_DIAGNOSTIC', c['row_id'], role, len(data), flush=True)
    write(out / 'report.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', row_id=c['row_id'],
          source_only=True, target_reads=0, U_reads=0, fitted_updates=0, results=totals, identity_direction_overlap=basis_report,
          overlap_scope='L class-centroid basis estimated descriptively; L is in-sample, V is held-out; not a causal TX subspace',
          interpretation='Received-IQ bounded echo proxy; response delay may jump; no calibrated TX claims',
          seconds=time.monotonic() - started))
    write(out / 'complete.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', row_id=c['row_id'],
          L_s=contract['counts']['L_s'], V=contract['counts']['V'], target_reads=0, U_reads=0))
