"""L-only channel-moment diagnostics for cached real shallow/middle maps.

This module performs no model forward, fitting, normalization of IQ, or file I/O.
The caller must verify that all supplied physical IDs belong to the legal L_s
cache. The feature maps are real learned activations, never received IQ.
"""
import math
import numbers

import torch

from experiments.cvs_early_geometry.multilevel import SOURCE_RX, weak_mixstyle


SOURCE_DAYS = (1, 2, 3)
SOURCE_LABELS = tuple(range(6))
ROUNDS_PER_DAY = 12
MOMENT_EPS = 1e-6


def _metadata(value, name, n, allowed):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != n:
        raise ValueError(f'{name} must be a length-N tensor')
    if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f'{name} must contain integer metadata')
    output = value.detach().cpu().to(torch.int64)
    if set(output.tolist()) != set(allowed):
        raise ValueError(f'{name} must cover exactly {allowed}')
    return output


def _validate(maps, labels, receivers, days, ids, seed):
    if (not isinstance(maps, torch.Tensor) or maps.ndim != 3 or maps.is_complex()
            or not maps.is_floating_point() or maps.shape[0] == 0
            or maps.shape[1] == 0 or maps.shape[2] < 2):
        raise ValueError('Expected nonempty real floating maps [N,C,T], T >= 2')
    if not torch.isfinite(maps).all():
        raise ValueError('Nonfinite cached feature maps')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= int(seed) < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    n = len(maps)
    y = _metadata(labels, 'labels', n, SOURCE_LABELS)
    rx = _metadata(receivers, 'receivers', n, SOURCE_RX)
    day = _metadata(days, 'days', n, SOURCE_DAYS)
    if not isinstance(ids, (list, tuple)) or len(ids) != n:
        raise ValueError('ids must contain one physical ID per map')
    if any(isinstance(pid, bool) or not isinstance(pid, (str, numbers.Integral))
           or (isinstance(pid, str) and not pid.strip())
           or (isinstance(pid, numbers.Integral) and pid < 0) for pid in ids):
        raise ValueError('Physical IDs must be nonempty strings or nonnegative integers')
    pid = [str(v) if isinstance(v, str) else int(v) for v in ids]
    if len(set(pid)) != n:
        raise ValueError('Repeated physical IDs in source L')
    members = {}
    for d in SOURCE_DAYS:
        for tx in SOURCE_LABELS:
            for r in SOURCE_RX:
                index = torch.where((y == tx) & (rx == r) & (day == d))[0]
                if len(index) < 2:
                    raise ValueError(f'L cell TX={tx}, RX={r}, day={d} requires >=2 distinct PID')
                members[d, tx, r] = index
    return y, rx, day, pid, members


def _moments(value):
    # Chunking keeps FP64 diagnostic accumulation bounded for full-L caches.
    mus, logs = [], []
    for chunk in value.detach().split(256):
        data = chunk.to(dtype=torch.float64)
        mus.append(data.mean(-1).cpu())
        logs.append((data.var(-1, unbiased=False) + MOMENT_EPS).sqrt().log().cpu())
    return dict(mu=torch.cat(mus), log_sigma=torch.cat(logs))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _conditional(moment, members, day, tx):
    c = moment.shape[1]
    groups = [moment[members[day, tx, r]] for r in SOURCE_RX]
    means = torch.stack([g.mean(0) for g in groups])
    center = means.mean(0)
    within = [(g - means[i]).square().mean() for i, g in enumerate(groups)]
    between = float((means - center).square().mean())
    within_mse = float(torch.stack(within).mean())
    # An unbiased sample-variance estimate of centroid noise; subtracting the
    # common RX grand mean removes 1/R of this noise under independent samples.
    mean_noise = float(torch.stack([v / (len(g) - 1) for v, g in zip(within, groups)]).mean())
    noise_in_between = (len(SOURCE_RX) - 1) / len(SOURCE_RX) * mean_noise
    return dict(between_rx_mse=between, within_rx_mse=within_mse,
                between_over_within=_ratio(between, within_mse),
                receiver_mean_noise_mse_estimate=mean_noise,
                noise_contribution_to_between_mse_estimate=noise_in_between,
                between_minus_noise_mse=between - noise_in_between,
                between_minus_noise_clipped_mse=max(0., between - noise_in_between),
                between_denominator=len(SOURCE_RX) * c,
                within_aggregation='equal mean over 5 RX; each RX uses n_rx*C',
                receivers=[dict(rx=int(r), count=len(g), coordinates=c,
                                within_denominator=len(g)*c,
                                within_mse=float(v), mean_noise_mse_estimate=float(v)/(len(g)-1))
                           for r, g, v in zip(SOURCE_RX, groups, within)])


def _displacement(before, after):
    delta = after - before
    mse = float(delta.square().mean())
    energy = float(before.square().mean())
    return dict(delta_mse=mse, delta_rms=math.sqrt(mse), before_mean_square=energy,
                relative_rms=math.sqrt(mse / energy) if energy > 0 else None,
                denominator=before.numel())


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


@torch.no_grad()
def style_audit(maps, labels, receivers, days, ids, seed):
    """Return JSON-safe statistics and 36 fixed L-only MixStyle measurements.

    Full-L moment dispersion conditions on TX and day. Between-RX statistics
    weight each of the five RX equally. Within-RX statistics first divide by
    n_rx*C, then average RX. No p-values or causal RX-identification claim is
    implied by the finite-sample noise correction.

    Each of 3 days has 12 independent fixed-seed rounds of 6 TX x 5 RX x 2 PID.
    The two maps' separate calls use identical packet sampling and donor RNG
    seeds. MixStyle is invoked unchanged at p=.5, alpha=.1, eta=.3; its actual
    enabled/disabled outcome is retained, with no retry of disabled rounds.
    """
    y, rx, day, pid, members = _validate(maps, labels, receivers, days, ids, seed)
    moments = _moments(maps)
    conditional = []
    for d in SOURCE_DAYS:
        for tx in SOURCE_LABELS:
            counts = {str(r): len(members[d, tx, r]) for r in SOURCE_RX}
            conditional.append(dict(tx=tx, day=d, count=sum(counts.values()), rx_counts=counts,
                                    moments={k: _conditional(v, members, d, tx)
                                             for k, v in moments.items()}))
    rounds = []
    for di, d in enumerate(SOURCE_DAYS):
        for ri in range(ROUNDS_PER_DAY):
            ordinal = di * ROUNDS_PER_DAY + ri
            sample_seed = (int(seed) + 700001 + ordinal * 99991) % 2**63
            mix_seed = (int(seed) + 900001 + ordinal * 701) % 2**63
            generator = torch.Generator().manual_seed(sample_seed)
            index = torch.cat([members[d, tx, r][torch.randperm(len(members[d, tx, r]),
                              generator=generator)[:2]] for tx in SOURCE_LABELS for r in SOURCE_RX])
            selected = index.tolist()
            batch = maps.detach().index_select(0, index.to(maps.device))
            by, brx, bd = (v[index] for v in (y, rx, day))
            bid = [pid[i] for i in selected]
            mixed, detail = weak_mixstyle(batch, by, brx, bd, bid, mix_seed, p=.5, alpha=.1, eta=.3)
            donors = detail['donors']
            if detail['enabled']:
                if len(donors) != 60 or any(by[i] != by[j] or brx[i] == brx[j]
                        or bd[i] != bd[j] or bid[i] == bid[j] for i, j in enumerate(donors)):
                    raise ValueError('Existing MixStyle returned an illegal L donor')
            elif donors:
                raise ValueError('Disabled MixStyle unexpectedly returned donors')
            before = {k: v[index] for k, v in moments.items()}
            after = _moments(mixed)
            clean = batch.to(dtype=torch.float64)
            delta = mixed.to(dtype=torch.float64) - clean
            norms = clean.flatten(1).norm(dim=1)
            relative = delta.flatten(1).norm(dim=1) / norms.clamp_min(1e-6)
            record = dict(day=d, round=ri+1, sample_seed=sample_seed, mix_seed=mix_seed,
                          count=60, distinct_pid=60, sampling='6 TX x 5 RX x 2; one day',
                          sample_indices=selected, physical_ids=bid, enabled=bool(detail['enabled']),
                          eligible_packets=int(detail['eligible_packets']),
                          changed_packets=int(detail['changed_packets']),
                          changed_ratio=float(detail['changed_ratio']),
                          movement_mean=float(detail['movement_mean']),
                          movement_max=float(detail.get('movement_max', 0.)),
                          donor_batch_indices=donors, donor_physical_ids=[bid[j] for j in donors],
                          donor_rules_verified=True,
                          moments={k: _displacement(before[k], after[k]) for k in moments},
                          feature=dict(delta_mse=float(delta.square().mean()),
                                       relative_l2_mean=float(relative.mean()),
                                       relative_l2_median=float(relative.median()),
                                       relative_l2_p90=float(torch.quantile(relative, .9)),
                                       relative_l2_max=float(relative.max()),
                                       native_relative_l2_mean=float(detail['delta_ratio']),
                                       zero_input_norm_packets=int((norms == 0).sum()),
                                       denominator=batch.numel(), norm_floor=1e-6))
            rounds.append(record)
    enabled = [record for record in rounds if record['enabled']]
    moment_summary = {}
    for key in moments:
        cells = [record['moments'][key] for record in conditional]
        between = _mean(cells, 'between_rx_mse')
        within = _mean(cells, 'within_rx_mse')
        moment_summary[key] = dict(between_rx_mse_mean=between, within_rx_mse_mean=within,
                                  ratio_of_mean_between_to_mean_within=_ratio(between, within),
                                  noise_in_between_mse_mean=_mean(cells, 'noise_contribution_to_between_mse_estimate'),
                                  actual_delta_mse_mean=_mean([r['moments'][key] for r in rounds], 'delta_mse'),
                                  enabled_delta_mse_mean=_mean([r['moments'][key] for r in enabled], 'delta_mse') if enabled else None)
    return dict(schema='e20_L_moment_style_audit_v1', source_role='L_s', seed=int(seed),
                target_access=False, V_access=False, U_access=False, U_truth=False,
                fitting=False, input_kind='real learned feature maps; not IQ',
                input_shape=list(maps.shape), input_dtype=str(maps.dtype),
                moment_definition=dict(mu='temporal mean', log_sigma='log(sqrt(temporal population variance + 1e-6))',
                                       temporal_variance_denominator=int(maps.shape[2]), eps=MOMENT_EPS,
                                       accumulation='float64; no normalization or map mutation'),
                mixstyle=dict(implementation='experiments.cvs_early_geometry.multilevel.weak_mixstyle',
                              p=.5, alpha=.1, strength=.3, rounds_per_day=ROUNDS_PER_DAY,
                              source_days=list(SOURCE_DAYS), samples_per_TX_RX=2,
                              donor='same TX, same day, different RX and physical ID',
                              sampling_within_round='without replacement', sampling_across_rounds='independent; reuse allowed'),
                summary=dict(count=len(maps), unique_pid=len(pid), conditional_cells=len(conditional),
                             rounds=len(rounds), enabled_rounds=len(enabled),
                             packet_occurrences=len(rounds)*60,
                             changed_packet_occurrences=sum(r['changed_packets'] for r in rounds),
                             changed_ratio=_mean(rounds, 'changed_ratio'),
                             feature_relative_l2_mean=_mean([r['feature'] for r in rounds], 'relative_l2_mean'),
                             feature_relative_l2_enabled_mean=_mean([r['feature'] for r in enabled], 'relative_l2_mean') if enabled else None,
                             moments=moment_summary, conditional_aggregation='equal mean over 18 TX/day cells'),
                conditional_cells=conditional, rounds=rounds,
                limitations=['Caller verifies L_s role IDs against the existing source contract.',
                             'Between-RX dispersion is conditional association, not isolated causal hardware effect.',
                             'Centroid-noise correction assumes independent packet observations within each cell.',
                             'Cached maps measure local feature perturbation, not downstream identity accuracy.'])
