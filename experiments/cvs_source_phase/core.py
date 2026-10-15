"""Circular source statistics, without phase unwrapping or fitted classifiers."""
import numpy as np
from . import design as d

# Known L-STF coefficients copied exactly from the already verified project
# known_excitation.py. Frequency unit is Fs/20=1.25MHz, not 312.5kHz.
K = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
SIGNS = np.array([1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1])


def closure_matrix():
    rows = []
    for k in (-5, -4, -3, -2, 2, 3, 4, 5):
        rows.append({k - 1: 1, k: -2, k + 1: 1})
    # Bridge the missing DC: together the ten rows span all 12 phases
    # modulo one constant and one frequency-linear gauge.
    rows += [{-2: 1, 2: 1, -1: -1, 1: -1}, {-1: 1, 3: 1, 1: -2}]
    return np.array([[row.get(int(k), 0) for k in K] for row in rows], dtype=np.float64)


A = closure_matrix()


def closures(h):
    return np.exp(1j * (np.angle(h) @ A.T))


def observe(iq, window):
    x = np.asarray(iq, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (2, 256) or not np.isfinite(x).all():
        raise ValueError('Finite received IQ Bx2x256 required')
    lo, hi = window
    if tuple(window) not in d.WINDOWS:
        raise ValueError('Unregistered observation window')
    w = (x[:, 0] + 1j * x[:, 1])[:, lo:hi]
    cross = (w[:, 20:] * w[:, :-20].conjugate()).sum(1)
    denom = np.sqrt((abs(w[:, 20:])**2).sum(1) * (abs(w[:, :-20])**2).sum(1))
    coherence = abs(cross) / np.maximum(denom, 1e-24)
    omega = np.angle(cross) / 20
    corrected = w * np.exp(-1j * omega[:, None] * np.arange(80))
    cycles = np.fft.fft(corrected.reshape(-1, 4, 20), axis=2)
    mean = cycles.mean(1)
    occupied = np.mod(K, 20)
    tone = mean[:, occupied]
    noise_of_mean = (abs(cycles[:, :, occupied] - tone[:, None])**2).sum(1) / 12
    ratio = abs(tone)**2 / np.maximum(noise_of_mean, 1e-24)
    fraction = (abs(tone)**2).sum(1) / np.maximum((abs(mean)**2).sum(1), 1e-24)
    # Known coefficient division removes excitation phase, never TX hardware.
    h = tone / (SIGNS * (1 + 1j))[None]
    value = closures(h)
    odd = closures(cycles[:, (0, 2), :].mean(1)[:, occupied] / (SIGNS * (1 + 1j))[None])
    even = closures(cycles[:, (1, 3), :].mean(1)[:, occupied] / (SIGNS * (1 + 1j))[None])
    limits = d.config()['availability']
    good_tone = ratio >= limits['min_cycle_repeatability_ratio']
    eligible = np.stack([good_tone[:, row != 0].all(1) for row in A], axis=1)
    eligible &= ((coherence >= limits['min_lag20_coherence']) &
                 (fraction >= limits['min_occupied_fraction']))[:, None]
    output = dict(value=value, eligible=eligible, split_cosine=(odd * even.conjugate()).real,
        coherence=coherence, occupied_fraction=fraction, tone_repeatability_ratio=ratio,
        wrapped_CFO_hz=omega * 25000000 / (2 * np.pi))
    if any(not np.isfinite(v).all() for v in output.values()):
        raise FloatingPointError('Nonfinite source phase statistic')
    return output


def phase_audit(observation, tx, rx, day):
    tx, rx, day = (np.asarray(v, dtype=int) for v in (tx, rx, day))
    if len(tx) != 6300 or set(tx) != set(range(6)) or set(rx) != set(d.RX) or set(day) != {1, 2, 3}:
        raise ValueError('Only fixed L6300 TX/RX/day roles permitted')
    value, eligible = observation['value'], observation['eligible']
    if value.shape != (6300, 10) or eligible.shape != value.shape:
        raise ValueError('Fixed ten-coordinate observations required')
    cells, counts, means, split = [], [], [], []
    for t in range(6):
        for day_value in (1, 2, 3):
            for r in d.RX:
                mask = (tx == t) & (day == day_value) & (rx == r)
                if mask.sum() != 70:
                    raise ValueError('Each L TX/RX/day cell must contain exactly 70 packets')
                n = eligible[mask].sum(0)
                mean = (value[mask] * eligible[mask]).sum(0) / np.maximum(n, 1)
                repeat = (observation['split_cosine'][mask] * eligible[mask]).sum(0) / np.maximum(n, 1)
                cells.append(dict(TX=t, day=day_value, RX=r, packets=70, eligible=n.tolist(),
                    mean_real=mean.real.tolist(), mean_imag=mean.imag.tolist(), split_cosine=repeat.tolist()))
                counts.append(n); means.append(mean); split.append(repeat)
    counts = np.array(counts).reshape(6, 3, 5, 10)
    means = np.array(means).reshape(6, 3, 5, 10)
    split = np.array(split).reshape(6, 3, 5, 10)
    folds = []; prediction_norm = []
    for t in range(6):
        for v in range(3):
            train = means[np.arange(6) != t][:, np.arange(3) != v]
            norm = train.mean((0, 1)); prediction_norm.append(abs(norm))
            prediction = norm / np.maximum(abs(norm), 1e-24)
            pooled = train.mean((0, 1, 2)); pooled /= np.maximum(abs(pooled), 1e-24)
            held = means[t, v]
            baseline = 1 - (held * pooled.conjugate()[None]).real
            risk = 1 - (held * prediction.conjugate()).real
            gains = (baseline - risk).mean(1)
            wrong = [float((baseline - (1 - (held * np.roll(prediction, shift, axis=0).conjugate()).real)).mean()) for shift in (1, 2, 3, 4)]
            folds.append(dict(held_TX=t, held_day=v + 1, train_TX=[i for i in range(6) if i != t],
                train_days=[i for i in (1, 2, 3) if i != v + 1],
                fit_mean_real=norm.real.tolist(), fit_mean_imag=norm.imag.tolist(),
                baseline_risk=float(baseline.mean()), RX_risk=float(risk.mean()),
                gain=float(gains.mean()), per_RX_gain=gains.tolist(), wrong_RX_gains=wrong))
    gain = float(np.mean([f['gain'] for f in folds]))
    per_rx = np.mean([f['per_RX_gain'] for f in folds], axis=0)
    wrong = np.mean([f['wrong_RX_gains'] for f in folds], axis=0)
    split_rx = split.mean((0, 1, 3))
    norm_min = float(np.min(prediction_norm))
    criterion = d.config()['criterion']; availability = d.config()['availability']
    checks = dict(coverage=bool((counts >= availability['min_eligible_per_TX_RX_day_coordinate']).all()),
        risk_gain=bool(gain >= criterion['min_risk_gain']),
        fold_repeatability=sum(f['gain'] > 0 for f in folds) >= criterion['min_positive_folds'],
        all_RX_nonnegative=bool((per_rx >= criterion['min_RX_risk_gain']).all()),
        wrong_RX_separation=bool(gain - max(wrong) >= criterion['min_gain_over_best_wrong_RX']),
        fitted_direction_concentration=bool(norm_min >= criterion['min_fit_resultant']),
        cycle_repeatability=bool((split_rx >= criterion['min_RX_split_cosine']).all()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, mean_risk_gain=gain,
        positive_folds=sum(f['gain'] > 0 for f in folds), per_RX_risk_gain=dict(zip(map(str, d.RX), per_rx.tolist())),
        wrong_RX_mean_gains=wrong.tolist(), gain_over_best_wrong_RX=gain - float(max(wrong)),
        minimum_fit_resultant=norm_min, per_RX_split_cosine=dict(zip(map(str, d.RX), split_rx.tolist())),
        minimum_cell_coordinate_eligible=int(counts.min()), eligible_fraction=float(eligible.mean()),
        CFO_summary=dict(mean=float(observation['wrapped_CFO_hz'].mean()),
            near_alias_boundary_count=int((abs(observation['wrapped_CFO_hz']) > 562500).sum())),
        folds=folds, cells=cells,
        interpretation='Association screen only; no unique IQ convention/RX hardware identification, no classifier benefit')
