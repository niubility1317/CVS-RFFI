"""Source-only waveform diagnostics. No TX labels, F outputs, or target inputs.

Clean-dependent gain/delay calculations below are explicit diagnostics. They
are never exported as restoration predictions or used as deployed operators.
"""
import argparse
import json
from pathlib import Path
import numpy as np

SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
PERIODS = (16, 32, 64, 128)


def complex_array(value):
    return value[:, 0].astype(np.float64)+1j*value[:, 1].astype(np.float64)


def describe(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite diagnostic value')
    return dict(mean=float(value.mean()), median=float(np.median(value)),
                p10=float(np.quantile(value, .1)), p90=float(np.quantile(value, .9)))


def normalized_error(estimate, clean):
    return np.mean(abs(estimate-clean)**2, axis=-1)/np.maximum(np.mean(abs(clean)**2, axis=-1), 1e-4)


def coherence(a, b):
    return abs(np.sum(a.conj()*b, axis=-1))/np.maximum(
        np.sqrt(np.sum(abs(a)**2, axis=-1)*np.sum(abs(b)**2, axis=-1)), 1e-12)


def periodicity(clean):
    return {str(p): describe(coherence(clean[:, :-p], clean[:, p:])) for p in PERIODS}


def advance(received, fill):
    """Fixed known latency removal, with received-only boundary completion."""
    prefix = received[:, 24:]
    if fill == 'zero':
        tail = np.zeros_like(received[:, :24])
        selected = None
    elif fill == 'reflect':
        tail = prefix[:, -25:-1][:, ::-1]
        selected = None
    elif fill == 'periodic':
        scores = np.stack([coherence(prefix[:, :-p], prefix[:, p:]) for p in PERIODS], -1)
        selected = np.asarray(PERIODS)[scores.argmax(-1)]
        tail = np.empty_like(received[:, :24])
        # Recursive repetition also supports period16 < missing length24.
        temporary = np.concatenate((prefix, tail), -1)
        for offset in range(24):
            temporary[:, 232+offset] = temporary[np.arange(len(prefix)), 232+offset-selected]
        tail = temporary[:, 232:]
    else:
        raise ValueError('Declared received-only boundary rule required')
    return np.concatenate((prefix, tail), -1), selected


def waveform_metrics(estimate, clean):
    result = dict(full_nmse=describe(normalized_error(estimate, clean)),
                  observed_prefix232_nmse=describe(normalized_error(estimate[:, :232], clean[:, :232])),
                  unavailable_tail24_nmse=describe(normalized_error(estimate[:, 232:], clean[:, 232:])))
    target = np.fft.fft(clean, norm='ortho')
    spectrum = np.fft.fft(estimate, norm='ortho')
    power = abs(target)**2
    order = np.argsort(power, axis=-1)
    relative = abs(spectrum-target)**2/np.maximum(power, .01)
    result['lowest_power_quartile_complex_relative_error_capped_denominator'] = describe(
        np.take_along_axis(relative, order[:, :64], axis=-1).mean(-1))
    result['highest_power_quartile_complex_relative_error_capped_denominator'] = describe(
        np.take_along_axis(relative, order[:, -64:], axis=-1).mean(-1))
    return result


def diagnostic_gain(estimate, clean):
    e, x = estimate[:, :232], clean[:, :232]
    gain = np.sum(e.conj()*x, -1)/np.maximum(np.sum(abs(e)**2, -1), 1e-12)
    return dict(oracle_only=True, absolute_correction_phase_deg=describe(abs(np.angle(gain))*180/np.pi),
                correction_magnitude=describe(abs(gain)), prefix_coherence=describe(coherence(e, x)),
                aligned_prefix_nmse=describe(normalized_error(gain[:, None]*e, x)))


def analyze(source_cache, observations, output):
    from .data import IQSource
    if output.exists():
        raise FileExistsError('Immutable diagnostic output exists')
    roles = {}
    for role in ('L_s', 'U_s'):
        source = IQSource(source_cache, role)
        clean = complex_array(source.iq)
        roles[role] = dict(count=len(clean), coherence_by_lag=periodicity(clean))
    clean = complex_array(np.load(observations/'validation/clean.npy', allow_pickle=False))
    ids = np.load(observations/'validation/physical_ids.npy', allow_pickle=False)
    if len(clean) != 1800 or len(set(ids.tolist())) != 1800:
        raise ValueError('Expected original fixed source V waveform panel')
    source_v = IQSource(source_cache, 'V')
    if not set(ids.tolist()) <= set(source_v.ids.tolist()):
        raise ValueError('No target or training record may enter source V diagnostics')
    views = {}
    for scene in ('clean', *SCENES):
        received = complex_array(np.load(observations/'validation'/(scene+'.npy'), allow_pickle=False))
        restored = complex_array(np.load(observations/'unet'/(scene+'.npy'), allow_pickle=False))
        row = dict(identity=waveform_metrics(received, clean),
                   unet=waveform_metrics(restored, clean), unet_oracle_diagnostic=diagnostic_gain(restored, clean))
        if scene != 'clean':
            baselines = {}
            for rule in ('zero', 'reflect', 'periodic'):
                value, selected = advance(received, rule)
                item = dict(metrics=waveform_metrics(value, clean), oracle_diagnostic=diagnostic_gain(value, clean))
                if selected is not None:
                    item['selected_period_counts'] = {str(p): int((selected == p).sum()) for p in PERIODS}
                baselines[rule] = item
            row['fixed24_advance_received_only'] = baselines
        views[scene] = row
    result = dict(status='SOURCE_WAVEFORM_DIAGNOSTIC_COMPLETE', TX_labels_read=0, F_outputs_read=0,
                  target_reads=0, source_roles=roles, source_V_count=len(clean), views=views,
                  known_numerical_latency=24, available_prefix=232,
                  oracle_metrics_are_diagnostic_only=True,
                  warnings=['No TX identity claim from waveform metrics',
                            'Fixed shift controls apply to known augmented source scenes only; no oracle deployment router',
                            'Prefix232 describes nominal missing tail; noncausal FIR support and multipath affect exact observability'])
    output.mkdir(parents=True)
    with (output/'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source_cache, args.observations, args.output)
    print(json.dumps(dict(status=result['status'],TX_labels_read=0,target_reads=0,
                         source_counts={role:v['count'] for role,v in result['source_roles'].items()})))


if __name__ == '__main__':
    main()
