"""Source-free RF identifiability counterexamples; never a training augmentation.

All signals and hardware parameters are hand set. The periodic input has the
L-STF frequency support/period but is NOT the standard L-STF sequence or WiSig.
No dataset, checkpoint, label, prediction, or remote input is accepted.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def wl(a, b):
    return np.array([[a, b], [b.conjugate(), a.conjugate()]], dtype=np.complex128)


def counterexamples():
    fs = 25_000_000.0
    n = np.arange(320)
    tones = np.r_[np.arange(-6, 0), np.arange(1, 7)]
    coefficients = np.exp(1j * (0.23 * tones + 0.071 * tones**2))
    s = np.sum(coefficients[:, None] * np.exp(2j * np.pi * tones[:, None] * n / 20), axis=0)
    s /= np.sqrt(np.mean(abs(s)**2))
    cases = []

    def add(name, error, **details):
        cases.append(dict(name=name, maximum_absolute_error=float(error), **details))

    tx, rx, gauge = 1.1*np.exp(.2j), .9*np.exp(-.1j), 1.08*np.exp(.12j)
    add('tx_rx_gain_gauge', np.max(abs(rx*tx*s - (rx/gauge)*(gauge*tx)*s)),
        tx_before=[float(tx.real),float(tx.imag)], tx_after=[float((gauge*tx).real),float((gauge*tx).imag)],
        assumption='Unknown flat complex TX/RX gains; known input does not fix the factorization.')

    ftx, frx, shift = 18_000.0, -12_000.0, 20_000.0
    x = s*np.exp(2j*np.pi*(ftx-frx)*n/fs)
    other = s*np.exp(2j*np.pi*((ftx+shift)-(frx+shift))*n/fs)
    add('tx_rx_oscillator_gauge', np.max(abs(x-other)),
        first_pair_hz=[ftx,frx], second_pair_hz=[ftx+shift,frx+shift], observed_offset_hz=ftx-frx,
        assumption='No calibrated/common oscillator reference; received CFO is a difference.')

    wt = wl(1+.02j, .035+.012j)
    wr = wl(.96-.03j, -.016+.009j)
    q = wl(1.04*np.exp(.1j), .02+.01j)
    alternate_t = q@wt
    alternate_r = wr@np.linalg.inv(q)
    augmented = np.stack([s,s.conjugate()])
    add('tx_rx_iq_factorization_gauge', np.max(abs(wr@wt@augmented-alternate_r@alternate_t@augmented)),
        distinct_tx_image_coefficient_difference=float(abs(wt[0,1]-alternate_t[0,1])),
        cascade_matrix_error=float(np.max(abs(wr@wt-alternate_r@alternate_t))),
        assumption='Invertible memoryless widely-linear TX/RX chains, unity channel; a valid counterexample, not every channel.')

    def cubic(z):
        return z+(-.02+.03j)*z*abs(z)**2
    # TX=cubic,RX=identity and TX=identity,RX=cubic yield the same observation.
    tx_distorted = cubic(s)
    rx_distorted = cubic(s)
    add('nonlinearity_tx_rx_attribution', np.max(abs(tx_distorted-rx_distorted)),
        cubic_coefficient=[-.02,.03], distortion_rms=float(np.sqrt(np.mean(abs(tx_distorted-s)**2))),
        assumption='Unity channel and identical admissible memoryless cubic at either chain end; attribution cannot be universal.')

    period = s[:20]
    linear_spectrum = np.fft.fft(period)
    nonlinear_spectrum = np.fft.fft(period*abs(period)**2)
    occupied = np.zeros(20,dtype=bool)
    occupied[tones % 20] = True
    # A causal 20-tap FIR can realize this circular response in periodic steady state.
    response = np.zeros(20,dtype=np.complex128)
    response[occupied] = nonlinear_spectrum[occupied]/linear_spectrum[occupied]
    impulse = np.fft.ifft(response)
    circular_output = sum(impulse[m]*np.roll(period,m) for m in range(20))
    overlapping_distortion = np.fft.ifft(np.where(occupied,nonlinear_spectrum,0))
    fraction = float(np.sum(abs(nonlinear_spectrum[occupied])**2)/np.sum(abs(nonlinear_spectrum)**2))
    add('periodic_nonlinearity_linear_subspace_overlap', np.max(abs(circular_output-overlapping_distortion)),
        overlapping_distortion_energy_fraction=fraction, outside_occupied_energy_fraction=1-fraction,
        input='Hand-set 12-tone period-20 signal; correct support, not standard L-STF coefficients.',
        assumption='Unknown arbitrary 20-tap complex FIR in periodic steady state; outside-support distortion is not absorbed by this construction.')

    rms = lambda z: z/np.sqrt(np.mean(abs(z)**2))
    add('packet_rms_absolute_gain_loss', np.max(abs(rms(s)-rms(1.7*s))),
        positive_gain_ratio=1.7, assumption='Ideal noiseless positive scalar gain; SNR/saturation effects are outside this counterexample.')

    def zero_delayed(z, delay):
        return np.r_[np.zeros(delay,dtype=z.dtype),z[:len(z)-delay]] if delay else z.copy()
    add('periodic_delay_alias_after_onset', np.max(abs(s[20:]-zero_delayed(s,20)[20:])),
        delays_samples=[0,20], delays_ns=[0,800], onset_difference=float(np.max(abs(s[:20]-zero_delayed(s,20)[:20]))),
        assumption='Periodic steady-state segment only; onset/L-LTF transition can break this ambiguity.')

    keep = np.r_[np.arange(38,64), np.arange(1,27)]
    rejected = np.exp(2j*np.pi*27*np.arange(64)/64)
    spectrum = np.fft.fft(rejected)
    masked = np.zeros_like(spectrum)
    masked[keep] = spectrum[keep]
    add('official_equalization_rejected_fft_bin', np.max(abs(np.fft.ifft(masked))),
        fft_size=64, rejected_signed_bin=27, retained_signed_bins='-26..-1,1..26',
        assumption='Isolates the official pre-CFO-reapplication spectral mask. No full MATLAB/noise/equalization/resampler emulation.')

    tolerance = 1e-11
    if any(r['maximum_absolute_error'] >= tolerance for r in cases):
        raise AssertionError('A counterexample numerical equality failed')
    if cases[2]['distinct_tx_image_coefficient_difference'] <= 1e-3:
        raise AssertionError('The IQ factorizations must differ')
    if not 0 < fraction < 1:
        raise AssertionError('Must retain both overlap and distinguishable outside-support distortion')
    if cases[6]['onset_difference'] <= .1:
        raise AssertionError('Onset must distinguish the delays')
    return dict(status='VERIFIED', scope='SYNTHETIC_IDENTIFIABILITY_DIAGNOSTIC_ONLY',
                formal_dataset_access=False, target_access=False, checkpoint_access=False,
                training=False, training_augmentation=False, inference_method=False,
                dtype='complex128/float64', sample_rate_hz=fs, tolerance=tolerance, cases=cases,
                conclusion='Received-IQ CE recognition is possible; unique universal TX hardware recovery is not established. Counterexamples do not prove impossibility under additional calibrated assumptions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    files = [args.output/name for name in ('resolved_config.json','counterexamples.json')]
    if any(path.exists() for path in files):
        raise FileExistsError('Preserve prior diagnostic artifacts; use a new registered output')
    config = dict(command=sys.argv, python=sys.executable, platform=platform.platform(),
                  dataset=None, checkpoint=None, random_seed=None, model=None,
                  data_role='hand_set_synthetic_only', dtype='complex128/float64',
                  note='No RNG, optimizer, model initialization, split, support, augmentation or target evaluation.')
    files[0].write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    start=time.perf_counter()
    results=counterexamples()
    results['elapsed_seconds']=time.perf_counter()-start
    files[1].write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=results['status'], cases=len(results['cases']),
                         maximum_equality_error=max(r['maximum_absolute_error'] for r in results['cases']),
                         output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    main()
