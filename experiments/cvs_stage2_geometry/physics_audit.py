"""Fixed source-IQ physical observations, with explicit identifiability limits.

No labels, target I/O, model parameters, cross-packet fits, or estimated noise
covariance enter this module. Fourier coefficients are expanded before ideal
receiver band limitation and final sampling; no sampled polynomial is used as
the physical nonlinear basis.
"""
import math

import torch


SAMPLE_RATE = 25_000_000.
PERIOD = 20
HARMONICS = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
SIGNS = (1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1)
STRICT_BAND_HZ = 10_000_000.
EPS = 1e-12


def _convolve(a, b):
    result = {}
    for k, v in a.items():
        for j, w in b.items():
            result[k + j] = result.get(k + j, 0j) + v * w
    return result


def fourier_coefficients(order=1, strict_inband=False):
    """Continuous-periodic Fourier convolution, harmonic spacing 1.25 MHz."""
    if order not in (1, 3, 5):
        raise ValueError('Only the declared first, third and fifth orders exist')
    base = {k: sign * complex(1, 1) / math.sqrt(24)
            for k, sign in zip(HARMONICS, SIGNS)}
    conjugate = {-k: v.conjugate() for k, v in base.items()}
    result = base
    for _ in range((order - 1) // 2):
        result = _convolve(_convolve(result, base), conjugate)
    if strict_inband:
        result = {k: v for k, v in result.items() if abs(k * SAMPLE_RATE / PERIOD) < STRICT_BAND_HZ}
    return result


def sample_series(coefficients, samples):
    """Sample an analytic Fourier series on arbitrary final-rate coordinates."""
    samples = torch.as_tensor(samples)
    if samples.dtype not in (torch.float32, torch.float64):
        samples = samples.double()
    cdtype = torch.complex128 if samples.dtype == torch.float64 else torch.complex64
    harmonics = torch.tensor(list(coefficients), device=samples.device, dtype=samples.dtype)
    values = torch.tensor(list(coefficients.values()), device=samples.device, dtype=cdtype)
    return torch.exp(2j * math.pi * samples[..., None] * harmonics / PERIOD) @ values


def template(samples, delay=0.):
    return sample_series(fourier_coefficients(), torch.as_tensor(samples) - delay)


def strict_lowpass(z):
    """Ideal |f|<10 MHz mask for an explicitly periodic finite-window audit.

    This is an analysis assumption, not an estimate of the actual RX filter.
    Window transients and synchronization leakage remain possible confounds.
    """
    f = torch.fft.fftfreq(z.shape[-1], d=1 / SAMPLE_RATE, device=z.device, dtype=z.real.dtype)
    return torch.fft.ifft(torch.fft.fft(z, dim=-1) * (f.abs() < STRICT_BAND_HZ), dim=-1)


def _range(matrix, relative_tolerance=1e-10):
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(1e-12, float(singular[0]) * relative_tolerance) if len(singular) else 1e-12
    return u[:, singular > threshold], singular, threshold


def _residual(z, nuisance):
    # Row observations, column orthonormal basis. Complex conjugates matter.
    return z - (z @ nuisance.conj()) @ nuisance.T


def nonlinear_basis(length, device='cpu', dtype=torch.float64):
    """Conservative complex nuisance span and non-inverted residual SVD.

    Twelve independently free linear tones plus DC absorb arbitrary static
    in-band linear response. Phase and delay tangents are included explicitly
    but already lie in that span. A local CFO tangent is an extra column. Its
    complex span is conservative relative to the one-real-parameter CFO model.
    Sigma is unknown: the declared whitening is identity, not fitted noise.
    """
    if length not in (20, 80):
        raise ValueError('Basis length must be one or four known periods')
    t = torch.arange(length, device=device, dtype=dtype)
    h = torch.tensor(HARMONICS, device=device, dtype=dtype)
    s = template(t)
    tones = torch.exp(2j * math.pi * t[:, None] * h[None] / PERIOD)
    derivative = sample_series({k: v * (2j * math.pi * k / PERIOD)
                                for k, v in fourier_coefficients().items()}, t)
    columns = torch.cat((tones, torch.ones_like(s[:, None]), 1j * s[:, None],
                         derivative[:, None], (1j * (t - t.mean()) / length * s)[:, None]), 1)
    columns = strict_lowpass(columns.T).T
    columns = columns / columns.norm(dim=0, keepdim=True).clamp_min(EPS)
    nuisance, nuisance_singular, nuisance_threshold = _range(columns)
    physical = torch.stack([sample_series(fourier_coefficients(p, True), t) for p in (3, 5)], 1)
    residual = physical - nuisance @ (nuisance.conj().T @ physical)
    identifiable, singular, threshold = _range(residual)
    # Fix each singular vector's arbitrary complex phase for portable features.
    for j in range(identifiable.shape[1]):
        anchor = identifiable[:, j][identifiable[:, j].abs().argmax()]
        identifiable[:, j] *= anchor.conj() / anchor.abs()
    return dict(nuisance=nuisance, identifiable=identifiable, residual_design=residual,
                nonlinear_singular=singular, nuisance_singular=nuisance_singular,
                rank=int(identifiable.shape[1]), nuisance_rank=int(nuisance.shape[1]),
                relative_threshold=1e-10, absolute_threshold=threshold,
                nuisance_absolute_threshold=nuisance_threshold)


def projective_time(y, absolute_floor=EPS):
    """Two-window rank-one projectors at each tone; zeros have a validity mask.

    Input [N,2,K] complex spectral vectors. Output is the three real independent
    coordinates (P00-P11, 2 Re(P01), 2 Im(P01)) of each trace-one projector.
    A nonzero shared complex gain at a tone cancels exactly above the floor.
    """
    if not torch.is_complex(y) or y.ndim != 3 or y.shape[1] != 2:
        raise ValueError('Expected complex [N,2,K] early/reference spectra')
    energy = y.abs().square().sum(1)
    valid = energy > absolute_floor
    denom = torch.where(valid, energy, torch.ones_like(energy))
    cross = y[:, 0] * y[:, 1].conj()
    features = torch.stack(((y[:, 0].abs().square() - y[:, 1].abs().square()) / denom,
                            2 * cross.real / denom, 2 * cross.imag / denom), -1)
    features = features * valid[..., None]
    return features.flatten(1), valid, energy


def repeat_energy(cycles):
    """Four-cycle descriptive energy, retaining negative off-diagonal estimates.

    Unbiased common-response energy needs independent zero-mean innovations
    after a fixed projection. Estimated shared CFO and actual phase dynamics
    can break that assumption. Innovation is never labeled thermal noise.
    """
    if cycles.ndim != 3 or cycles.shape[1] < 2 or not torch.is_complex(cycles):
        raise ValueError('Expected complex [N,B>=2,L] cycles')
    count, width = cycles.shape[1:]
    sum_energy = cycles.abs().square().sum((-2, -1))
    coherent_sum_energy = cycles.sum(1).abs().square().sum(-1)
    repeated = (coherent_sum_energy - sum_energy) / (count * (count - 1) * width)
    mean = cycles.mean(1, keepdim=True)
    innovation = (cycles - mean).abs().square().sum((-2, -1)) / ((count - 1) * width)
    return dict(repeat_energy=repeated, innovation_energy=innovation,
                mean_response_energy=mean.abs().square().sum((-2, -1)) / width,
                total_cycle_energy=sum_energy / (count * width))


def _quality(window):
    cycles = window.reshape(-1, 4, PERIOD)
    first, second = window[:, :-PERIOD], window[:, PERIOD:]
    cross = (second * first.conj()).sum(-1)
    denom = (first.abs().square().sum(-1) * second.abs().square().sum(-1)).sqrt()
    mean = cycles.mean(1)
    power = window.abs().square().mean(-1)
    spectrum = torch.fft.fft(mean, dim=-1) / PERIOD
    occupied = torch.tensor([k % PERIOD for k in HARMONICS], device=window.device)
    total = spectrum.abs().square().sum(-1)
    return dict(power=power, lag20_coherence=cross.abs() / denom.clamp_min(EPS),
                coherent_cycle_fraction=mean.abs().square().mean(-1) / power.clamp_min(EPS),
                occupied_tone_fraction=spectrum[:, occupied].abs().square().sum(-1) / total.clamp_min(EPS),
                dc_fraction=spectrum[:, 0].abs().square() / total.clamp_min(EPS)), spectrum


class PhysicalAudit:
    """Packet-local fixed public transforms, instantiated once per worker."""
    def __init__(self, device='cpu', dtype=torch.float64):
        if dtype != torch.float64:
            raise ValueError('Physical SVD audit is defined in float64')
        self.device = torch.device(device)
        self.dtype = dtype
        self.basis20 = nonlinear_basis(20, self.device, dtype)
        self.basis80 = nonlinear_basis(80, self.device, dtype)
        if self.basis20['rank'] != 1 or self.basis80['rank'] != 1:
            raise RuntimeError('Known strict-inband nonlinear rank differs from one')
        self.occupied = torch.tensor([k % PERIOD for k in HARMONICS], device=self.device)
        self.delay_grid = torch.arange(320, device=self.device, dtype=dtype) / 16
        self.template_bank = template(torch.arange(PERIOD, device=self.device, dtype=dtype)[None]
                                      - self.delay_grid[:, None])
        self.names = [f'time_tone_{k}_{component}' for k in HARMONICS
                      for component in ('power_contrast', 'cross_real', 'cross_imag')]
        self.names += ['nonlinear_ref_coordinate_real', 'nonlinear_ref_coordinate_imag',
                       'nonlinear_cycle_coordinate_real', 'nonlinear_cycle_coordinate_imag',
                       'nonlinear_repeat_energy', 'nonlinear_innovation_energy']
        self.groups = dict(temporal_projective=list(range(36)), nonlinear_residual=list(range(36, 42)))
        self.contract = dict(schema='stage2_geometry_physics_v1', input='IQ[N,2,256]',
            sample_rate_hz=SAMPLE_RATE, fixed_period_samples=PERIOD,
            windows=dict(early=[0, 80], reference=[80, 160], candidate_quality_only=[160, 240], tail_quality_only=[240, 256]),
            candidate_window_assumed_STF=False, tail_is_complete_symbol=False,
            timing_diagnostic='best public-template correlation on320 delays at1/16 sample, each window only reported, not corrected',
            temporal_projective='per-tone [early,reference] complex projectors; 3 independent real coordinates',
            temporal_gain_invariance='shared nonzero complex gain per tone, above declared energy floor',
            convolution_window_invariance=False, absolute_tone_energy_floor=EPS,
            nonlinear_generation='continuous Fourier convolution -> strict |f|<10MHz -> final 25Msps samples',
            nonlinear_orders=[3, 5], nonlinear_identifiable_complex_rank=1,
            nuisance='12 free complex linear tones, DC, phase, delay derivative, complex-span local CFO tangent',
            whitening='identity Sigma assumption; unknown noise covariance is not estimated',
            coefficient_inversion=False, nonlinear_timing_gauge='fixed public template origin; no claim of recovered PA coefficients',
            nonlinear_80_window='periodic finite-window ideal mask, then fixed 80-sample nuisance projection and residual SVD',
            nonlinear_repeat='same fixed 20-sample projection separately on four cycles; off-diagonal energy not clamped',
            innovation_is_thermal_noise=False,
            repeat_energy_unbiased_condition='only independent zero-mean innovations after fixed preprocessing; shared estimated CFO can violate this',
            cfo='one reference[80,160) lag20 wrapped estimate corrects all256 samples; no per-window CFO estimation',
            cfo_alias_interval_hz=[-625000., 625000.], cfo_is_pure_tx_or_rx=False,
            cross_packet_fit=False, label_access=False, target_access=False,
            limitations=['Static equal steady windows produce a constant temporal projector and no new TX evidence.',
                         'RX AGC, synchronization, link response and preprocessing may also produce dynamic residuals.',
                         'Complex nonlinear coordinates retain a received phase/timing gauge; they are not arbitrary-channel invariants.',
                         'Strict ideal 20MHz passband is an explicit analysis assumption, not measured receiver filtering.'])

    @torch.no_grad()
    def features(self, iq):
        if not torch.is_tensor(iq) or iq.ndim != 3 or tuple(iq.shape[1:]) != (2, 256) or not len(iq):
            raise ValueError('Expected nonempty real IQ[N,2,256] torch tensor')
        if not iq.is_floating_point() or not bool(torch.isfinite(iq).all()):
            raise ValueError('IQ must be finite floating point')
        a = iq.to(device=self.device, dtype=self.dtype)
        z = torch.complex(a[:, 0], a[:, 1])
        ref = z[:, 80:160]
        cross = (ref[:, PERIOD:] * ref[:, :-PERIOD].conj()).sum(-1)
        ref_power = ref.abs().square().mean(-1)
        cfo_valid = (cross.abs() > EPS) & (ref_power > EPS)
        omega = torch.where(cfo_valid, torch.angle(cross) / PERIOD, torch.zeros_like(ref_power))
        n = torch.arange(256, device=self.device, dtype=self.dtype)
        corrected = z * torch.exp(-1j * omega[:, None] * n)
        packet_power = z.abs().square().mean(-1)
        stats = dict(reference_cfo_hz=omega * SAMPLE_RATE / (2 * math.pi),
                     reference_cfo_valid=cfo_valid, packet_power=packet_power,
                     zero_energy_packet=packet_power <= EPS)
        spectra = {}
        cycle_means = {}
        for start in (0, 80, 160):
            qualities, spectrum = _quality(corrected[:, start:start + 80])
            for name, value in qualities.items():
                stats[f'w{start}_{start + 80}_{name}'] = value
            spectra[start] = spectrum
            mean = corrected[:, start:start + 80].reshape(-1, 4, PERIOD).mean(1)
            cycle_means[start] = mean
            dot = mean @ self.template_bank.conj().T
            denominator = mean.abs().square().sum(-1)[:, None] * self.template_bank.abs().square().sum(-1)[None]
            score = dot.abs().square() / denominator.clamp_min(EPS)
            best = score.argmax(-1)
            stats[f'w{start}_{start + 80}_template_correlation'] = score.gather(1, best[:, None]).squeeze(1).clamp(0, 1).sqrt()
            stats[f'w{start}_{start + 80}_best_public_delay_samples'] = self.delay_grid[best]
        early_mean, reference_mean = cycle_means[0], cycle_means[80]
        denominator = (early_mean.abs().square().sum(-1) * reference_mean.abs().square().sum(-1)).sqrt()
        stats['early_reference_cycle_correlation'] = (early_mean * reference_mean.conj()).sum(-1).abs() / denominator.clamp_min(EPS)
        pair = torch.stack((spectra[0][:, self.occupied], spectra[80][:, self.occupied]), 1)
        temporal, valid, energy = projective_time(pair)
        stats['temporal_valid_tone_fraction'] = valid.double().mean(-1)
        stats['temporal_min_tone_energy'] = energy.min(-1).values
        stats['temporal_mean_tone_energy'] = energy.mean(-1)
        steady = temporal.reshape(-1, 12, 3).new_tensor([0., 1., 0.])
        stats['temporal_distance_from_steady'] = ((temporal.reshape(-1, 12, 3) - steady).square().sum(-1) * valid).sum(-1).sqrt()
        for j, harmonic in enumerate(HARMONICS):
            stats[f'temporal_tone_{harmonic}_valid'] = valid[:, j]
        reference = corrected[:, 80:160]
        r80 = _residual(strict_lowpass(reference), self.basis80['nuisance'])
        coord80 = (r80 @ self.basis80['identifiable'].conj())[:, 0] / math.sqrt(80)
        cycles = reference.reshape(-1, 4, PERIOD)
        r20 = _residual(strict_lowpass(cycles), self.basis20['nuisance'])
        cycle_coords = (r20 @ self.basis20['identifiable'].conj())[:, :, 0] / math.sqrt(PERIOD)
        energies = repeat_energy(r20)
        denom = ref_power.clamp_min(EPS)
        scale = denom.sqrt()
        mean_coord = cycle_coords.mean(1)
        nonlinear = torch.stack((coord80.real / scale, coord80.imag / scale,
                                 mean_coord.real / scale, mean_coord.imag / scale,
                                 energies['repeat_energy'] / denom, energies['innovation_energy'] / denom), -1)
        for name, value in energies.items():
            stats['nonlinear_' + name] = value
        stats['nonlinear_80_residual_power'] = r80.abs().square().mean(-1)
        stats['nonlinear_80_identifiable_power'] = coord80.abs().square()
        stats['nonlinear_repeat_estimate_negative'] = energies['repeat_energy'] < 0
        stats['nonlinear_reference_energy_valid'] = ref_power > EPS
        tail = corrected[:, 240:256]
        tail_power = tail.abs().square().mean(-1)
        stats['tail16_power_ratio'] = tail_power / packet_power.clamp_min(EPS)
        stats['tail16_dc_fraction'] = tail.mean(-1).abs().square() / tail_power.clamp_min(EPS)
        # The extension is assessed as evidence, never accepted by a hidden threshold.
        stats['candidate160_240_reference_power_ratio'] = stats['w160_240_power'] / denom
        features = torch.cat((temporal, nonlinear), -1)
        if not bool(torch.isfinite(features).all()) or any(not bool(torch.isfinite(v).all()) for v in stats.values()):
            raise FloatingPointError('Nonfinite fixed physical feature or quality statistic')
        return dict(features=features, feature_names=list(self.names), feature_groups=dict(self.groups),
                    stats=stats, contract=dict(self.contract))

    __call__ = features

    @torch.no_grad()
    def linear_stability(self, iq, clean=None):
        """Two predeclared causal L corners on each actual source packet.

        No auxiliary network or new data is used. Compare deterministic packet
        features after the same complete preprocessing; changes include the
        reference CFO estimator's response and finite capture boundaries.
        """
        clean = self.features(iq) if clean is None else clean
        a = iq.to(device=self.device, dtype=self.dtype)
        z = torch.complex(a[:, 0], a[:, 1])
        s1 = torch.nn.functional.pad(z[:, :-1], (1, 0))
        s3 = torch.nn.functional.pad(z[:, :-3], (3, 0))
        values = {}
        corners = ((.06 + .06j, -.06 + .06j), (-.06 + .06j, .06 - .06j))
        for index, (h1, h3) in enumerate(corners):
            changed = z + h1 * s1 + h3 * s3
            measured = self.features(torch.stack((changed.real, changed.imag), 1))
            prefix = f'controlled_L_corner{index}_'
            difference = measured['features'] - clean['features']
            values[prefix + 'temporal_feature_l2'] = difference[:, :36].norm(dim=-1)
            values[prefix + 'nonlinear_feature_l2'] = difference[:, 36:].norm(dim=-1)
            values[prefix + 'cfo_proxy_change_hz'] = measured['stats']['reference_cfo_hz'] - clean['stats']['reference_cfo_hz']
            values[prefix + 'valid_tone_fraction_change'] = measured['stats']['temporal_valid_tone_fraction'] - clean['stats']['temporal_valid_tone_fraction']
            values[prefix + 'iq_relative_l2'] = (changed - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(EPS)
        return dict(stats=values, contract=dict(operator='I+h1*S1+h3*S3, causal zero extension of actual capture',
                    tap_components=[[[h1.real, h1.imag], [h3.real, h3.imag]] for h1, h3 in corners],
                    frozen_actions=True, source_only=True, target_access=False,
                    scope='finite bounded post-reception stability; changes may include window and shared-CFO-estimator effects'))


def algebraic_audit(device='cpu'):
    """Small, public-template proofs and numerical evidence; no dataset needed."""
    t = torch.arange(20, device=device, dtype=torch.float64)
    s = template(t)
    discrete = torch.stack((s, s * s.abs().square(), s * s.abs().pow(4)), -1)
    singular = torch.linalg.svdvals(discrete)
    p3, p5 = fourier_coefficients(3), fourier_coefficients(5)
    ratio_error = max(abs(p5[k] - 17 / 8 * p3[k]) for k in (-7, 7))
    sidebands = torch.tensor([[p3[k], p5[k]] for k in (-7, 7)], device=device, dtype=torch.complex128)
    b20, b80 = nonlinear_basis(20, device), nonlinear_basis(80, device)
    # Bounds use ||S_d||<=1; they hold for every registered component bound.
    radius = 2 * math.sqrt(2) * .06
    n = 256
    shift = torch.zeros(n, n, device=device, dtype=torch.complex128)
    ix = torch.arange(1, n, device=device)
    shift[ix, ix - 1] = 1
    matrix = torch.eye(n, device=device, dtype=torch.complex128) + .06 * (1 + 1j) * (shift + shift @ shift @ shift)
    operator_singular = torch.linalg.svdvals(matrix)
    # Contrast physical bandlimited coefficients with aliased sampled powers.
    sampled3 = torch.fft.fft(s * s.abs().square()) / PERIOD
    alias_error = max(abs(complex(sampled3[k % PERIOD].item()) - p3.get(k, 0j)) for k in (-7, 7))
    return dict(schema='stage2_geometry_physics_algebra_v1', target_access=False,
                template_conjugate_delay_error=float((s.conj() + 1j * template(t, 10.)).abs().max()),
                conjugate_delay_identity='conj(s[n])=-j*s[n-10]; only nuisance models admitting this delay/free-tone response absorb it',
                discrete_orders_135_rank=int(torch.linalg.matrix_rank(discrete)),
                discrete_four_repeats_rank=int(torch.linalg.matrix_rank(discrete.repeat(4, 1))),
                discrete_orders_135_condition=float(singular[0] / singular[-1]),
                discrete_orders_135_singular=singular.tolist(),
                continuous_harmonic_extent={str(p): max(abs(k) for k in fourier_coefficients(p)) for p in (1, 3, 5)},
                continuous_max_frequency_hz={str(p): max(abs(k) for k in fourier_coefficients(p)) * SAMPLE_RATE / PERIOD for p in (1, 3, 5)},
                strict_inband_harmonics=list(range(-7, 8)), original_active_harmonics=list(HARMONICS),
                after_linear_tones_and_DC_new_harmonics=[-7, 7],
                fifth_to_third_sideband_ratio=17 / 8, sideband_ratio_error=ratio_error,
                sideband_orders_35_complex_rank=int(torch.linalg.matrix_rank(sidebands)),
                sampled_cubic_sideband_alias_error=alias_error,
                residual_svd={str(n): dict(rank=b['rank'], nuisance_rank=b['nuisance_rank'],
                    singular=b['nonlinear_singular'].tolist(), threshold=b['absolute_threshold'],
                    identity_whitened_complex_gram=dict(
                        real=(b['residual_design'].conj().T @ b['residual_design']).real.tolist(),
                        imag=(b['residual_design'].conj().T @ b['residual_design']).imag.tolist()))
                    for n, b in ((20, b20), (80, b80))},
                effective_information_scope='identity-whitened complex Gram diagnostic; not estimated physical noise Fisher',
                linear_action_bound=dict(tap_component_limit=.06, delays=[1, 3], lower=1 - radius, upper=1 + radius,
                    proof='triangle/reverse-triangle with each finite causal shift operator norm<=1',
                    finite_matrix_size=n, checked_singular_min=float(operator_singular[-1]), checked_singular_max=float(operator_singular[0]),
                    deep_zero_possible_within_bound=False),
                whitening='Sigma unknown; identity whitening assumption',
                independent_repeats_add_rank=False, physical_PA_orders_separately_identifiable=False)
