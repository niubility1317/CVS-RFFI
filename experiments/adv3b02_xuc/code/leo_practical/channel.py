"""LEO_practical v1: physics-constrained *incremental* IQ channel proxy.

NumPy only. Engineering parameters are not measured LEO parameters. Noise in
the supplied ground recording is retained; reported SNR concerns added noise.
No dependency on, or mutation of, historical channel implementations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from typing import Optional

import numpy as np

VERSION = "leo_practical_v3_full_residual_20260918"
C = 299792458.0
RE = 6371000.0
GM = 3.986004418e14
STATES = ("LOS", "PARTIAL", "BLOCKED")
SCENARIOS = {
    "practical_high": ("suburban", (45.0, 80.0)),
    "practical_mid": ("suburban", (20.0, 45.0)),
    "practical_low_urban": ("urban", (10.0, 30.0)),
    "practical_low_suburban": ("suburban", (10.0, 30.0)),
    "practical_mid_urban": ("urban", (20.0, 45.0)),
    "practical_high_urban": ("urban", (45.0, 80.0)),
}


def stable_seed(seed: int, *parts) -> int:
    """Independent named streams; unchanged by batch order/subset selection."""
    raw = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class Config:
    fs_hz: float  # Required: never silently inherit CVS's 25 MHz.
    scenario: str = "practical_mid"
    fc_hz: float = 2.45e9
    altitude_m: float = 600000.0
    reference_altitude_m: float = 600000.0
    reference_elevation_deg: float = 60.0
    reference_snr_db: float = 30.0
    receiver_cfo_std_hz: float = 200.0
    residual_cfo_std_hz: float = 100.0
    phase_noise_linewidth_hz: float = 1.0
    iq_amp_max_db: float = 0.2
    iq_phase_max_deg: float = 1.0
    shadow_corr_distance_m: float = 3.0
    terminal_speed_mps: float = 1.0
    shadow_corr_time_s: Optional[float] = None
    state_mix_time_s: float = 1.0
    scatter_doppler_hz: Optional[float] = None
    scatter_sinusoids: int = 32
    los_mean_db: tuple = (0.0, -8.0, 0.0)
    los_std_db: tuple = (1.0, 3.0, 0.0)
    diffuse_power_db: tuple = (-18.0, -18.0, -20.0)
    rms_delay_ns: tuple = (20.0, 50.0, 100.0)
    mode: str = "post_sync"
    receiver_location: str = "satellite"
    input_processing_state: str = "unknown_ground_recording_processing"
    iq_compensation_enabled: bool = True
    iq_estimation_amp_std_db: float = 0.05
    iq_estimation_phase_std_deg: float = 0.2
    phase_tracking_enabled: bool = True
    phase_tracking_residual_std_deg: float = 1.0
    phase_tracking_correlation_time_s: float = 0.001
    agc_enabled: bool = True
    agc_limit_db: float = 30.0
    boundary: str = "reflect"
    atmosphere_loss_db: float = 0.0
    atmosphere_source: str = "clear_simplified_no_weather_injection"
    local_geometry_max_duration_s: float = 10.0
    quality_adaptation_enabled: bool = True
    locked_snr_db: float = 15.0
    unlocked_snr_db: float = 5.0
    quality_reference_snr_db: float = 20.0
    max_residual_scale: float = 10.0
    coarse_doppler_error_std_hz: float = 500.0
    fractional_delay_half_length: int = 24
    processing_route: str = "full"
    equalization_enabled: bool = False
    equalizer_method: str = "mmse"
    equalizer_length: int = 129
    equalizer_delay_samples: int = 64
    equalizer_max_gain_db: float = 20.0
    channel_estimation_nmse_db: float = -25.0
    zf_regularization: float = 1e-6

    def __post_init__(self):
        if self.scenario not in SCENARIOS or self.mode not in {"pre_sync", "post_sync"}:
            raise ValueError("invalid scenario or synchronization mode")
        if self.receiver_location not in {"ground", "satellite"}:
            raise ValueError("receiver_location must be ground or satellite")
        if self.processing_route not in {"full", "residual"} or self.equalizer_method not in {"mmse", "zf"}:
            raise ValueError("invalid processing route or equalizer method")
        if self.processing_route == "residual" and self.mode != "post_sync":
            raise ValueError("residual route describes post_sync only")
        if self.equalization_enabled and self.mode != "post_sync":
            raise ValueError("equalization requires post_sync")
        if int(self.equalizer_length) != self.equalizer_length or self.equalizer_length < 3 or int(self.equalizer_delay_samples) != self.equalizer_delay_samples or not 0 <= self.equalizer_delay_samples < self.equalizer_length:
            raise ValueError("invalid equalizer FIR length/delay")
        if not all(math.isfinite(v) for v in (self.equalizer_max_gain_db, self.channel_estimation_nmse_db, self.zf_regularization)) or self.equalizer_max_gain_db < 0 or self.zf_regularization <= 0:
            raise ValueError("invalid equalizer gain/regularization")
        if not self.input_processing_state.strip():
            raise ValueError("input processing state must be documented")
        if not all(math.isfinite(v) for v in (self.locked_snr_db, self.unlocked_snr_db,
                self.quality_reference_snr_db, self.max_residual_scale)) or self.locked_snr_db <= self.unlocked_snr_db or self.max_residual_scale < 1:
            raise ValueError("invalid receiver quality thresholds")
        if int(self.fractional_delay_half_length) != self.fractional_delay_half_length or self.fractional_delay_half_length < 8:
            raise ValueError("fractional delay half length must be an integer >= 8")
        if self.boundary not in {"reflect", "edge", "require_context"}:
            raise ValueError("boundary must be reflect, edge, or require_context")
        positive = (self.fs_hz, self.fc_hz, self.altitude_m,
                    self.reference_altitude_m, self.shadow_corr_distance_m,
                    self.state_mix_time_s, self.local_geometry_max_duration_s,
                    self.phase_tracking_correlation_time_s)
        if not all(v is not None and math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("frequencies, heights and correlation scales must be positive")
        if not 10 <= self.reference_elevation_deg <= 90:
            raise ValueError("reference elevation must be 10..90 degrees")
        nonnegative = (self.receiver_cfo_std_hz, self.residual_cfo_std_hz,
                       self.phase_noise_linewidth_hz, self.iq_amp_max_db,
                       self.iq_phase_max_deg, self.terminal_speed_mps,
                       self.agc_limit_db, self.atmosphere_loss_db,
                       self.iq_estimation_amp_std_db, self.iq_estimation_phase_std_deg,
                       self.phase_tracking_residual_std_deg, self.coarse_doppler_error_std_hz)
        if not all(math.isfinite(v) and v >= 0 for v in nonnegative):
            raise ValueError("impairment scales must be finite and nonnegative")
        if not math.isfinite(self.reference_snr_db):
            raise ValueError("reference SNR must be finite")
        if self.shadow_corr_time_s is not None and (
                not math.isfinite(self.shadow_corr_time_s) or self.shadow_corr_time_s <= 0):
            raise ValueError("explicit shadow correlation time must be positive")
        if self.terminal_speed_mps == 0 and self.shadow_corr_time_s is None:
            raise ValueError("stationary terminal requires explicit shadow_corr_time_s")
        if self.scatter_doppler_hz is not None and (
                not math.isfinite(self.scatter_doppler_hz) or self.scatter_doppler_hz < 0):
            raise ValueError("scatter Doppler must be finite and nonnegative")
        if self.scatter_sinusoids < 8 or int(self.scatter_sinusoids) != self.scatter_sinusoids:
            raise ValueError("scatter_sinusoids must be an integer >= 8")
        for values in (self.los_mean_db, self.los_std_db,
                       self.diffuse_power_db, self.rms_delay_ns):
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError("state parameter arrays require three finite values")
        if min(self.los_std_db) < 0 or min(self.rms_delay_ns) < 0:
            raise ValueError("standard deviations/delays cannot be negative")
        if self.atmosphere_loss_db and self.atmosphere_source == "clear_simplified_no_weather_injection":
            raise ValueError("nonzero atmosphere loss requires a provenance description")

    @property
    def correlation_time_s(self):
        return (self.shadow_corr_time_s if self.shadow_corr_time_s is not None
                else self.shadow_corr_distance_m / self.terminal_speed_mps)

    @cached_property
    def config_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def slant_range(altitude_m, elevation_deg):
    angle = np.deg2rad(elevation_deg)
    return np.sqrt((RE + altitude_m)**2 - (RE * np.cos(angle))**2) - RE * np.sin(angle)


def state_probabilities(elevation_deg, environment):
    """Nie 2012 Eq.14-16, mapped by physical names, NOT the swapped Eq.9."""
    if not 10 <= elevation_deg <= 90 or environment not in {"urban", "suburban"}:
        raise ValueError("paper visibility prior supports urban/suburban, 10..90 deg")
    clear = 1 - (90 - elevation_deg)**2 / (7000 if environment == "urban" else 16600)
    partial = (1 - clear) * (0.2 if environment == "urban" else 0.8)
    return np.array([clear, partial, 1 - clear - partial], dtype=np.float64)


def transition_matrix(probabilities, dt_s, mix_time_s):
    if dt_s < 0 or mix_time_s <= 0:
        raise ValueError("invalid time interval")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("invalid stationary probabilities")
    a = math.exp(-dt_s / mix_time_s)
    return a * np.eye(3) + (1 - a) * np.broadcast_to(p, (3, 3))


@dataclass(frozen=True)
class Geometry:
    altitude_m: float
    elevation_deg: float
    distance_m: float
    range_rate_mps: float  # Positive means receding.
    source: str


def sample_geometry(cfg, rng):
    _, limits = SCENARIOS[cfg.scenario]
    elevation = float(rng.uniform(*limits))
    speed = math.sqrt(GM / (RE + cfg.altitude_m))
    rate = float(rng.choice([-1, 1]) * speed * math.cos(math.radians(elevation)))
    return Geometry(cfg.altitude_m, elevation, float(slant_range(cfg.altitude_m, elevation)),
                    rate, "simplified_circular_overpass_not_ephemeris")


def geometry_from_vectors(satellite_position_m, satellite_velocity_mps,
                          terminal_position_m, terminal_velocity_mps):
    """All vectors in the same Earth-centred frame, at the same epoch."""
    rs, vs, rg, vg = [np.asarray(v, dtype=float) for v in (
        satellite_position_m, satellite_velocity_mps,
        terminal_position_m, terminal_velocity_mps)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (rs, vs, rg, vg)):
        raise ValueError("geometry vectors must be finite length-three arrays")
    delta = rs - rg
    distance = np.linalg.norm(delta)
    if distance <= 0 or np.linalg.norm(rg) <= 0:
        raise ValueError("invalid satellite/terminal position")
    unit = delta / distance
    elevation = np.rad2deg(np.arcsin(np.clip(unit @ (rg / np.linalg.norm(rg)), -1, 1)))
    return Geometry(float(np.linalg.norm(rs) - RE), float(elevation), float(distance),
                    float((vs - vg) @ unit), "supplied_state_vectors_local_linearization")


@dataclass(frozen=True)
class Receiver:
    session_id: str
    amplitude_imbalance_db: float
    quadrature_error_deg: float
    oscillator_cfo_hz: float
    estimated_amplitude_imbalance_db: float
    estimated_quadrature_error_deg: float


def receiver_for_session(cfg, seed, session_id):
    rng = np.random.default_rng(stable_seed(seed, "receiver", str(session_id)))
    amplitude = float(rng.uniform(-cfg.iq_amp_max_db, cfg.iq_amp_max_db))
    quadrature = float(rng.uniform(-cfg.iq_phase_max_deg, cfg.iq_phase_max_deg))
    cfo = float(rng.normal(0, cfg.receiver_cfo_std_hz))
    return Receiver(str(session_id), amplitude, quadrature, cfo,
                    amplitude + float(rng.normal(0, cfg.iq_estimation_amp_std_db)),
                    quadrature + float(rng.normal(0, cfg.iq_estimation_phase_std_deg)))


def iq_imbalance(x, receiver):
    gi = 10**(receiver.amplitude_imbalance_db / 40)
    gq = 10**(-receiver.amplitude_imbalance_db / 40)
    phi = np.deg2rad(receiver.quadrature_error_deg)
    return gi * x.real * np.exp(-0.5j * phi) + 1j * gq * x.imag * np.exp(0.5j * phi)


def compensate_iq(x, amplitude_estimate_db, quadrature_estimate_deg):
    """Invert the estimated widely-linear RX response, not TX impairments.

    Estimates may come from an external calibration; receiver_for_session uses
    a declared truth-plus-error simulation proxy, NOT a blind IQ estimator.
    """
    gi, gq = 10**(amplitude_estimate_db/40), 10**(-amplitude_estimate_db/40)
    angle = math.radians(quadrature_estimate_deg)
    ci, cq = gi*np.exp(-0.5j*angle), gq*np.exp(0.5j*angle)
    alpha, beta = (ci+cq)/2, (ci-cq)/2
    determinant = abs(alpha)**2 - abs(beta)**2
    if abs(determinant) < 1e-6:
        raise ValueError("ill-conditioned estimated IQ correction")
    return (np.conj(alpha)*x - beta*np.conj(x))/determinant


def waveform_metrics(x):
    power = np.abs(x)**2
    total = float(power.sum())
    if total == 0:
        return dict(rms=0.0, papr_db=None, effective_energy_samples=0.0, peak_energy_share=0.0)
    return dict(rms=float(np.sqrt(power.mean())),
                papr_db=float(10*np.log10(power.max()/power.mean())),
                effective_energy_samples=total**2/float(np.square(power).sum()),
                peak_energy_share=float(power.max()/total))


def fractional_delay_kernel(fraction, half_length=24):
    """Kaiser-windowed sinc, common causal latency=half_length samples.

    A propagation delay filter, never a receiver equalizer. The integer/common
    latency is identical across taps and is recorded explicitly.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0,1)")
    positions = np.arange(2*half_length+1, dtype=float)
    kernel = np.sinc(positions-half_length-fraction)*np.kaiser(len(positions), 8.6)
    return kernel/kernel.sum()


def receiver_quality(cfg, snr_db):
    """Engineering lock/error proxy, not a measured acquisition probability."""
    if not math.isfinite(snr_db):
        raise ValueError("quality SNR must be finite")
    if not cfg.quality_adaptation_enabled:
        return "locked", 1.0
    status = ("locked" if snr_db >= cfg.locked_snr_db else
              "degraded" if snr_db >= cfg.unlocked_snr_db else "unlocked")
    exponent = np.clip((cfg.quality_reference_snr_db-snr_db)/20, 0, math.log10(cfg.max_residual_scale))
    return status, float(10**exponent)


def normalize_input(x):
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("input must be a finite 1D complex record of at least four samples")
    rms = float(np.sqrt(np.mean(np.abs(x)**2)))
    if rms <= 0:
        raise ValueError("zero-energy record: cannot define reference power")
    return x / rms, rms


class ChannelStream:
    """Continuous raw-IQ stream; reuse ONLY for proven contiguous recordings.

    Local geometry is held at its epoch with a linear range-rate approximation.
    Long orbital tracks must be segmented with externally supplied geometry;
    this is not an orbit propagator. State/OU/scatter/phase/input history persist.
    AGC and receiver-quality decisions are blockwise and depend on boundaries.
    Input is in shared reference units: do not renormalize consecutive blocks.
    """
    def __init__(self, cfg, seed, receiver, geometry=None, receiver_processor=None):
        self.cfg, self.seed, self.receiver = cfg, int(seed), receiver
        self.receiver_processor = receiver_processor
        if receiver_processor is not None:
            raise ValueError("RFF-safe policy: external receiver transforms/equalizers are disabled")
        self.rngs = {k: np.random.default_rng(stable_seed(seed, k)) for k in
                     ("geometry", "states", "shadow", "scatter", "phase", "noise", "sync", "tracking", "equalizer")}
        self.geometry = geometry or sample_geometry(cfg, self.rngs["geometry"])
        g = self.geometry
        if not all(math.isfinite(v) for v in (g.altitude_m, g.elevation_deg, g.distance_m, g.range_rate_mps)) or g.distance_m <= 0 or g.altitude_m <= 0:
            raise ValueError("invalid geometry")
        self.probs = state_probabilities(g.elevation_deg, SCENARIOS[cfg.scenario][0])
        self.state = int(self.rngs["states"].choice(3, p=self.probs))
        self.z = float(self.rngs["shadow"].normal())
        self.index = 0
        self.phase = float(self.rngs["phase"].uniform(-np.pi, np.pi))
        self.tracking_z = float(self.rngs["tracking"].normal())
        self.residual_hz = float(self.rngs["sync"].normal(0, cfg.residual_cfo_std_hz))
        self.coarse_doppler_error_hz = float(self.rngs["sync"].normal(0, cfg.coarse_doppler_error_std_hz))
        self.compensation_phase = 0.0
        self.f_doppler = -cfg.fc_hz * g.range_rate_mps / C
        self.f_total = self.f_doppler + receiver.oscillator_cfo_hz
        self.f_hat = self.f_total - self.residual_hz
        fd = (cfg.scatter_doppler_hz if cfg.scatter_doppler_hz is not None
              else cfg.terminal_speed_mps * cfg.fc_hz / C)
        self.scatter_fd = fd
        # Gaussian coefficients give exact complex Gaussian marginals;
        # uniformly sampled ray angles approximate a Jakes temporal spectrum.
        shape = (3, cfg.scatter_sinusoids)
        r = self.rngs["scatter"]
        self.ray_hz = fd * np.cos(r.uniform(-np.pi, np.pi, shape))
        self.ray_coef = (r.normal(size=shape) + 1j*r.normal(size=shape)) / math.sqrt(2*cfg.scatter_sinusoids)
        self.diffuse = 10**(np.asarray(cfg.diffuse_power_db)/10)
        a = math.log(10)/20
        self.mean_los_power = np.exp(2*a*np.asarray(cfg.los_mean_db) + 2*a*a*np.square(cfg.los_std_db))
        self.mean_los_power[2] = 0
        self.reference_power = float(self.mean_los_power[0] + self.diffuse[0])
        # Three diffuse rays; one LOS component shares delay zero. Scale delays
        # against the TOTAL ensemble PDP, including direct-path power.
        self.pdp = np.exp(-np.arange(3, dtype=float))
        self.pdp /= self.pdp.sum()
        base = np.array([0., 1., 3.])
        delays = []
        for s in range(3):
            weights = self.diffuse[s] * self.pdp
            weights = weights.copy()
            weights[0] += self.mean_los_power[s]
            weights /= weights.sum()
            spread = np.sqrt(np.sum(weights*(base - weights @ base)**2))
            delays.append(base * cfg.rms_delay_ns[s]*1e-9 * cfg.fs_hz / spread)
        self.delays = np.asarray(delays)
        self.history_size = int(np.floor(self.delays.max())) + 2*cfg.fractional_delay_half_length
        self.history = None
        self.eq_signal_history = None
        self.eq_noise_history = None

    def process(self, x, *, preceding_context=None, return_traces=False, quality_snr_db=None):
        cfg = self.cfg
        if cfg.processing_route != "full":
            raise ValueError("use ResidualChannel or batch API for the residual route")
        x = np.asarray(x, dtype=np.complex128)
        if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
            raise ValueError("expected finite 1D IQ, at least four samples")
        n = len(x)
        if (self.index + n)/cfg.fs_hz > cfg.local_geometry_max_duration_s:
            raise ValueError("local geometry time limit exceeded; use an updated trajectory model")
        first = self.index == 0
        if preceding_context is not None:
            if not first:
                raise ValueError("context can be supplied only at stream start")
            context = np.asarray(preceding_context, dtype=np.complex128)
            if context.ndim != 1 or len(context) < self.history_size or not np.isfinite(context).all():
                raise ValueError("insufficient or invalid preceding context")
            self.history = context[-self.history_size:].copy()
        if self.history is None:
            if cfg.boundary == "require_context":
                raise ValueError("real preceding IQ context required by boundary policy")
            self.history = np.pad(x, (self.history_size, 0), mode=cfg.boundary)[:self.history_size]
            boundary_used = "synthetic_" + cfg.boundary
        else:
            boundary_used = "provided_context" if first else "continuous_input_history"
        extended = np.concatenate([self.history, x])
        self.history = extended[-self.history_size:].copy()

        dt = 1 / cfg.fs_hz
        t = (self.index + np.arange(n)) * dt
        distance = self.geometry.distance_m + self.geometry.range_rate_mps*t
        if np.any(distance <= 0):
            raise ValueError("local geometry expired; supply updated trajectory geometry")
        ref_distance = slant_range(cfg.reference_altitude_m, cfg.reference_elevation_deg)
        gain = ref_distance / distance * 10**(-cfg.atmosphere_loss_db/20)
        # Exact stationary OU discretization and CTMC transition probability.
        rho = math.exp(-dt/cfg.correlation_time_s)
        innovation = math.sqrt(-math.expm1(-2*dt/cfg.correlation_time_s))
        refresh_probability = -math.expm1(-dt/cfg.state_mix_time_s)
        draws = self.rngs["states"].random((n, 2))
        innovations = self.rngs["shadow"].normal(size=n)
        state = np.empty(n, dtype=np.int64)
        shadow = np.empty(n)
        cumulative = np.cumsum(self.probs)
        for k in range(n):
            state[k], shadow[k] = self.state, self.z
            if draws[k, 0] < refresh_probability:
                self.state = min(2, int(np.searchsorted(cumulative, draws[k, 1])))
            self.z = rho*self.z + innovation*innovations[k]
        mu = np.asarray(cfg.los_mean_db)[state]
        sigma = np.asarray(cfg.los_std_db)[state]
        los = 10**((mu+sigma*shadow)/20)
        los[state == 2] = 0
        fading = np.empty((n, 3), dtype=np.complex128)
        for tap in range(3):
            fading[:, tap] = np.exp(2j*np.pi*t[:, None]*self.ray_hz[tap]) @ self.ray_coef[tap]
        fading *= np.sqrt(self.diffuse[state, None]*self.pdp[None, :])
        fading[:, 0] += los
        fading /= math.sqrt(self.reference_power)

        signal = np.zeros(n, dtype=np.complex128)
        indices = self.history_size + np.arange(n)
        # Windowed-sinc propagation delay; common latency is NOT silently removed.
        for s in range(3):
            mask = state == s
            if not mask.any():
                continue
            for tap, delay in enumerate(self.delays[s]):
                integer, fraction = int(np.floor(delay)), delay % 1
                shifted = np.zeros(n, dtype=np.complex128)
                for j, coefficient in enumerate(fractional_delay_kernel(fraction, cfg.fractional_delay_half_length)):
                    shifted += coefficient * extended[indices-integer-j]
                signal[mask] += fading[mask, tap]*shifted[mask]
        signal *= gain
        # Noise floor is referenced to unit input and clear ensemble power,
        # never to this particular faded signal's power.
        noise_variance = 10**(-cfg.reference_snr_db/10)
        simulated_quality = float(10*np.log10(max(float(np.mean(abs(signal)**2)), np.finfo(float).tiny)/noise_variance))
        quality = simulated_quality if quality_snr_db is None else float(quality_snr_db)
        lock_status, residual_scale = receiver_quality(cfg, quality)
        if lock_status == "unlocked":
            # Geometry-based coarse precompensation remains; no fine carrier lock.
            self.f_hat = self.f_doppler + self.coarse_doppler_error_hz
        else:
            self.f_hat = self.f_total - self.residual_hz*residual_scale
        phase_tracking_active = cfg.mode == "post_sync" and cfg.phase_tracking_enabled and lock_status != "unlocked"
        nr = self.rngs["noise"].normal(size=(n, 2))
        noise = math.sqrt(noise_variance/2)*(nr[:, 0]+1j*nr[:, 1])
        phase_inc = self.rngs["phase"].normal(size=n)*math.sqrt(2*np.pi*cfg.phase_noise_linewidth_hz*dt)
        phase = self.phase + np.concatenate([[0.], np.cumsum(phase_inc[:-1])])
        self.phase += float(phase_inc.sum())
        tracking_rho = math.exp(-dt/cfg.phase_tracking_correlation_time_s)
        tracking_scale = math.sqrt(-math.expm1(-2*dt/cfg.phase_tracking_correlation_time_s))
        tracking_innovations = self.rngs["tracking"].normal(size=n)
        tracking_residual = np.empty(n)
        for k in range(n):
            tracking_residual[k] = residual_scale*math.radians(cfg.phase_tracking_residual_std_deg)*self.tracking_z
            self.tracking_z = tracking_rho*self.tracking_z + tracking_scale*tracking_innovations[k]
        rotation = np.exp(1j*(2*np.pi*self.f_total*t + phase))
        signal = iq_imbalance(signal*rotation, self.receiver)
        noise = iq_imbalance(noise*rotation, self.receiver)
        if cfg.mode == "post_sync":
            if cfg.iq_compensation_enabled:
                signal = compensate_iq(signal, self.receiver.estimated_amplitude_imbalance_db,
                                       self.receiver.estimated_quadrature_error_deg)
                noise = compensate_iq(noise, self.receiver.estimated_amplitude_imbalance_db,
                                      self.receiver.estimated_quadrature_error_deg)
            correction_phase = self.compensation_phase + 2*np.pi*self.f_hat*np.arange(n)*dt
            correction = np.exp(-1j*correction_phase)
            self.compensation_phase = float((self.compensation_phase + 2*np.pi*self.f_hat*n*dt) % (2*np.pi))
            if phase_tracking_active:
                # Error-model simulation: no claim that pilots/PLL were run.
                correction *= np.exp(-1j*(phase-tracking_residual))
            signal *= correction
            noise *= correction
        eq_applied = cfg.equalization_enabled and lock_status != "unlocked"
        eq_meta = {"reason": "disabled_by_default" if not cfg.equalization_enabled else "receiver_unlocked"}
        if eq_applied:
            from .equalization import propagation_kernel, design_equalizer, linear_filter
            h = propagation_kernel(self, int(state[0]), fading[0], gain[0])
            equalizer, eq_meta = design_equalizer(h, cfg, self.rngs["equalizer"], noise_variance,
                float(np.mean(abs(x)**2)), residual_scale)
            signal, self.eq_signal_history = linear_filter(signal, equalizer, cfg.boundary, self.eq_signal_history)
            noise, self.eq_noise_history = linear_filter(noise, equalizer, cfg.boundary, self.eq_noise_history)
        else:
            self.eq_signal_history = self.eq_noise_history = None
        y = signal + noise
        external_processing = {"timing_sync": "not_run_cropped_input",
                               "channel_equalization": cfg.equalizer_method if eq_applied else "not_applied"}
        rms = float(np.sqrt(np.mean(np.abs(y)**2)))
        requested_gain_db = -20*math.log10(max(rms, np.finfo(float).tiny))
        agc_db = float(np.clip(requested_gain_db, -cfg.agc_limit_db, cfg.agc_limit_db)) if cfg.agc_enabled else 0.
        y *= 10**(agc_db/20)
        self.index += n
        if not np.isfinite(y).all():
            raise FloatingPointError("nonfinite channel output")
        ps = float(np.mean(np.abs(signal)**2))
        pn = float(np.mean(np.abs(noise)**2))
        meta = dict(version=VERSION, config_hash=cfg.config_hash, seed=self.seed,
                    scenario=cfg.scenario, geometry=asdict(self.geometry),
                    fs_hz=cfg.fs_hz, fc_hz=cfg.fc_hz, receiver=asdict(self.receiver),
                    start_sample=self.index-n, state_probabilities=self.probs.tolist(),
                    state_counts={name: int(np.sum(state == i)) for i, name in enumerate(STATES)},
                    state_start=STATES[int(state[0])], state_end=STATES[int(state[-1])],
                    shadow_correlation_time_s=cfg.correlation_time_s,
                    scatter_max_doppler_hz=self.scatter_fd,
                    scatter_model="finite_gaussian_rays_Jakes_approximation",
                    total_diffuse_power_by_state=self.diffuse.tolist(),
                    expected_los_power_by_state=self.mean_los_power.tolist(),
                    expected_k_linear_by_state=(self.mean_los_power/self.diffuse).tolist(),
                    delays_samples_by_state=self.delays.tolist(),
                    delays_seconds_by_state=(self.delays/cfg.fs_hz).tolist(),
                    rms_delay_ns_by_state=list(cfg.rms_delay_ns),
                    diffuse_pdp=self.pdp.tolist(), boundary=boundary_used,
                    orbital_doppler_hz=self.f_doppler, input_oscillator_cfo_hz=self.receiver.oscillator_cfo_hz,
                    estimated_frequency_hz=self.f_hat if cfg.mode == "post_sync" else 0.,
                    output_frequency_hz=self.f_total-self.f_hat if cfg.mode == "post_sync" else self.f_total,
                    synchronization="sampled_estimation_error_proxy", mode=cfg.mode,
                    receiver_location=cfg.receiver_location,
                    link_direction="ground_to_satellite" if cfg.receiver_location == "satellite" else "satellite_to_ground",
                    input_processing_state=cfg.input_processing_state,
                    compensation_scope="newly_injected_channel_and_virtual_RX_only",
                    frequency_compensation_applied=cfg.mode == "post_sync",
                    iq_compensation_applied=cfg.mode == "post_sync" and cfg.iq_compensation_enabled,
                    phase_tracking_applied=phase_tracking_active,
                    receiver_lock_status=lock_status,
                    quality_snr_db=quality,
                    quality_source="simulated_signal_to_added_noise_not_total_snr" if quality_snr_db is None else "caller_supplied_receiver_quality",
                    residual_scale=residual_scale,
                    fine_frequency_tracking_applied=cfg.mode == "post_sync" and lock_status != "unlocked",
                    coarse_doppler_compensation_applied=cfg.mode == "post_sync",
                    processing_route="full", full_channel_waveform_synthesized=True,
                    channel_equalization_applied=eq_applied,
                    equalizer=eq_meta,
                    tx_impairment_correction_applied=False,
                    rff_protection_policy="no_TX_correction_propagation_EQ_opt_in",
                    fractional_delay_filter="kaiser_sinc_beta8.6",
                    numerical_filter_latency_samples=cfg.fractional_delay_half_length,
                    phase_tracking_residual_std_deg=cfg.phase_tracking_residual_std_deg,
                    effective_phase_tracking_residual_std_deg=cfg.phase_tracking_residual_std_deg*residual_scale if phase_tracking_active else None,
                    phase_tracking_model="correlated_estimation_error_proxy_not_PLL",
                    external_receiver_processing=external_processing,
                    snr_measurement_stage="after_parametric_compensation_before_AGC",
                    phase_noise_q_rad2_s=2*np.pi*cfg.phase_noise_linewidth_hz,
                    path_gain_db_start=float(20*np.log10(gain[0])),
                    common_delay_s=self.geometry.distance_m/C,
                    atmosphere_loss_db=cfg.atmosphere_loss_db,
                    atmosphere_source=cfg.atmosphere_source,
                    added_noise_variance_reference=noise_variance,
                    measured_added_noise_power_pre_agc=pn,
                    signal_to_added_noise_db=float(10*np.log10(ps/pn)) if ps > 0 and pn > 0 else None,
                    true_total_snr_known=False, agc_gain_db=agc_db,
                    agc_limited=bool(cfg.agc_enabled and abs(requested_gain_db) > cfg.agc_limit_db),
                    input_metrics=waveform_metrics(x), output_metrics=waveform_metrics(y),
                    parameter_provenance={"visibility": "Nie2012_semantic_state_mapping",
                        "geometry": self.geometry.source, "fading_rx_noise": "engineering_assumption",
                        "atmosphere": cfg.atmosphere_source})
        if return_traces:
            return y.astype(np.complex64), meta, dict(state=state, shadow_z=shadow,
                coefficients=fading, pre_agc_signal=signal, pre_agc_noise=noise)
        return y.astype(np.complex64), meta
