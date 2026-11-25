"""Propagation-only, trajectory-driven satellite channel prototype.

All fading, delay and correlation parameters must be supplied explicitly and
calibrated for the target environment. This is not a 3GPP/ITU conformance model.
"""
from dataclasses import dataclass
import math

import numpy as np

from .channel import C, fractional_delay_kernel, geometry_from_vectors, state_probabilities
from .equalization import linear_filter


@dataclass(frozen=True)
class StateParameters:
    """Direct amplitude in dB and diffuse tap powers relative to LOS reference.

    ``direct_mean_db=None`` removes the direct component. Delay and power tuples
    describe diffuse paths; their power entries are linear, not dB.
    """
    direct_mean_db: float | None
    direct_std_db: float
    diffuse_delays_s: tuple[float, ...]
    diffuse_powers: tuple[float, ...]

    def __post_init__(self):
        if self.direct_mean_db is not None and not math.isfinite(self.direct_mean_db):
            raise ValueError("direct mean must be finite")
        if not math.isfinite(self.direct_std_db) or self.direct_std_db < 0:
            raise ValueError("direct standard deviation must be nonnegative")
        if not self.diffuse_delays_s or len(self.diffuse_delays_s) != len(self.diffuse_powers):
            raise ValueError("provide equal nonempty diffuse delay and power lists")
        if any(not math.isfinite(v) or v < 0 for v in (*self.diffuse_delays_s, *self.diffuse_powers)):
            raise ValueError("delays and powers must be finite and nonnegative")


class TrajectoryChannel:
    """Sequential short blocks from one pass, with caller-provided geometry.

    State, shadow and diffuse coefficients retain correlation between calls.
    The caller supplies signal history and absolute input timestamps. The FIR
    adds ``half_length / fs_hz`` seconds common implementation latency.
    Absolute satellite propagation delay is a timestamp offset, not inserted
    into each input buffer. Intra-block geometry uses a linear range model.
    """

    def __init__(self, *, fs_hz, fc_hz, environment, state_parameters,
                 state_mix_time_s, shadow_corr_time_s, diffuse_corr_time_s,
                 gain_mode, reference_distance_m=None, half_length=24, seed=0):
        values = (fs_hz, fc_hz, state_mix_time_s, shadow_corr_time_s, diffuse_corr_time_s)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("frequency and correlation times must be positive")
        if environment not in ("urban", "suburban") or gain_mode not in ("relative", "absolute"):
            raise ValueError("invalid environment or gain mode")
        if gain_mode == "relative" and (reference_distance_m is None or
                not math.isfinite(reference_distance_m) or reference_distance_m <= 0):
            raise ValueError("relative gain requires reference distance")
        if len(state_parameters) != 3 or not all(isinstance(p, StateParameters) for p in state_parameters):
            raise ValueError("provide LOS, PARTIAL, BLOCKED StateParameters")
        if int(half_length) != half_length or half_length < 8:
            raise ValueError("half_length must be an integer >= 8")
        self.fs, self.fc, self.environment = fs_hz, fc_hz, environment
        self.parameters = tuple(state_parameters)
        self.mix, self.shadow_tau, self.diffuse_tau = state_mix_time_s, shadow_corr_time_s, diffuse_corr_time_s
        self.gain_mode, self.reference_distance = gain_mode, reference_distance_m
        self.half = int(half_length)
        self.rng = np.random.default_rng(seed)
        self.max_taps = max(len(p.diffuse_delays_s) for p in self.parameters)
        self.history_size = 2*self.half + max(int(math.floor(max(p.diffuse_delays_s)*fs_hz)) for p in self.parameters)
        self.last_time = None
        self.state = None
        self.shadow = None
        self.diffuse = None

    def _complex_normal(self):
        return (self.rng.standard_normal(self.max_taps) + 1j*self.rng.standard_normal(self.max_taps))/math.sqrt(2)

    def process(self, x, *, time_s, satellite_position_m, satellite_velocity_mps,
                terminal_position_m, terminal_velocity_mps, history, atmosphere_loss_db=0.0):
        x = np.asarray(x, dtype=np.complex128)
        history = np.asarray(history, dtype=np.complex128)
        if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("x must be a nonempty finite 1-D signal")
        if history.ndim != 1 or len(history) < self.history_size or not np.isfinite(history).all():
            raise ValueError("a finite preceding history of history_size samples is required")
        if not math.isfinite(time_s) or (self.last_time is not None and time_s <= self.last_time):
            raise ValueError("timestamps must strictly increase")
        if not math.isfinite(atmosphere_loss_db) or atmosphere_loss_db < 0:
            raise ValueError("atmosphere loss must be finite and nonnegative")
        geometry = geometry_from_vectors(satellite_position_m, satellite_velocity_mps,
                                         terminal_position_m, terminal_velocity_mps)
        probabilities = state_probabilities(geometry.elevation_deg, self.environment)
        t = np.arange(len(x))/self.fs
        distance = geometry.distance_m + geometry.range_rate_mps*t
        if np.any(distance <= 0):
            raise ValueError("linearized range is nonpositive")
        if self.last_time is None:
            self.state = int(self.rng.choice(3, p=probabilities))
            self.shadow = self.rng.standard_normal()
            self.diffuse = self._complex_normal()
        else:
            dt = time_s - self.last_time
            # At constant elevation, this kernel has the desired stationary law.
            if self.rng.random() >= math.exp(-dt/self.mix):
                self.state = int(self.rng.choice(3, p=probabilities))
            a = math.exp(-dt/self.shadow_tau)
            b = math.exp(-dt/self.diffuse_tau)
            self.shadow = a*self.shadow + math.sqrt(1-a*a)*self.rng.standard_normal()
            self.diffuse = b*self.diffuse + math.sqrt(1-b*b)*self._complex_normal()
        self.last_time = time_s
        params = self.parameters[self.state]
        kernel = np.zeros(self.history_size + 1, complex)
        if params.direct_mean_db is not None:
            direct = 10**((params.direct_mean_db + params.direct_std_db*self.shadow)/20)
            kernel[:2*self.half+1] += direct * fractional_delay_kernel(0.0, self.half)
        for i, (delay, power) in enumerate(zip(params.diffuse_delays_s, params.diffuse_powers)):
            samples = delay*self.fs
            integer = int(math.floor(samples))
            kernel[integer:integer+2*self.half+1] += (math.sqrt(power)*self.diffuse[i] *
                fractional_delay_kernel(samples-integer, self.half))
        filtered, _ = linear_filter(x, kernel, boundary="require_context", context=history)
        amplitude = (self.reference_distance/distance if self.gain_mode == "relative"
                     else C/(4*math.pi*self.fc*distance))
        phase = -2*math.pi*(self.fc/C)*(distance-geometry.distance_m)
        # Reference initial phase to distance; reduce cycles for numerical stability.
        phase += -2*math.pi*math.remainder(self.fc*geometry.distance_m/C, 1.0)
        y = filtered * amplitude * 10**(-atmosphere_loss_db/20) * np.exp(1j*phase)
        return y, {"state": ("LOS", "PARTIAL", "BLOCKED")[self.state],
                   "state_probabilities": probabilities, "distance_m": geometry.distance_m,
                   "elevation_deg": geometry.elevation_deg,
                   "doppler_hz": -self.fc*geometry.range_rate_mps/C,
                   "common_filter_delay_s": self.half/self.fs,
                   "absolute_propagation_delay_s": geometry.distance_m/C,
                   "gain_mode": self.gain_mode}
