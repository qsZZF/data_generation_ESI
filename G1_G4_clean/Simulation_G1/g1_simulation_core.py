from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import scipy.sparse

try:
    from .noise_covariance import (baseline_covariance_payload, baseline_sample_count,
                                   sensor_noise_with_baseline)
    from .spatial import (SpatialDynamicsConfig, gaussian_weights, patch_hop_order,
                          spatial_envelope, validate_disjoint_supports)
except ImportError:
    from noise_covariance import (baseline_covariance_payload, baseline_sample_count,
                                  sensor_noise_with_baseline)
    from spatial import (SpatialDynamicsConfig, gaussian_weights, patch_hop_order,
                         spatial_envelope, validate_disjoint_supports)


Array = np.ndarray


@dataclass
class G1SimulationConfig:
    sample_id: str = "G1_sample"
    template_id: str = "fsaverage5"
    source_space_id: str = "unknown_source_space"
    leadfield_id: str = "unknown_leadfield"
    modality: str = "EEG"
    srate: float = 250.0
    time_window: tuple[float, float] = (-0.2, 0.8)
    n_patch: int = 1
    patch_area_mm2: float | list[float] = 100.0
    decay_mm: float | list[float] = 15.0
    seed_vertices: list[int] | None = None
    min_seed_distance_mm: float = 20.0
    connectivity_mode: str = "independent"
    coupling: float | list[float] = 0.0
    waveform_params: list[tuple[str, dict[str, Any]]] = field(
        default_factory=lambda: [
            (
                "erp",
                {
                    "components": [
                        {
                            "type": "gaussian",
                            "latency_s": 0.1,
                            "width_s": 0.05,
                            "amplitude_rel": 1.0,
                        }
                    ],
                    "amplitude": 1.0,
                },
            )
        ]
    )
    source_background_type: str | None = None
    source_snr_db: float | None = None
    sensor_noise_type: str = "white"
    sensor_snr_db: float | None = 10.0
    sensor_snr_mode: str = "robust_global_rms"
    sensor_snr_clip_quantile: float = 0.95
    real_noise_id: str | None = None
    average_reference: bool = True
    normalize_source: bool = True
    normalize_sensor: bool = False
    random_seed: int | None = None
    spatial: SpatialDynamicsConfig = field(default_factory=SpatialDynamicsConfig)
    component_seeds: dict[str, int] | None = None
    patch_vertices: list[list[int]] | None = None
    patch_gains: list[float] | None = None
    sensor_noise_peak_ratio: float | None = None
    baseline_duration_s: float = 2.0
    source_model: str = 'patch'
    spectral: dict[str, Any] = field(default_factory=dict)

    def validate_recipe(self) -> None:
        """Validate opt-in replay fields without changing legacy RNG behavior."""
        baseline_sample_count(self.baseline_duration_s, self.srate)
        if self.source_model not in ('patch', 'lbo', 'lbo_network'):
            raise ValueError('source_model must be patch, lbo or lbo_network')
        if self.source_model == 'patch' and self.spectral:
            raise ValueError('spectral settings require source_model=lbo or lbo_network')
        if self.component_seeds is not None:
            expected = {"spatial", "waveform", "gain", "background", "sensor"}
            if not isinstance(self.component_seeds, dict) or set(self.component_seeds) != expected:
                raise ValueError("component_seeds requires exactly spatial, waveform, gain, background, sensor")
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
                   for v in self.component_seeds.values()):
                raise ValueError("component_seeds values must be nonnegative integers")
        if self.patch_gains is not None:
            gains = np.asarray(self.patch_gains)
            if (gains.shape != (self.n_patch,) or gains.dtype.kind not in "iuf"
                    or not np.isfinite(gains).all() or np.any(gains <= 0)):
                raise ValueError("patch_gains must contain n_patch finite positive numbers")
        if self.sensor_noise_peak_ratio is not None:
            ratio = self.sensor_noise_peak_ratio
            if (isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, (int, float, np.integer, np.floating))
                    or not np.isfinite(ratio) or ratio <= 0):
                raise ValueError("sensor_noise_peak_ratio must be finite and positive, or None")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def estimate_power_spectrum_summary(
    signal: Array,
    srate: float,
    fit_range_hz: tuple[float, float] = (2.0, 40.0),
    alpha_range_hz: tuple[float, float] = (8.0, 13.0),
) -> dict[str, float | None]:
    """Return compact PSD diagnostics for generated signals.

    The slope is fitted in log-log power space and is intended as a lightweight
    generator QC metric, not as a replacement for specparam or IRASA.
    """
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        traces = arr.reshape(1, -1)
    elif arr.ndim == 2:
        channel_power = np.mean(arr**2, axis=1)
        keep = channel_power > np.quantile(channel_power, 0.25)
        traces = arr[keep] if np.any(keep) else arr
    else:
        traces = arr.reshape(-1, arr.shape[-1])
    if traces.shape[-1] < 8 or not np.any(np.isfinite(traces)):
        return {"aperiodic_exponent_estimate": None, "alpha_relative_power": None, "peak_frequency_hz": None}
    nperseg = min(traces.shape[-1], max(8, int(round(2.0 * srate))))
    freqs, psd_by_trace = scipy.signal.welch(traces, fs=srate, nperseg=nperseg, axis=-1)
    psd = np.mean(psd_by_trace, axis=0)
    valid = (freqs >= fit_range_hz[0]) & (freqs <= fit_range_hz[1]) & np.isfinite(psd) & (psd > 0)
    exponent = None
    if int(np.sum(valid)) >= 5:
        slope, _ = np.polyfit(np.log10(freqs[valid]), np.log10(psd[valid]), 1)
        exponent = float(-slope)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total = float(integrate(psd[freqs > 0], freqs[freqs > 0]))
    alpha_mask = (freqs >= alpha_range_hz[0]) & (freqs <= alpha_range_hz[1])
    alpha_ratio = float(integrate(psd[alpha_mask], freqs[alpha_mask]) / total) if np.any(alpha_mask) and total > 0 else None
    peak_frequency = float(freqs[int(np.argmax(psd[1:])) + 1]) if psd.size > 1 else None
    return {
        "aperiodic_exponent_estimate": exponent,
        "alpha_relative_power": alpha_ratio,
        "peak_frequency_hz": peak_frequency,
    }


def normalize_cortex(
    vertices: Array,
    faces: Array,
    vert_conn: scipy.sparse.spmatrix | dict[str, Array],
) -> tuple[Array, Array, scipy.sparse.csc_matrix]:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T

    faces = np.asarray(faces, dtype=int)
    if faces.shape[0] == 3 and faces.shape[1] != 3:
        faces = faces.T
    if faces.min() == 1:
        faces = faces - 1

    if scipy.sparse.issparse(vert_conn):
        conn = vert_conn.tocsc()
    elif isinstance(vert_conn, dict) and {"data", "ir", "jc"}.issubset(vert_conn):
        conn = scipy.sparse.csc_matrix((vert_conn["data"], vert_conn["ir"], vert_conn["jc"]))
    else:
        conn = scipy.sparse.csc_matrix(vert_conn)

    return vertices, faces, conn


def mesh_vertex_areas(vertices: Array, faces: Array) -> tuple[Array, Array]:
    r12 = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    r13 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    face_area = 0.5 * np.linalg.norm(np.cross(r12, r13), axis=1)
    vertex_area = np.zeros(vertices.shape[0], dtype=float)
    for dim in range(3):
        np.add.at(vertex_area, faces[:, dim], face_area)
    return face_area, vertex_area / 3.0


def coordinate_unit_scale_to_mm(vertices: Array) -> float:
    # Freesurfer/Brainstorm surfaces are often in meters or millimeters.
    # Values below 10 are treated as meters and converted to millimeters.
    return 1000.0 if float(np.max(np.abs(vertices))) < 10.0 else 1.0


def _as_list(value: float | list[float], length: int) -> list[float]:
    if isinstance(value, list):
        if len(value) == length:
            return [float(v) for v in value]
        if len(value) == 1:
            return [float(value[0])] * length
        raise ValueError(f"Expected scalar, one-element list, or list of length {length}")
    return [float(value)] * length


def grow_patch_by_area(
    seed: int,
    vert_conn: scipy.sparse.csc_matrix,
    vertex_area_coord2: Array,
    target_area_mm2: float,
    coord_to_mm: float,
) -> Array:
    target_area_coord2 = target_area_mm2 / (coord_to_mm**2)
    patch = [int(seed)]
    current_area = float(vertex_area_coord2[seed])
    boundary = {int(seed)}
    visited = {int(seed)}

    while current_area < target_area_coord2:
        new_boundary: set[int] = set()
        for vertex in sorted(boundary):
            neighbors = vert_conn.indices[vert_conn.indptr[vertex] : vert_conn.indptr[vertex + 1]]
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_boundary.add(neighbor)
                patch.append(neighbor)
                current_area += float(vertex_area_coord2[neighbor])
                if current_area >= target_area_coord2:
                    return np.asarray(patch, dtype=int)
        if not new_boundary:
            break
        boundary = new_boundary

    return np.asarray(patch, dtype=int)


class TimeSeriesGenerator:
    def __init__(self, srate: float, time_window: tuple[float, float], rng: np.random.Generator):
        self.srate = float(srate)
        self.time_window = tuple(float(v) for v in time_window)
        self.n_time = int(round((self.time_window[1] - self.time_window[0]) * self.srate))
        if self.n_time <= 0:
            raise ValueError("time_window and srate produce no samples")
        self.time = np.linspace(self.time_window[0], self.time_window[1], self.n_time, endpoint=False)
        self.rng = rng

    def white(self, n_samples: int | None = None) -> Array:
        return self.rng.standard_normal(self.n_time if n_samples is None else int(n_samples))

    def pink(self, n_samples: int | None = None) -> Array:
        n_samples = self.n_time if n_samples is None else int(n_samples)
        n_fft = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))
        freqs = np.linspace(0.0, 1.0, n_fft // 2 + 1)
        with np.errstate(divide="ignore"):
            spectrum_scale = 1.0 / np.sqrt(freqs)
        spectrum_scale[0] = 0.0
        phases = np.exp(1j * 2.0 * np.pi * self.rng.random(len(freqs)))
        half = spectrum_scale * phases
        full = np.concatenate([half, np.conj(half[-2:0:-1])])
        signal = np.real(np.fft.ifft(full))
        return signal[:n_samples]

    def aperiodic_noise(
        self,
        n_samples: int | None = None,
        exponent: float = 1.0,
        white_floor: float = 0.08,
        low_freq_knee_hz: float = 0.5,
        envelope_depth: float = 0.15,
    ) -> Array:
        """Generate non-oscillatory 1/f-like activity with a broadband floor."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        n_fft = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.srate)
        spectral_power = 1.0 / (freqs + low_freq_knee_hz) ** exponent
        spectral_power[0] = 0.0
        spectral_amplitude = np.sqrt(spectral_power + white_floor)
        spectral_amplitude *= self.rng.lognormal(mean=0.0, sigma=0.20, size=len(freqs))
        phases = np.exp(1j * 2.0 * np.pi * self.rng.random(len(freqs)))
        spectrum = spectral_amplitude * phases
        spectrum[0] = 0.0
        signal = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
        if envelope_depth > 0:
            signal *= self._smooth_random_envelope(
                n_samples=n_samples,
                min_gain=max(0.2, 1.0 - envelope_depth),
                max_gain=1.0 + envelope_depth,
                control_interval_s=0.8,
            )
        signal -= float(np.mean(signal))
        return signal / max(float(np.std(signal)), 1e-12)

    def powerlaw_noise(
        self,
        n_samples: int | None = None,
        exponent: float = 1.0,
        offset: float = 0.0,
        knee_hz: float | None = None,
        white_floor: float = 0.02,
    ) -> Array:
        """Generate Timmer-Konig-style stochastic power-law activity."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        n_fft = 2 ** int(np.ceil(np.log2(max(n_samples, 8))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.srate)
        if knee_hz is None:
            power = np.power(np.maximum(freqs, 1.0 / (n_fft / self.srate)), -float(exponent))
        else:
            power = 1.0 / (float(knee_hz) + np.power(freqs, float(exponent)))
        power = np.power(10.0, float(offset)) * power + float(white_floor)
        power[0] = 0.0
        coeff = self.rng.standard_normal(len(freqs)) + 1j * self.rng.standard_normal(len(freqs))
        spectrum = coeff * np.sqrt(np.maximum(power, 0.0) / 2.0)
        spectrum[0] = 0.0
        if n_fft % 2 == 0:
            spectrum[-1] = np.real(spectrum[-1]) + 0j
        signal = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
        return self._standardize_vector(signal)

    def specparam_signal(
        self,
        n_samples: int | None = None,
        exponent: float = 1.0,
        offset: float = 0.0,
        knee_hz: float | None = None,
        peaks: list[dict[str, float]] | None = None,
        white_floor: float = 0.01,
    ) -> Array:
        """Synthesize a stochastic signal from specparam-like PSD parameters."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        n_fft = 2 ** int(np.ceil(np.log2(max(n_samples, 8))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.srate)
        safe_freqs = np.maximum(freqs, 1.0 / (n_fft / self.srate))
        if knee_hz is None:
            log_power = float(offset) - float(exponent) * np.log10(safe_freqs)
        else:
            log_power = float(offset) - np.log10(float(knee_hz) + np.power(safe_freqs, float(exponent)))
        for peak in peaks or []:
            center = float(peak.get("center_hz", peak.get("center", 10.0)))
            bandwidth = max(float(peak.get("bandwidth_hz", peak.get("bandwidth", 2.0))), 1e-6)
            height = float(peak.get("height", peak.get("power", 0.5)))
            log_power += height * np.exp(-((freqs - center) ** 2) / (2.0 * bandwidth**2))
        power = np.power(10.0, log_power) + float(white_floor)
        power[0] = 0.0
        coeff = self.rng.standard_normal(len(freqs)) + 1j * self.rng.standard_normal(len(freqs))
        spectrum = coeff * np.sqrt(np.maximum(power, 0.0) / 2.0)
        spectrum[0] = 0.0
        if n_fft % 2 == 0:
            spectrum[-1] = np.real(spectrum[-1]) + 0j
        signal = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
        return self._standardize_vector(signal)

    def synaptic_shot_noise(
        self,
        n_samples: int | None = None,
        event_rate_hz: float = 35.0,
        tau_rise_s: float = 0.005,
        tau_decay_s: float = 0.05,
        amplitude_scale: float = 1.0,
    ) -> Array:
        """Generate Poisson synaptic events convolved with a rise-decay kernel."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        events = self.rng.poisson(float(event_rate_hz) / self.srate, n_samples).astype(float)
        events *= self.rng.lognormal(mean=0.0, sigma=0.45, size=n_samples)
        kernel_t = np.arange(max(4, int(round(1.0 * self.srate)))) / self.srate
        kernel = np.exp(-kernel_t / max(tau_decay_s, 1e-6)) - np.exp(-kernel_t / max(tau_rise_s, 1e-6))
        kernel[kernel < 0.0] = 0.0
        kernel /= max(float(np.max(kernel)), 1e-12)
        signal = scipy.signal.fftconvolve(events, kernel, mode="full")[:n_samples]
        signal *= float(amplitude_scale)
        return self._standardize_vector(signal)

    def broadband_noise(self, n_samples: int | None = None) -> Array:
        """Generate broadband non-rhythmic noise with mild temporal coloring."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        raw = self.white(n_samples + int(self.srate))
        nyquist = self.srate / 2.0
        high = min(45.0, nyquist - 1e-6)
        if high <= 1.0:
            signal = raw[-n_samples:]
        else:
            b, a = scipy.signal.butter(2, [0.5 / nyquist, high / nyquist], btype="bandpass")
            signal = scipy.signal.lfilter(b, a, raw)[-n_samples:]
        drift = self.aperiodic_noise(n_samples, exponent=1.8, white_floor=0.01, envelope_depth=0.05)
        signal = signal + 0.15 * drift
        signal -= float(np.mean(signal))
        return signal / max(float(np.std(signal)), 1e-12)

    def eegsourcesim_resting_noise(
        self,
        n_samples: int | None = None,
        exponent: float = 1.0,
        alpha_peak_hz: float = 10.0,
        alpha_width_hz: float = 1.8,
        alpha_gain: float = 1.0,
        white_floor: float = 0.02,
        low_freq_knee_hz: float = 0.4,
        envelope_depth: float = 0.0,
        alpha_fraction: float = 0.20,
    ) -> Array:
        """Generate resting EEG with controlled 1/f background and a local alpha bump."""
        n_samples = self.n_time if n_samples is None else int(n_samples)
        aperiodic = self._freq_shaped_noise(
            n_samples=n_samples,
            exponent=exponent,
            white_floor=white_floor,
            low_freq_knee_hz=low_freq_knee_hz,
        )
        alpha = self._alpha_bump_noise(
            n_samples=n_samples,
            peak_hz=alpha_peak_hz,
            width_hz=alpha_width_hz,
        )
        alpha_weight = float(np.clip(alpha_fraction * alpha_gain, 0.0, 0.55))
        signal = (1.0 - alpha_weight) * aperiodic + alpha_weight * alpha
        if envelope_depth > 0:
            signal *= self._smooth_random_envelope(
                n_samples=n_samples,
                min_gain=max(0.1, 1.0 - envelope_depth),
                max_gain=1.0 + envelope_depth,
                control_interval_s=0.9,
            )
        signal -= float(np.mean(signal))
        return signal / max(float(np.std(signal)), 1e-12)

    def waveform(self, waveform_type: str, params: dict[str, Any] | None = None) -> Array:
        params = params or {}
        method = getattr(self, f"_wave_{waveform_type.lower()}", None)
        if method is None:
            if waveform_type.lower().startswith("resting_"):
                return self._wave_resting(params | {"band": waveform_type.lower().replace("resting_", "")})
            raise ValueError(f"Unsupported waveform type: {waveform_type}")
        return method(params)

    def _wave_pink(self, params: dict[str, Any]) -> Array:
        return self._scale(self.pink(), params)

    def _wave_white(self, params: dict[str, Any]) -> Array:
        return self._scale(self.white(), params)

    def _wave_powerlaw(self, params: dict[str, Any]) -> Array:
        wave = self.powerlaw_noise(
            exponent=float(params.get("exponent", 1.0)),
            offset=float(params.get("offset", 0.0)),
            knee_hz=params.get("knee_hz", None),
            white_floor=float(params.get("white_floor", 0.02)),
        )
        return self._scale(wave, params)

    def _wave_knee_powerlaw(self, params: dict[str, Any]) -> Array:
        wave = self.powerlaw_noise(
            exponent=float(params.get("exponent", 1.2)),
            offset=float(params.get("offset", 0.0)),
            knee_hz=float(params.get("knee_hz", 1.0)),
            white_floor=float(params.get("white_floor", 0.02)),
        )
        return self._scale(wave, params)

    def _wave_specparam(self, params: dict[str, Any]) -> Array:
        wave = self.specparam_signal(
            exponent=float(params.get("exponent", 1.0)),
            offset=float(params.get("offset", 0.0)),
            knee_hz=params.get("knee_hz", None),
            peaks=params.get("peaks", [{"center_hz": 10.0, "bandwidth_hz": 1.8, "height": 0.6}]),
            white_floor=float(params.get("white_floor", 0.01)),
        )
        envelope_depth = float(params.get("envelope_depth", 0.0))
        if envelope_depth > 0:
            wave *= self._smooth_random_envelope(
                n_samples=self.n_time,
                min_gain=max(0.1, 1.0 - envelope_depth),
                max_gain=1.0 + envelope_depth,
                control_interval_s=float(params.get("control_interval_s", 0.8)),
            )
        return self._scale(wave, params)

    def _wave_synaptic_noise(self, params: dict[str, Any]) -> Array:
        wave = self.synaptic_shot_noise(
            event_rate_hz=float(params.get("event_rate_hz", 35.0)),
            tau_rise_s=float(params.get("tau_rise_s", 0.005)),
            tau_decay_s=float(params.get("tau_decay_s", 0.05)),
            amplitude_scale=float(params.get("amplitude_scale", 1.0)),
        )
        return self._scale(wave, params)

    def _wave_oscillation(self, params: dict[str, Any]) -> Array:
        freq = float(params.get("frequency_hz", params.get("freq", 10.0)))
        phase = float(params.get("phase_rad", self.rng.uniform(0.0, 2.0 * np.pi)))
        freq_jitter = float(params.get("frequency_jitter_hz", 0.0))
        if freq_jitter > 0:
            inst_freq = freq + freq_jitter * self._smooth_random_envelope(self.n_time, -1.0, 1.0, 0.5)
            phase_series = phase + 2.0 * np.pi * np.cumsum(inst_freq) / self.srate
            wave = np.sin(phase_series)
        else:
            wave = np.sin(phase + 2.0 * np.pi * freq * self.time)
        if params.get("envelope", "smooth") == "smooth":
            wave *= self._smooth_random_envelope(
                n_samples=self.n_time,
                min_gain=float(params.get("min_gain", 0.35)),
                max_gain=float(params.get("max_gain", 1.2)),
                control_interval_s=float(params.get("control_interval_s", 0.5)),
            )
        return self._scale(wave, params)

    def _wave_burst_oscillation(self, params: dict[str, Any]) -> Array:
        params = dict(params)
        params["envelope"] = "none"
        wave = self._wave_oscillation(params)
        env = self._random_burst_envelope(
            duty_cycle=float(params.get("duty_cycle", 0.45)),
            min_duration_s=float(params.get("min_duration_s", 0.08)),
            max_duration_s=float(params.get("max_duration_s", 0.35)),
            min_gain=float(params.get("min_gain", 0.05)),
        )
        return wave * env

    def _wave_nonsinusoidal_oscillation(self, params: dict[str, Any]) -> Array:
        freq = float(params.get("frequency_hz", params.get("freq", 10.0)))
        phase = float(params.get("phase_rad", self.rng.uniform(0.0, 2.0 * np.pi)))
        harmonics = params.get("harmonics", {2: 0.35, 3: 0.15})
        wave = np.sin(phase + 2.0 * np.pi * freq * self.time)
        for harmonic, rel_amp in dict(harmonics).items():
            wave += float(rel_amp) * np.sin(phase + 2.0 * np.pi * freq * float(harmonic) * self.time)
        wave *= self._smooth_random_envelope(
            n_samples=self.n_time,
            min_gain=float(params.get("min_gain", 0.4)),
            max_gain=float(params.get("max_gain", 1.2)),
            control_interval_s=float(params.get("control_interval_s", 0.6)),
        )
        return self._scale(wave, params)

    def _wave_erp(self, params: dict[str, Any]) -> Array:
        wave = np.zeros(self.n_time)
        components = params.get(
            "components",
            [{"type": "gaussian", "latency_s": self._center_time(), "width_s": 0.05, "amplitude_rel": 1.0}],
        )
        for component in components:
            component_type = component.get("type", "gaussian").lower()
            latency = float(component.get("latency_s", component.get("latency", self._center_time())))
            amp = float(component.get("amplitude_rel", component.get("amp", 1.0)))
            if component_type == "gabor":
                width = float(component.get("std_dev_s", component.get("width_s", component.get("width", 0.05))))
                freq = float(component.get("frequency_Hz", component.get("freq", 10.0)))
                phase = float(component.get("phase_rad", component.get("phase", 0.0)))
                env = np.exp(-((self.time - latency) ** 2) / (2.0 * width**2))
                wave += amp * env * np.sin(2.0 * np.pi * freq * (self.time - latency) + phase)
            elif component_type == "gaussian":
                width = float(component.get("width_s", component.get("width", 0.05)))
                wave += amp * np.exp(-((self.time - latency) ** 2) / (2.0 * width**2))
            else:
                raise ValueError(f"Unsupported ERP component type: {component_type}")
        return self._scale(wave, params)

    def _wave_gaussian_pulse(self, params: dict[str, Any]) -> Array:
        center = float(params.get("pulse_center_time_s", self._center_time()))
        width = float(params.get("pulse_width_time_s", 0.1 * (self.time_window[1] - self.time_window[0])))
        wave = np.exp(-((self.time - center) ** 2) / (2.0 * width**2))
        return self._scale(wave, params)

    def _wave_ersp(self, params: dict[str, Any]) -> Array:
        base = params.get("base_frequency_band_Hz", params.get("freq", 10.0))
        phase = float(params.get("base_phase_rad", 0.0))
        if np.isscalar(base):
            carrier = np.sin(phase + 2.0 * np.pi * float(base) * self.time)
        else:
            band = np.asarray(base, dtype=float).reshape(-1)
            if band.size == 4:
                low, high = float(band[1]), float(band[2])
            elif band.size == 2:
                low, high = float(band[0]), float(band[1])
            else:
                raise ValueError("base_frequency_band_Hz must be scalar, 2-element, or 4-element")
            carrier = self._band_limited_noise(low, high)

        modulation = params.get("modulation", {})
        mod_type = modulation.get("type", params.get("mod_type", "burst")).lower()
        min_amp = float(modulation.get("min_rel_amplitude", params.get("min_amp", 0.1)))
        env = np.ones(self.n_time) * min_amp
        if mod_type in {"burst", "invburst"}:
            latency = float(modulation.get("latency_s", params.get("latency", self._center_time())))
            width = float(modulation.get("width_s", params.get("width", 0.2)))
            taper = float(modulation.get("taper", 0.25))
            mask = (self.time >= latency - width / 2.0) & (self.time <= latency + width / 2.0)
            if np.any(mask):
                burst = scipy.signal.windows.tukey(int(mask.sum()), alpha=taper)
                if mod_type == "invburst":
                    burst = 1.0 - burst
                env[mask] = min_amp + (1.0 - min_amp) * burst
        elif mod_type == "ampmod":
            freq = float(modulation.get("mod_frequency_Hz", 1.0))
            mod_phase = float(modulation.get("mod_phase_rad", 0.0))
            mod = np.sin(mod_phase + 2.0 * np.pi * freq * self.time)
            env = min_amp + (1.0 - min_amp) * (mod - mod.min()) / max(mod.max() - mod.min(), 1e-12)
        else:
            raise ValueError(f"Unsupported ERSP modulation type: {mod_type}")

        wave = float(params.get("base_amplitude", 1.0)) * carrier * env
        return self._scale(wave, params)

    def _wave_spike(self, params: dict[str, Any]) -> Array:
        latency = float(params.get("latency_s", params.get("latency", self._center_time())))
        duration = float(params.get("duration_s", params.get("duration", 0.07)))
        shape = params.get("shape", "exponential_decay").lower()
        if shape == "triangular":
            wave = np.zeros(self.n_time)
            half = duration / 2.0
            rising = (self.time >= latency - half) & (self.time <= latency)
            falling = (self.time > latency) & (self.time <= latency + half)
            wave[rising] = (self.time[rising] - (latency - half)) / max(half, 1e-12)
            wave[falling] = 1.0 - (self.time[falling] - latency) / max(half, 1e-12)
        elif shape == "exponential_decay":
            tau = float(params.get("decay_tau_s", duration / 4.0))
            wave = np.exp(-np.abs(self.time - latency) / max(tau, 1e-12))
            wave[np.abs(self.time - latency) > 0.75 * duration] = 0.0
        else:
            raise ValueError(f"Unsupported spike shape: {shape}")
        return self._scale(wave, params)

    def _wave_resting(self, params: dict[str, Any]) -> Array:
        band = params.get("band", "alpha").lower()
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
        low, high = bands.get(band, (8.0, 13.0))
        wave = self._band_limited_noise(low, high)
        envelope = params.get("envelope", "smooth")
        if envelope == "smooth":
            wave *= self._smooth_random_envelope(
                n_samples=self.n_time,
                min_gain=float(params.get("min_gain", 0.35)),
                max_gain=float(params.get("max_gain", 1.2)),
                control_interval_s=float(params.get("control_interval_s", 0.35)),
            )
        elif envelope == "burst":
            burst = params.get("burst_properties", {})
            wave *= self._random_burst_envelope(
                duty_cycle=float(burst.get("duty_cycle", params.get("duty_cycle", 0.7))),
                min_duration_s=float(burst.get("min_duration_s", 0.2)),
                max_duration_s=float(burst.get("max_duration_s", 0.8)),
                min_gain=float(burst.get("min_gain", 0.25)),
            )
        elif envelope in {None, "none"}:
            pass
        else:
            raise ValueError(f"Unsupported resting envelope: {envelope}")
        return self._scale(wave, params)

    def _wave_resting_pinknoise(self, params: dict[str, Any]) -> Array:
        return self._scale(self.pink(), params)

    def _wave_eegsourcesim_resting_noise(self, params: dict[str, Any]) -> Array:
        wave = self.eegsourcesim_resting_noise(
            exponent=float(params.get("exponent", 1.0)),
            alpha_peak_hz=float(params.get("alpha_peak_hz", 10.0)),
            alpha_width_hz=float(params.get("alpha_width_hz", 1.8)),
            alpha_gain=float(params.get("alpha_gain", 1.0)),
            white_floor=float(params.get("white_floor", 0.02)),
            low_freq_knee_hz=float(params.get("low_freq_knee_hz", 0.5)),
            envelope_depth=float(params.get("envelope_depth", 0.0)),
            alpha_fraction=float(params.get("alpha_fraction", 0.25)),
        )
        return self._scale(wave, params)

    def _wave_ar(self, params: dict[str, Any]) -> Array:
        if "ar_coeffs" in params:
            coeffs = np.asarray(params["ar_coeffs"], dtype=float).reshape(-1)
        else:
            freq = float(params.get("freq", params.get("center_frequency_Hz", 10.0)))
            radius = float(params.get("radius", 0.95))
            theta = 2.0 * np.pi * freq / self.srate
            coeffs = np.asarray([2.0 * radius * np.cos(theta), -(radius**2)])
        burn = int(params.get("burn_in_samples", 100))
        innovations = self.rng.standard_normal(self.n_time + burn) * np.sqrt(float(params.get("innovation_variance", 1.0)))
        signal = scipy.signal.lfilter([1.0], np.concatenate([[1.0], -coeffs]), innovations)[burn:]
        return self._scale(signal, params)

    def _band_limited_noise(self, low_hz: float, high_hz: float) -> Array:
        nyquist = self.srate / 2.0
        low_hz = max(float(low_hz), 1e-6)
        high_hz = min(float(high_hz), nyquist - 1e-6)
        if low_hz >= high_hz:
            return self.pink()
        b, a = scipy.signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
        return scipy.signal.lfilter(b, a, self.pink(self.n_time + int(self.srate)))[int(self.srate) :]

    def _freq_shaped_noise(
        self,
        n_samples: int,
        exponent: float = 1.0,
        white_floor: float = 0.02,
        low_freq_knee_hz: float = 0.4,
    ) -> Array:
        n_fft = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.srate)
        amp = 1.0 / np.power(freqs + low_freq_knee_hz, exponent / 2.0)
        amp[0] = 0.0
        amp = np.sqrt(amp**2 + white_floor)
        phases = np.exp(1j * 2.0 * np.pi * self.rng.random(len(freqs)))
        spectrum = amp * phases
        spectrum[0] = 0.0
        signal = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
        return signal / max(float(np.std(signal)), 1e-12)

    def _alpha_bump_noise(self, n_samples: int, peak_hz: float = 10.0, width_hz: float = 1.8) -> Array:
        alpha_low = max(1.0, peak_hz - width_hz)
        alpha_high = min(self.srate / 2.0 - 1e-6, peak_hz + width_hz)
        return self._band_limited_noise(alpha_low, alpha_high)

    def _smooth_random_envelope(
        self,
        n_samples: int,
        min_gain: float = 0.35,
        max_gain: float = 1.2,
        control_interval_s: float = 0.35,
    ) -> Array:
        n_controls = max(4, int(np.ceil((n_samples / self.srate) / max(control_interval_s, 1e-3))) + 2)
        control_x = np.linspace(0, n_samples - 1, n_controls)
        control_y = self.rng.uniform(min_gain, max_gain, n_controls)
        envelope = np.interp(np.arange(n_samples), control_x, control_y)
        kernel = max(5, int(round(0.12 * self.srate)) | 1)
        kernel = min(kernel, n_samples if n_samples % 2 == 1 else n_samples - 1)
        if kernel >= 5:
            envelope = scipy.signal.savgol_filter(envelope, window_length=kernel, polyorder=2, mode="interp")
        return np.clip(envelope, min_gain, max_gain)

    def _random_burst_envelope(
        self,
        duty_cycle: float,
        min_duration_s: float,
        max_duration_s: float,
        min_gain: float = 0.25,
    ) -> Array:
        env = np.ones(self.n_time) * min_gain
        idx = 0
        while idx < self.n_time:
            duration = self.rng.uniform(min_duration_s, max_duration_s)
            width = max(1, int(round(duration * self.srate)))
            end = min(self.n_time, idx + width)
            if self.rng.random() < duty_cycle:
                env[idx:end] = 1.0
            idx = end
        if np.any(env):
            env = scipy.signal.medfilt(env, kernel_size=min(21, max(3, self.n_time // 10 * 2 + 1)))
        return env

    def _center_time(self) -> float:
        return 0.5 * (self.time_window[0] + self.time_window[1])

    @staticmethod
    def _standardize_vector(signal: Array) -> Array:
        signal = np.asarray(signal, dtype=float)
        signal = signal - float(np.mean(signal))
        return signal / max(float(np.std(signal)), 1e-12)

    @staticmethod
    def _scale(wave: Array, params: dict[str, Any]) -> Array:
        amplitude = float(params.get("amplitude", 1.0))
        wave = np.asarray(wave, dtype=float)
        if bool(params.get("normalize", False)):
            wave = wave / max(float(np.max(np.abs(wave))), 1e-12)
        return amplitude * wave


class G1SourceSimulator:
    def __init__(
        self,
        leadfield: Array,
        vertices: Array,
        faces: Array,
        vert_conn: scipy.sparse.spmatrix | dict[str, Array],
    ):
        self.leadfield = np.asarray(leadfield, dtype=float)
        self.vertices, self.faces, self.vert_conn = normalize_cortex(vertices, faces, vert_conn)
        if self.leadfield.shape[1] != self.vertices.shape[0]:
            raise ValueError("leadfield column count must match number of source vertices")
        self.coord_to_mm = coordinate_unit_scale_to_mm(self.vertices)
        _, self.vertex_area_coord2 = mesh_vertex_areas(self.vertices, self.faces)
        self.n_sources = self.vertices.shape[0]
        self.n_sensors = self.leadfield.shape[0]

    @classmethod
    def from_brainstorm_cortex(cls, leadfield: Array, cortex: dict[str, Any]) -> "G1SourceSimulator":
        return cls(leadfield, cortex["Vertices"], cortex["Faces"], cortex["VertConn"])

    def generate_source(self, config: G1SimulationConfig) -> tuple[Array, dict[str, Any]]:
        source, meta, _, _ = self._generate_source_components(config)
        return source, meta

    def _generate_source_components(self, config: G1SimulationConfig) -> tuple[Array, dict[str, Any], Array | None, Array | None]:
        """Generate baseline and event background jointly with common spatial mixing."""
        config.validate_recipe()
        if config.source_model != 'patch':
            try:
                from .spectral import generate_source_components
            except ImportError:
                from spectral import generate_source_components
            source, details, background, baseline = generate_source_components(self, config)
            metadata = {**asdict(config), **details, 'generator_type': 'G1_spectral',
                        'storage_policy': 'metadata_first', 'n_sources': self.n_sources,
                        'n_sensors': self.n_sensors, 'seed_vertices': [], 'patches': [],
                        'active_vertices': details['active_vertices'],
                        'source_background_saved': background is not None}
            return source, _json_safe(metadata), background, baseline
        rng = np.random.default_rng(config.random_seed)
        ts = TimeSeriesGenerator(config.srate, config.time_window, rng)
        source = np.zeros((self.n_sources, ts.n_time), dtype=np.float32)
        streams = ({name: np.random.default_rng(seed) for name, seed in config.component_seeds.items()}
                   if config.component_seeds is not None else {})
        seeds, patch_indices, patch_weights = self._resolve_spatial_patches(config, streams.get("spatial", rng))
        roi_signals = self._build_roi_signals(config, ts, rng, len(seeds))

        patch_gains = (np.asarray(config.patch_gains, dtype=float) if config.patch_gains is not None
                       else streams.get("gain", rng).uniform(0.9, 1.1, len(seeds)))
        spatial_metadata = []
        for idx, vertices in enumerate(patch_indices):
            area = float(self.vertex_area_coord2[vertices].sum() * self.coord_to_mm**2)
            hops = patch_hop_order(vertices, int(seeds[idx]), self.vert_conn)
            weights, details = spatial_envelope(patch_weights[idx], np.arange(ts.n_time)/config.srate,
                config.spatial, area_mm2=area, hop_order=hops)
            source[vertices, :] = (weights * (roi_signals[idx] * patch_gains[idx])[None, :]).astype(np.float32)
            spatial_metadata.append(details)

        # Dynamic envelopes can introduce DC; calibrate background to the exported AC signal.
        source -= source.mean(axis=1, keepdims=True)
        foreground_power = None
        foreground_source = None
        source_background = None
        baseline_background = None
        if config.source_snr_db is not None and config.source_background_type is not None:
            foreground_source = source.copy()
            foreground_power = float(np.mean(source.astype(float)**2))
            background_rng = streams.get("background", rng)
            background_ts = (TimeSeriesGenerator(config.srate, config.time_window, background_rng)
                             if config.component_seeds is not None else ts)
            baseline_samples = baseline_sample_count(config.baseline_duration_s, config.srate)
            joint_background = self._make_source_background(
                source, background_ts, background_rng, config.source_background_type,
                config.source_snr_db, baseline_samples=baseline_samples)
            baseline_background = joint_background[:, :baseline_samples].astype(np.float32)
            source_background = joint_background[:, baseline_samples:]
            source += source_background.astype(np.float32)

        source -= source.mean(axis=1, keepdims=True)
        source_scale = 1.0
        if config.normalize_source:
            source_scale = max(float(np.max(np.abs(source))), 1e-12)
            source /= source_scale

        exported_background = None
        if foreground_source is not None:
            foreground_source -= foreground_source.mean(axis=1, keepdims=True)
            foreground_source /= source_scale
            # The residual includes float32 addition/normalization rounding in the actual source.
            exported_background = source.astype(float) - foreground_source.astype(float)
            baseline_background /= source_scale

        meta = self._source_metadata(config, seeds, patch_indices, patch_weights, patch_gains, source_background)
        for patch, details in zip(meta["patches"], spatial_metadata):
            patch.update(area_mm2=details["area_mm2"], area_cm2=details["area_mm2"]/100,
                         hop_order=details["hop_order"], spatial_dynamics=details,
                         spatial_weights_role="base_before_temporal_envelope")
        meta.update(foreground_overlap_policy="disjoint_full_supports",
                    spatial_transform_order="patch_envelope_then_temporal_demean_then_background_snr_then_optional_global_normalization",
                    source_snr_definition="whole-mesh mean vertex power after temporal demeaning")
        if source_background is not None and foreground_power > 0:
            noise_power = float(np.mean(source_background**2))
            meta["realized_source_snr_db"] = (float(10*np.log10(foreground_power/noise_power))
                                                if noise_power > 0 else None)
        return source, meta, exported_background, baseline_background

    def generate_sensor(
        self,
        config: G1SimulationConfig,
        real_sensor_noise: Array | None = None,
        *,
        compute_diagnostics: bool = True,
    ) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        """Generate a sample; optional peak limiting may increase the requested SNR.

        Peak limiting rescales the entire noise after referencing, never clips it.
        Disabling diagnostics skips PSD summaries; covariance and SNR metadata remain.
        noise_covariance is estimated from the separate baseline_sensor segment.
        oracle_noise_covariance describes event background plus sensor noise.
        """
        baseline_samples = baseline_sample_count(config.baseline_duration_s, config.srate)
        baseline_background = None
        if config.source_model != 'patch' or (config.source_snr_db is not None and config.source_background_type is not None):
            source, meta, source_background, baseline_background = self._generate_source_components(config)
        else:
            source, meta = self.generate_source(config)
            source_background = None
        clean = self.leadfield @ source
        clean = clean - clean.mean(axis=1, keepdims=True)
        average_reference = config.average_reference and config.modality.upper() == "EEG"
        if average_reference:
            clean -= clean.mean(axis=0, keepdims=True)
        background_sensor = None
        baseline_sensor = np.zeros((clean.shape[0], baseline_samples))
        if source_background is not None:
            background_sensor = self.leadfield @ source_background
            background_sensor -= background_sensor.mean(axis=1, keepdims=True)
            if average_reference:
                background_sensor -= background_sensor.mean(axis=0, keepdims=True)
            baseline_sensor = self.leadfield @ baseline_background
            baseline_sensor -= baseline_sensor.mean(axis=1, keepdims=True)
            if average_reference:
                baseline_sensor -= baseline_sensor.mean(axis=0, keepdims=True)
        meta["reference_mode"] = "average" if average_reference else "as_supplied"
        noise = np.zeros_like(clean)
        baseline_noise = np.zeros_like(baseline_sensor)
        baseline_metadata = {'source_background_jointly_generated': source_background is not None,
                             'baseline_duration_s': config.baseline_duration_s,
                             'baseline_samples': baseline_samples, 'srate_hz': config.srate,
                             'baseline_interval_samples': [0, baseline_samples],
                             'event_interval_samples': [baseline_samples, baseline_samples + clean.shape[1]],
                             'foreground_present': False}
        if config.sensor_snr_db is not None:
            sensor_seed = (config.component_seeds["sensor"] if config.component_seeds is not None
                           else None if config.random_seed is None else config.random_seed + 1)
            noise, baseline_noise, noise_info = sensor_noise_with_baseline(
                self, clean, srate=config.srate, sensor_noise_type=config.sensor_noise_type,
                snr_db=config.sensor_snr_db, seed=sensor_seed, baseline_samples=baseline_samples,
                snr_mode=config.sensor_snr_mode,
                clip_quantile=config.sensor_snr_clip_quantile,
                real_sensor_noise=real_sensor_noise,
                average_reference=average_reference,
            )
            baseline_metadata.update(noise_info)
        peak_scale = 1.0
        if config.sensor_noise_peak_ratio is not None:
            noise_peak = float(np.max(np.abs(noise)))
            if noise_peak > 0:
                peak_scale = min(1.0, config.sensor_noise_peak_ratio * float(np.max(np.abs(clean))) / noise_peak)
                noise *= peak_scale
                baseline_noise *= peak_scale
        baseline_sensor += baseline_noise
        meta["noise_peak_limit_applied"] = peak_scale < 1.0
        meta["noise_peak_limit_scale"] = peak_scale
        sensor = clean + noise
        if config.normalize_sensor:
            scale = max(float(np.max(np.abs(sensor))), 1e-12)
            sensor = sensor / scale
            clean = clean / scale
            noise = noise / scale
            baseline_sensor /= scale
            if background_sensor is not None:
                background_sensor /= scale
            meta["sensor_normalization_scale"] = scale
        meta["sensor_shape"] = list(sensor.shape)
        meta["source_shape"] = list(source.shape)
        meta["clean_sensor_power"] = float(np.mean(clean**2))
        meta["noise_sensor_power"] = float(np.mean(noise**2))
        meta["sensor_snr_mode"] = config.sensor_snr_mode
        power_floor = np.finfo(float).tiny
        noise_log_power = np.log10(max(float(np.mean(noise**2)), power_floor))
        meta["realized_sensor_snr_db"] = float(10.0 * (np.log10(max(float(np.mean(clean**2)), power_floor)) - noise_log_power))
        meta["realized_sensor_snr_reference_db"] = float(
            10.0
            * (
                np.log10(max(self._snr_reference_power(clean, config.sensor_snr_mode, config.sensor_snr_clip_quantile), power_floor))
                - noise_log_power
            )
        )
        gfp = np.sqrt(np.mean(clean**2, axis=0))
        event_mask = gfp >= 0.2 * float(gfp.max())
        meta["event_sensor_snr_mask_rule"] = "GFP >= 0.2 * max(GFP) of referenced clean sensor signal"
        meta["event_sensor_snr_db"] = float(10 * (
            np.log10(max(float(np.mean(clean[:, event_mask]**2)), power_floor))
            - np.log10(max(float(np.mean(noise[:, event_mask]**2)), power_floor))))
        if compute_diagnostics:
            meta["source_psd_summary"] = estimate_power_spectrum_summary(source, config.srate)
            meta["clean_sensor_psd_summary"] = estimate_power_spectrum_summary(clean, config.srate)
            meta["noise_sensor_psd_summary"] = estimate_power_spectrum_summary(noise, config.srate)
        meta["real_sensor_noise_used"] = real_sensor_noise is not None
        noise = noise.astype(np.float32)
        covariance, covariance_info = baseline_covariance_payload(
            baseline_sensor, noise, background_sensor=background_sensor,
            average_reference=average_reference, baseline_metadata=baseline_metadata)
        meta.update({name: values.tolist() for name, values in covariance.items()})
        meta.update(covariance_info)
        return sensor.astype(np.float32), clean.astype(np.float32), noise, source.astype(np.float32), meta

    def manifest_only(self, config: G1SimulationConfig) -> dict[str, Any]:
        _, _, _, _, meta = self.generate_sensor(config)
        return meta

    def save_sample(
        self,
        output_dir: str | Path,
        config: G1SimulationConfig,
        real_sensor_noise: Array | None = None,
        save_sensor: bool = False,
        save_clean_sensor: bool = False,
        save_sensor_noise: bool = True,
        save_source: bool = False,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sensor, clean, noise, source, manifest = self.generate_sensor(config, real_sensor_noise=real_sensor_noise)

        sample_id = config.sample_id
        covariance_path = output_dir / f"{sample_id}_noise_covariance.npz"
        np.savez_compressed(covariance_path, **{
            key: np.asarray(manifest.pop(key), dtype=np.float32 if key == "baseline_sensor" else np.float64)
            for key in ("baseline_sensor", "noise_covariance", "oracle_noise_covariance", "sensor_noise_covariance")})
        manifest["noise_covariance_file"] = covariance_path.name
        if save_sensor:
            path = output_dir / f"{sample_id}_sensor.npz"
            np.savez_compressed(path, sensor=sensor)
            manifest["sensor_data"] = path.name
        if save_clean_sensor:
            path = output_dir / f"{sample_id}_sensor_clean.npz"
            np.savez_compressed(path, sensor_clean=clean)
            manifest["sensor_clean"] = path.name
        if save_sensor_noise:
            path = output_dir / f"{sample_id}_sensor_noise.npz"
            np.savez_compressed(path, sensor_noise=noise)
            manifest["sensor_noise"] = path.name
        if save_source:
            path = output_dir / f"{sample_id}_source.npz"
            np.savez_compressed(path, source=source)
            manifest["source_truth"] = path.name

        manifest_path = output_dir / f"{sample_id}_manifest.json"
        manifest["manifest"] = manifest_path.name
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
        return manifest

    def _resolve_spatial_patches(self, config: G1SimulationConfig, rng: np.random.Generator):
        """Pack complete patches; never silently crop a source to remove overlap."""
        config.validate_recipe()
        if (isinstance(config.n_patch, bool) or not isinstance(config.n_patch, (int, np.integer))
                or not 1 <= config.n_patch <= self.n_sources):
            raise ValueError("n_patch must be a positive integer within the mesh size")
        areas = _as_list(config.patch_area_mm2, config.n_patch)
        decays = _as_list(config.decay_mm, config.n_patch)
        if not np.isfinite(areas).all() or np.any(np.asarray(areas) <= 0):
            raise ValueError("Patch area must be finite and positive")
        if not np.isfinite(decays).all() or np.any(np.asarray(decays) < 0):
            raise ValueError("Patch decay must be finite and nonnegative")
        if not np.isfinite(config.min_seed_distance_mm) or config.min_seed_distance_mm < 0:
            raise ValueError("min_seed_distance_mm must be finite and nonnegative")
        if sum(areas) > self.vertex_area_coord2.sum() * self.coord_to_mm**2:
            raise ValueError("Requested patch area exceeds the available mesh area")
        if config.patch_vertices is not None and config.seed_vertices is None:
            raise ValueError("patch_vertices requires explicit seed_vertices")
        if config.seed_vertices is not None:
            seeds = np.asarray(config.seed_vertices)
            if (seeds.shape != (config.n_patch,) or seeds.dtype.kind not in "iu"
                    or np.any((seeds < 0) | (seeds >= self.n_sources))
                    or len(np.unique(seeds)) != config.n_patch):
                raise ValueError("seed_vertices must contain n_patch unique valid integer vertices")
            if config.patch_vertices is None:
                patches, weights = self._build_spatial_patches(seeds, areas, decays)
            else:
                if not isinstance(config.patch_vertices, list) or len(config.patch_vertices) != config.n_patch:
                    raise ValueError("patch_vertices must contain n_patch support lists")
                patches, weights = [], []
                for seed, indices, area, decay in zip(seeds, config.patch_vertices, areas, decays):
                    support = np.asarray(indices)
                    if (support.ndim != 1 or not support.size or support.dtype.kind not in "iu"
                            or np.any((support < 0) | (support >= self.n_sources))
                            or len(np.unique(support)) != support.size):
                        raise ValueError("patch_vertices supports must contain unique valid integer vertices")
                    if seed not in support or patch_hop_order(support, int(seed), self.vert_conn) is None:
                        raise ValueError("Each patch_vertices support must be connected and contain its seed")
                    if self.vertex_area_coord2[support].sum() * self.coord_to_mm**2 + 1e-9 < area:
                        raise ValueError("Stored patch_vertices area is smaller than requested patch area")
                    distances = np.linalg.norm(self.vertices[support] - self.vertices[seed], axis=1) * self.coord_to_mm
                    patches.append(support)
                    weights.append(gaussian_weights(distances, decay).astype(np.float32))
                validate_disjoint_supports(patches)
            return seeds, patches, weights

        # Largest supports first reduce greedy packing failures while preserving ROI order.
        for _ in range(8):
            seeds = np.empty(config.n_patch, dtype=int)
            patches = [None] * config.n_patch
            valid = np.ones(self.n_sources, dtype=bool)
            occupied = np.zeros(self.n_sources, dtype=bool)
            for index in np.argsort(-np.asarray(areas), kind="stable"):
                for seed in rng.permutation(np.flatnonzero(valid))[:256]:
                    patch = grow_patch_by_area(int(seed), self.vert_conn, self.vertex_area_coord2,
                                               areas[index], self.coord_to_mm)
                    actual_area = self.vertex_area_coord2[patch].sum() * self.coord_to_mm**2
                    if occupied[patch].any() or actual_area + 1e-9 < areas[index]:
                        continue
                    seeds[index], patches[index] = seed, patch
                    occupied[patch] = True
                    distance = np.linalg.norm(self.vertices-self.vertices[seed], axis=1)*self.coord_to_mm
                    valid &= (distance >= config.min_seed_distance_mm) & ~occupied
                    break
                else:
                    break
            else:
                weights = [gaussian_weights(np.linalg.norm(self.vertices[p]-self.vertices[s], axis=1)
                           *self.coord_to_mm, d).astype(np.float32)
                           for p, s, d in zip(patches, seeds, decays)]
                return seeds, patches, weights
        raise RuntimeError("Could not pack disjoint full patches within the sampling budget; reduce patch area/count or seed spacing")

    def sample_spaced_vertices(self, n_vertices: int, min_distance_mm: float, rng: np.random.Generator) -> Array:
        valid = np.ones(self.n_sources, dtype=bool)
        selected: list[int] = []
        attempts = 0
        while len(selected) < n_vertices and attempts < 10000:
            candidates = np.where(valid)[0]
            if candidates.size == 0:
                break
            seed = int(rng.choice(candidates))
            selected.append(seed)
            dists_mm = np.linalg.norm(self.vertices - self.vertices[seed], axis=1) * self.coord_to_mm
            valid[dists_mm < min_distance_mm] = False
            valid[seed] = False
            attempts += 1
        if len(selected) < n_vertices:
            raise RuntimeError("Could not sample enough spatially separated vertices")
        return np.asarray(selected, dtype=int)

    def _build_spatial_patches(
        self,
        seeds: Array,
        patch_areas_mm2: list[float],
        decays_mm: list[float],
    ) -> tuple[list[Array], list[Array]]:
        patches = []
        weights = []
        for seed, area_mm2, decay_mm in zip(seeds, patch_areas_mm2, decays_mm):
            patch = grow_patch_by_area(int(seed), self.vert_conn, self.vertex_area_coord2, area_mm2, self.coord_to_mm)
            if self.vertex_area_coord2[patch].sum()*self.coord_to_mm**2 + 1e-9 < area_mm2:
                raise ValueError(f"Requested patch area cannot fit seed {seed}'s connected mesh component")
            dists_mm = np.linalg.norm(self.vertices[patch] - self.vertices[int(seed)], axis=1) * self.coord_to_mm
            weight = gaussian_weights(dists_mm, decay_mm)
            patches.append(patch)
            weights.append(weight.astype(np.float32))
        validate_disjoint_supports(patches)
        return patches, weights

    def _build_roi_signals(
        self,
        config: G1SimulationConfig,
        ts: TimeSeriesGenerator,
        rng: np.random.Generator,
        n_roi: int,
    ) -> Array:
        if not config.waveform_params:
            raise ValueError("waveform_params must contain at least one waveform")
        mode = config.connectivity_mode.lower()
        base = []
        for idx in range(1 if mode == "synchronous" else n_roi):
            waveform_ts = ts
            if config.component_seeds is not None:
                waveform_rng = np.random.default_rng(np.random.SeedSequence([config.component_seeds["waveform"], idx]))
                waveform_ts = TimeSeriesGenerator(config.srate, config.time_window, waveform_rng)
            kind, params = config.waveform_params[idx % len(config.waveform_params)]
            base.append(waveform_ts.waveform(kind, params))
        base = np.asarray(base, dtype=float)
        if mode == "synchronous":
            return np.repeat(base, n_roi, axis=0)
        if mode == "independent" or n_roi == 1:
            return base
        if mode == "correlated":
            corr = self._correlation_matrix(config.coupling, n_roi)
            whitened = np.asarray([self._standardize(row) for row in base])
            return np.linalg.cholesky(corr) @ whitened
        if mode == "causal":
            coupling = self._coupling_matrix(config.coupling, n_roi)
            radius = max(float(np.max(np.abs(np.linalg.eigvals(coupling)))), 1e-12)
            if radius >= 0.95:
                coupling = coupling * (0.95 / radius)
            signals = np.zeros_like(base)
            signals[:, 0] = base[:, 0]
            for time_idx in range(1, ts.n_time):
                signals[:, time_idx] = coupling @ signals[:, time_idx - 1] + base[:, time_idx]
            return signals
        raise ValueError(f"Unsupported connectivity_mode: {config.connectivity_mode}")

    @staticmethod
    def _correlation_matrix(coupling: float | list[float], n_roi: int) -> Array:
        if np.isscalar(coupling):
            rho = float(coupling)
            rho = float(np.clip(rho, -1.0 / max(n_roi - 1, 1) + 1e-6, 0.99))
            corr = np.eye(n_roi) * (1.0 - rho) + np.ones((n_roi, n_roi)) * rho
        else:
            corr = np.asarray(coupling, dtype=float).reshape(n_roi, n_roi)
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)
        eig_min = float(np.min(np.linalg.eigvalsh(corr)))
        if eig_min <= 1e-6:
            corr += np.eye(n_roi) * (1e-6 - eig_min)
        return corr

    @staticmethod
    def _coupling_matrix(coupling: float | list[float], n_roi: int) -> Array:
        if np.isscalar(coupling):
            mat = np.eye(n_roi) * 0.5
            off_diag = np.ones((n_roi, n_roi)) - np.eye(n_roi)
            mat += off_diag * float(coupling)
            return mat
        arr = np.asarray(coupling, dtype=float).reshape(-1)
        if arr.size != n_roi * n_roi:
            raise ValueError("causal coupling must be scalar or n_patch*n_patch values")
        return arr.reshape(n_roi, n_roi)

    def _make_source_background(
        self,
        source: Array,
        ts: TimeSeriesGenerator,
        rng: np.random.Generator,
        background_type: str,
        source_snr_db: float,
        rank: int = 50,
        baseline_samples: int = 0,
    ) -> Array:
        if baseline_samples:
            ts = TimeSeriesGenerator(ts.srate, (0, (ts.n_time + baseline_samples) / ts.srate), rng)
        rank = min(rank, self.n_sources)
        if background_type == "pink":
            bases = np.asarray([ts.aperiodic_noise(exponent=1.0, white_floor=0.08, envelope_depth=0.12) for _ in range(rank)])
        elif background_type in {"powerlaw", "neurodsp_powerlaw"}:
            bases = np.asarray([ts.powerlaw_noise(exponent=float(rng.uniform(0.8, 1.6)), white_floor=0.03) for _ in range(rank)])
        elif background_type in {"knee_powerlaw", "specparam"}:
            bases = np.asarray(
                [
                    ts.specparam_signal(
                        exponent=float(rng.uniform(0.8, 1.8)),
                        knee_hz=float(rng.uniform(0.4, 2.0)),
                        peaks=[{"center_hz": float(rng.uniform(8.0, 12.0)), "bandwidth_hz": float(rng.uniform(1.0, 3.0)), "height": float(rng.uniform(0.2, 0.7))}],
                        white_floor=0.02,
                    )
                    for _ in range(rank)
                ]
            )
        elif background_type in {"synaptic", "shot_noise"}:
            bases = np.asarray([ts.synaptic_shot_noise(event_rate_hz=float(rng.uniform(15.0, 60.0))) for _ in range(rank)])
        elif background_type == "eegsourcesim_resting":
            bases = np.asarray([ts.eegsourcesim_resting_noise() for _ in range(rank)])
        elif background_type in {"alpha", "beta", "theta", "delta", "gamma"}:
            bases = np.asarray([ts.waveform("resting", {"band": background_type}) for _ in range(rank)])
        elif background_type == "white":
            bases = np.asarray([ts.broadband_noise() for _ in range(rank)])
        else:
            raise ValueError(f"Unsupported source_background_type: {background_type}")
        mix = rng.standard_normal((self.n_sources, rank))
        background = mix @ bases
        event = background[:, baseline_samples:]
        event -= event.mean(axis=1, keepdims=True)
        if baseline_samples:
            baseline = background[:, :baseline_samples]
            baseline -= baseline.mean(axis=1, keepdims=True)
        signal_power = float(np.mean(source**2))
        background_power = float(np.mean(event**2))
        target_power = signal_power / max(10.0 ** (float(source_snr_db) / 10.0), 1e-12)
        return background * np.sqrt(target_power / max(background_power, 1e-12))

    def _make_sensor_noise(
        self,
        clean: Array,
        ts: TimeSeriesGenerator,
        noise_type: str,
        snr_db: float,
        snr_mode: str = "robust_global_rms",
        clip_quantile: float = 0.95,
        real_sensor_noise: Array | None = None,
        average_reference: bool = False,
    ) -> Array:
        if real_sensor_noise is not None:
            noise = self._prepare_real_sensor_noise(real_sensor_noise, clean.shape, ts.rng)
        elif noise_type == "real":
            raise ValueError("sensor_noise_type='real' requires real_sensor_noise")
        elif noise_type == "iid_white":
            noise = ts.rng.standard_normal(clean.shape)
        elif noise_type == "white":
            noise = self._make_broadband_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "pink":
            noise = self._make_aperiodic_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type in {"powerlaw", "neurodsp_powerlaw"}:
            noise = self._make_powerlaw_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type in {"specparam_resting", "knee_powerlaw"}:
            noise = self._make_specparam_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "mixed":
            noise = 0.55 * self._make_aperiodic_sensor_noise(clean.shape[0], clean.shape[1], ts)
            noise += 0.35 * self._make_broadband_sensor_noise(clean.shape[0], clean.shape[1], ts)
            noise += 0.10 * self._make_eegsourcesim_sensor_noise(clean.shape[0], clean.shape[1], ts)
        elif noise_type == "eegsourcesim_resting":
            noise = self._make_eegsourcesim_sensor_noise(clean.shape[0], clean.shape[1], ts)
        else:
            raise ValueError(f"Unsupported sensor_noise_type: {noise_type}")
        noise -= noise.mean(axis=1, keepdims=True)
        if average_reference:
            noise -= noise.mean(axis=0, keepdims=True)
        clean_power = self._snr_reference_power(clean, snr_mode, clip_quantile)
        noise_power = float(np.mean(noise**2))
        target_power = clean_power / max(10.0 ** (float(snr_db) / 10.0), 1e-12)
        if noise_power == 0 and target_power > 0:
            raise ValueError("Sensor noise has zero power after referencing; cannot reach the requested SNR")
        return noise * np.sqrt(target_power / max(noise_power, np.finfo(float).tiny))

    @staticmethod
    def _snr_reference_power(clean: Array, mode: str, clip_quantile: float) -> float:
        if mode == "global_rms":
            return float(np.mean(clean**2))
        if mode == "robust_global_rms":
            limit = float(np.quantile(np.abs(clean), np.clip(clip_quantile, 0.5, 0.999)))
            clipped = np.clip(clean, -limit, limit)
            return float(np.mean(clipped**2))
        if mode == "active_rms":
            global_field_power = np.sqrt(np.mean(clean**2, axis=0))
            threshold = float(np.quantile(global_field_power, np.clip(clip_quantile, 0.5, 0.999)))
            active = global_field_power >= threshold
            return float(np.mean(clean[:, active] ** 2)) if np.any(active) else float(np.mean(clean**2))
        raise ValueError(f"Unsupported sensor_snr_mode: {mode}")

    @staticmethod
    def _prepare_real_sensor_noise(real_noise: Array, target_shape: tuple[int, int], rng: np.random.Generator) -> Array:
        noise = np.array(real_noise, dtype=float, copy=True)
        if noise.ndim == 1:
            noise = np.tile(noise[None, :], (target_shape[0], 1))
        if noise.ndim != 2:
            raise ValueError("real_sensor_noise must be [channels, times] or [times]")
        if noise.shape[0] != target_shape[0]:
            raise ValueError(f"real noise has {noise.shape[0]} channels, expected {target_shape[0]}")
        if noise.shape[1] < target_shape[1]:
            repeats = int(np.ceil(target_shape[1] / noise.shape[1]))
            noise = np.tile(noise, (1, repeats))
        if noise.shape[1] > target_shape[1]:
            start = int(rng.integers(0, noise.shape[1] - target_shape[1] + 1))
            noise = noise[:, start : start + target_shape[1]]
        else:
            noise = noise[:, : target_shape[1]]
        noise -= np.mean(noise, axis=1, keepdims=True)
        return noise

    def _make_broadband_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray([ts.broadband_noise(n_times) for _ in range(min(4, n_channels))])
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray([ts.broadband_noise(n_times) for _ in range(n_channels)])
        noise = 0.30 * shared + 0.70 * independent
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_powerlaw_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.powerlaw_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.8)),
                    knee_hz=None,
                    white_floor=float(ts.rng.uniform(0.02, 0.12)),
                )
                for _ in range(min(6, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.powerlaw_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.6, 1.6)),
                    white_floor=float(ts.rng.uniform(0.04, 0.16)),
                )
                for _ in range(n_channels)
            ]
        )
        noise = 0.50 * shared + 0.45 * independent + 0.05 * ts.rng.standard_normal((n_channels, n_times))
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_specparam_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.specparam_signal(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.8)),
                    knee_hz=float(ts.rng.uniform(0.3, 2.0)),
                    peaks=[
                        {
                            "center_hz": float(ts.rng.uniform(8.0, 12.0)),
                            "bandwidth_hz": float(ts.rng.uniform(1.0, 3.5)),
                            "height": float(ts.rng.uniform(0.25, 0.9)),
                        }
                    ],
                    white_floor=float(ts.rng.uniform(0.01, 0.08)),
                )
                for _ in range(min(6, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.specparam_signal(
                    n_times,
                    exponent=float(ts.rng.uniform(0.7, 1.5)),
                    peaks=[
                        {
                            "center_hz": float(ts.rng.uniform(8.0, 12.0)),
                            "bandwidth_hz": float(ts.rng.uniform(1.5, 4.0)),
                            "height": float(ts.rng.uniform(0.1, 0.6)),
                        }
                    ],
                    white_floor=float(ts.rng.uniform(0.03, 0.12)),
                )
                for _ in range(n_channels)
            ]
        )
        noise = 0.45 * shared + 0.45 * independent + 0.10 * ts.rng.standard_normal((n_channels, n_times))
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_aperiodic_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        common = np.asarray(
            [
                ts.aperiodic_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.4)),
                    white_floor=float(ts.rng.uniform(0.06, 0.16)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.20)),
                )
                for _ in range(min(5, n_channels))
            ]
        )
        mixing = ts.rng.standard_normal((n_channels, common.shape[0]))
        shared = mixing @ common
        independent = np.asarray(
            [
                ts.aperiodic_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.7, 1.3)),
                    white_floor=float(ts.rng.uniform(0.08, 0.20)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.18)),
                )
                for _ in range(n_channels)
            ]
        )
        white_floor = ts.rng.standard_normal((n_channels, n_times)) * 0.08
        noise = 0.45 * shared + 0.50 * independent + white_floor
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _make_eegsourcesim_sensor_noise(self, n_channels: int, n_times: int, ts: TimeSeriesGenerator) -> Array:
        n_common = min(6, n_channels)
        common_basis = np.asarray(
            [
                ts.eegsourcesim_resting_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.9, 1.4)),
                    alpha_peak_hz=float(ts.rng.uniform(8.0, 12.0)),
                    alpha_width_hz=float(ts.rng.uniform(2.0, 4.0)),
                    alpha_gain=float(ts.rng.uniform(0.8, 2.5)),
                    white_floor=float(ts.rng.uniform(0.03, 0.08)),
                    envelope_depth=float(ts.rng.uniform(0.10, 0.25)),
                )
                for _ in range(n_common)
            ]
        )
        channel_mixing = ts.rng.standard_normal((n_channels, n_common))
        common_noise = channel_mixing @ common_basis
        independent_noise = np.asarray(
            [
                ts.eegsourcesim_resting_noise(
                    n_times,
                    exponent=float(ts.rng.uniform(0.8, 1.3)),
                    alpha_peak_hz=float(ts.rng.uniform(8.0, 12.0)),
                    alpha_width_hz=float(ts.rng.uniform(2.0, 5.0)),
                    alpha_gain=float(ts.rng.uniform(0.3, 1.5)),
                    white_floor=float(ts.rng.uniform(0.05, 0.12)),
                    envelope_depth=float(ts.rng.uniform(0.05, 0.20)),
                )
                for _ in range(n_channels)
            ]
        )
        white_floor = ts.rng.standard_normal((n_channels, n_times)) * 0.15
        noise = 0.55 * common_noise + 0.35 * independent_noise + white_floor
        noise -= noise.mean(axis=1, keepdims=True)
        return noise

    def _source_metadata(
        self,
        config: G1SimulationConfig,
        seeds: Array,
        patch_indices: list[Array],
        patch_weights: list[Array],
        patch_gains: Array,
        source_background: Array | None,
    ) -> dict[str, Any]:
        manifest = asdict(config)
        manifest.update(
            {
                "generator_type": "G1_math",
                "storage_policy": "metadata_first",
                "n_sources": self.n_sources,
                "n_sensors": self.n_sensors,
                "coordinate_unit_scale_to_mm": self.coord_to_mm,
                "seed_vertices": seeds.tolist(),
                "active_vertices": sorted({int(v) for patch in patch_indices for v in patch.tolist()}),
                "patches": [
                    {
                        "patch_index": idx,
                        "seed_vertex": int(seeds[idx]),
                        "vertices": patch_indices[idx].tolist(),
                        "spatial_weights": patch_weights[idx].tolist(),
                        "gain": float(patch_gains[idx]),
                    }
                    for idx in range(len(seeds))
                ],
                "source_background_saved": source_background is not None,
            }
        )
        if source_background is not None:
            manifest["source_background_power"] = float(np.mean(source_background**2))
        return _json_safe(manifest)

    @staticmethod
    def _standardize(signal: Array) -> Array:
        signal = signal - float(np.mean(signal))
        return signal / max(float(np.std(signal)), 1e-12)


class G1ScenarioSampler:
    def __init__(
        self,
        snr_db_range: tuple[float, float] = (0.0, 20.0),
        n_patch_range: tuple[int, int] = (1, 6),
        patch_area_mm2_range: tuple[float, float] = (25.0, 400.0),
        random_seed: int | None = None,
    ):
        self.rng = np.random.default_rng(random_seed)
        self.snr_db_range = snr_db_range
        self.n_patch_range = n_patch_range
        self.patch_area_mm2_range = patch_area_mm2_range
        self.waveforms = [
            "erp",
            "ersp",
            "spike",
            "resting",
            "resting_pinknoise",
            "eegsourcesim_resting_noise",
            "ar",
            "powerlaw",
            "knee_powerlaw",
            "specparam",
            "synaptic_noise",
            "oscillation",
            "burst_oscillation",
            "nonsinusoidal_oscillation",
        ]

    def sample(self, sample_id: str, template_id: str, source_space_id: str, leadfield_id: str) -> G1SimulationConfig:
        n_patch = int(self.rng.integers(self.n_patch_range[0], self.n_patch_range[1] + 1))
        waveform_params = [self._sample_waveform() for _ in range(n_patch)]
        mode = str(self.rng.choice(["independent", "correlated", "causal"], p=[0.5, 0.25, 0.25]))
        if mode == "correlated":
            coupling: float | list[float] = float(self.rng.uniform(0.3, 0.9))
        elif mode == "causal":
            coupling = self.rng.uniform(-0.2, 0.2, n_patch * n_patch).round(3).tolist()
        else:
            coupling = 0.0
        background_choice = self.rng.choice(
            ["pink", "eegsourcesim_resting", "alpha", "powerlaw", "specparam", "synaptic", "white", "none"],
            p=[0.18, 0.22, 0.15, 0.14, 0.12, 0.08, 0.05, 0.06],
        )
        background_type = None if background_choice == "none" else str(background_choice)
        return G1SimulationConfig(
            sample_id=sample_id,
            template_id=template_id,
            source_space_id=source_space_id,
            leadfield_id=leadfield_id,
            n_patch=n_patch,
            patch_area_mm2=self.rng.uniform(*self.patch_area_mm2_range, n_patch).round(3).tolist(),
            decay_mm=self.rng.uniform(5.0, 25.0, n_patch).round(3).tolist(),
            connectivity_mode=mode,
            coupling=coupling,
            waveform_params=waveform_params,
            source_background_type=background_type,
            source_snr_db=float(self.rng.uniform(20.0, 35.0)),
            sensor_noise_type=str(
                self.rng.choice(
                    ["white", "pink", "powerlaw", "specparam_resting", "mixed", "eegsourcesim_resting"],
                    p=[0.16, 0.18, 0.18, 0.18, 0.12, 0.18],
                )
            ),
            sensor_snr_db=float(self.rng.uniform(*self.snr_db_range)),
            random_seed=int(self.rng.integers(1, 2**31 - 1)),
        )

    def _sample_waveform(self, kind: str | None = None) -> tuple[str, dict[str, Any]]:
        if kind is not None and kind not in self.waveforms:
            raise ValueError(f"Unsupported sampled waveform: {kind}")
        kind = kind if kind is not None else str(
            self.rng.choice(
                self.waveforms,
                p=[0.12, 0.12, 0.07, 0.10, 0.06, 0.08, 0.06, 0.08, 0.07, 0.08, 0.06, 0.04, 0.03, 0.03],
            )
        )
        if kind == "erp":
            return (
                "erp",
                {
                    "components": [
                        {
                            "type": str(self.rng.choice(["gaussian", "gabor"])),
                            "latency_s": float(self.rng.uniform(0.05, 0.35)),
                            "width_s": float(self.rng.uniform(0.03, 0.12)),
                            "std_dev_s": float(self.rng.uniform(0.03, 0.12)),
                            "frequency_Hz": float(self.rng.uniform(4.0, 20.0)),
                            "amplitude_rel": float(self.rng.uniform(0.5, 1.5)),
                        }
                    ],
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "ersp":
            return (
                "ersp",
                {
                    "base_frequency_band_Hz": float(self.rng.choice([6, 10, 12, 20, 35])),
                    "modulation": {
                        "type": str(self.rng.choice(["burst", "invburst", "ampmod"], p=[0.7, 0.15, 0.15])),
                        "latency_s": float(self.rng.uniform(0.05, 0.35)),
                        "width_s": float(self.rng.uniform(0.1, 0.4)),
                        "min_rel_amplitude": float(self.rng.uniform(0.05, 0.4)),
                    },
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "spike":
            return (
                "spike",
                {
                    "latency_s": float(self.rng.uniform(0.05, 0.35)),
                    "duration_s": float(self.rng.uniform(0.04, 0.12)),
                    "shape": str(self.rng.choice(["exponential_decay", "triangular"])),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "resting":
            return (
                "resting",
                {
                    "band": str(self.rng.choice(["delta", "theta", "alpha", "beta", "gamma"])),
                    "envelope": "smooth",
                    "min_gain": float(self.rng.uniform(0.2, 0.5)),
                    "max_gain": float(self.rng.uniform(0.9, 1.4)),
                    "control_interval_s": float(self.rng.uniform(0.25, 0.7)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "ar":
            return ("ar", {"freq": float(self.rng.choice([5, 8, 10, 15, 20])), "amplitude": float(self.rng.uniform(0.5, 2.0))})
        if kind == "powerlaw":
            return (
                "powerlaw",
                {
                    "exponent": float(self.rng.uniform(0.8, 2.0)),
                    "white_floor": float(self.rng.uniform(0.01, 0.10)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "knee_powerlaw":
            return (
                "knee_powerlaw",
                {
                    "exponent": float(self.rng.uniform(0.8, 2.0)),
                    "knee_hz": float(self.rng.uniform(0.3, 3.0)),
                    "white_floor": float(self.rng.uniform(0.01, 0.08)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "specparam":
            return (
                "specparam",
                {
                    "exponent": float(self.rng.uniform(0.8, 1.8)),
                    "knee_hz": float(self.rng.uniform(0.3, 2.5)),
                    "peaks": [
                        {
                            "center_hz": float(self.rng.uniform(8.0, 12.0)),
                            "bandwidth_hz": float(self.rng.uniform(1.0, 3.0)),
                            "height": float(self.rng.uniform(0.3, 1.0)),
                        }
                    ],
                    "envelope_depth": float(self.rng.uniform(0.05, 0.25)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "synaptic_noise":
            return (
                "synaptic_noise",
                {
                    "event_rate_hz": float(self.rng.uniform(15.0, 80.0)),
                    "tau_rise_s": float(self.rng.uniform(0.003, 0.012)),
                    "tau_decay_s": float(self.rng.uniform(0.030, 0.120)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "oscillation":
            return (
                "oscillation",
                {
                    "frequency_hz": float(self.rng.choice([6, 8, 10, 12, 20])),
                    "frequency_jitter_hz": float(self.rng.uniform(0.0, 0.8)),
                    "min_gain": float(self.rng.uniform(0.25, 0.6)),
                    "max_gain": float(self.rng.uniform(0.9, 1.5)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "burst_oscillation":
            return (
                "burst_oscillation",
                {
                    "frequency_hz": float(self.rng.choice([8, 10, 12, 20])),
                    "duty_cycle": float(self.rng.uniform(0.25, 0.7)),
                    "min_duration_s": float(self.rng.uniform(0.06, 0.12)),
                    "max_duration_s": float(self.rng.uniform(0.18, 0.45)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "nonsinusoidal_oscillation":
            return (
                "nonsinusoidal_oscillation",
                {
                    "frequency_hz": float(self.rng.choice([6, 8, 10, 12])),
                    "harmonics": {2: float(self.rng.uniform(0.2, 0.5)), 3: float(self.rng.uniform(0.05, 0.25))},
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        if kind == "eegsourcesim_resting_noise":
            return (
                "eegsourcesim_resting_noise",
                {
                    "exponent": float(self.rng.uniform(0.8, 1.4)),
                    "alpha_peak_hz": float(self.rng.uniform(8.0, 12.0)),
                    "alpha_width_hz": float(self.rng.uniform(2.0, 4.0)),
                    "alpha_gain": float(self.rng.uniform(0.8, 3.0)),
                    "white_floor": float(self.rng.uniform(0.03, 0.10)),
                    "envelope_depth": float(self.rng.uniform(0.10, 0.30)),
                    "amplitude": float(self.rng.uniform(0.5, 2.0)),
                },
            )
        return ("resting_pinknoise", {"amplitude": float(self.rng.uniform(0.5, 2.0))})


class G1DifficultyScenarioSampler(G1ScenarioSampler):
    DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
        "easy": {
            "snr_db_range": (15.0, 25.0),
            "n_patch_range": (1, 1),
            "patch_area_mm2_range": (200.0, 500.0),
            "min_seed_distance_mm": 30.0,
            "connectivity_modes": ["independent"],
            "sensor_noise_types": ["white", "pink", "powerlaw"],
            "source_background_types": [None, "pink", "powerlaw"],
        },
        "medium": {
            "snr_db_range": (8.0, 18.0),
            "n_patch_range": (1, 3),
            "patch_area_mm2_range": (100.0, 350.0),
            "min_seed_distance_mm": 25.0,
            "connectivity_modes": ["independent", "correlated"],
            "sensor_noise_types": ["white", "pink", "powerlaw", "specparam_resting", "eegsourcesim_resting"],
            "source_background_types": [None, "pink", "powerlaw", "specparam", "alpha", "eegsourcesim_resting"],
        },
        "hard": {
            "snr_db_range": (0.0, 12.0),
            "n_patch_range": (2, 6),
            "patch_area_mm2_range": (40.0, 250.0),
            "min_seed_distance_mm": 20.0,
            "connectivity_modes": ["independent", "correlated", "causal"],
            "sensor_noise_types": ["pink", "powerlaw", "specparam_resting", "mixed", "eegsourcesim_resting"],
            "source_background_types": ["pink", "powerlaw", "specparam", "synaptic", "alpha", "eegsourcesim_resting", "white"],
        },
        "extreme": {
            "snr_db_range": (-5.0, 8.0),
            "n_patch_range": (3, 10),
            "patch_area_mm2_range": (20.0, 180.0),
            "min_seed_distance_mm": 15.0,
            "connectivity_modes": ["correlated", "causal"],
            "sensor_noise_types": ["mixed", "powerlaw", "specparam_resting", "eegsourcesim_resting"],
            "source_background_types": ["pink", "powerlaw", "specparam", "synaptic", "eegsourcesim_resting", "alpha"],
        },
    }

    def __init__(self, difficulty: str | int = "medium", random_seed: int | None = None):
        self.difficulty = self.normalize_difficulty(difficulty)
        preset = self.DIFFICULTY_PRESETS[self.difficulty]
        super().__init__(
            snr_db_range=preset["snr_db_range"],
            n_patch_range=preset["n_patch_range"],
            patch_area_mm2_range=preset["patch_area_mm2_range"],
            random_seed=random_seed,
        )

    @classmethod
    def normalize_difficulty(cls, difficulty: str | int) -> str:
        if isinstance(difficulty, int):
            levels = ["easy", "medium", "hard", "extreme"]
            return levels[max(0, min(difficulty, len(levels) - 1))]
        difficulty = difficulty.lower()
        aliases = {"0": "easy", "1": "medium", "2": "hard", "3": "extreme"}
        difficulty = aliases.get(difficulty, difficulty)
        if difficulty not in cls.DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        return difficulty

    def sample(self, sample_id: str, template_id: str, source_space_id: str, leadfield_id: str) -> G1SimulationConfig:
        config = super().sample(sample_id, template_id, source_space_id, leadfield_id)
        preset = self.DIFFICULTY_PRESETS[self.difficulty]
        mode = str(self.rng.choice(preset["connectivity_modes"]))
        config.connectivity_mode = mode
        config.min_seed_distance_mm = float(preset["min_seed_distance_mm"])
        config.sensor_noise_type = str(self.rng.choice(preset["sensor_noise_types"]))
        background = self.rng.choice(["none" if v is None else v for v in preset["source_background_types"]])
        config.source_background_type = None if background == "none" else str(background)
        if mode == "independent":
            config.coupling = 0.0
        elif mode == "correlated":
            config.coupling = float(self.rng.uniform(0.4, 0.9))
        else:
            config.coupling = self.rng.uniform(-0.2, 0.2, config.n_patch * config.n_patch).round(3).tolist()
        return config
