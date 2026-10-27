from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample

if __package__ in {None, ""}:
    from atlas_profiles import ATLAS_CHOICES, get_atlas_profile, validate_atlas_connectivity, load_connectivity_metadata
else:
    from .atlas_profiles import ATLAS_CHOICES, get_atlas_profile, validate_atlas_connectivity, load_connectivity_metadata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TVB_USER_HOME = ROOT / "Simulation_G3" / ".tvb_home"


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
    if isinstance(value, Path):
        return str(value)
    return value


def find_default_connectivity() -> Path:
    candidates = [
        ROOT / "Simulation_G3" / "anatomy" / "connectivity_998.zip",
        ROOT / "Anatomy" / "connectivity_998.zip",
        ROOT / "anatomy" / "connectivity_998.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find connectivity_998.zip. Put it under Simulation_G3/anatomy/ or Anatomy/, "
        "or pass --connectivity-path."
    )


@dataclass
class TVBSignalConfig:
    """Times are milliseconds; active IDs refer to the exported output atlas.

    Warm-up is extra simulated time and is never exported. simulation_length_ms
    is the post-warm-up window; optional discard_ms trims that window further.
    """
    sample_id: str = "G3_tvb_sample"
    signal_type: str = "spike_jr"
    connectivity_path: str | None = None
    active_regions: list[int] = field(default_factory=lambda: [10])
    simulation_length_ms: float = 1000.0
    warmup_ms: float = 1000.0
    dt_ms: float = 0.5
    output_srate_hz: float = 500.0
    random_seed: int = 0
    reshape_to_994: bool = True
    normalize: bool = True
    discard_ms: float = 0.0
    jr_background_A: float = 3.25
    jr_active_A: float | None = None
    jr_mu: float | None = None
    jr_noise_scale: float = 1.0
    coupling_strength: float = 1.0
    jr_coupling_mode: str = "sparse"


class TVBSourceGenerator:
    """Generate region-level neural mass source activity with TVB.

    The output is region x time and is intentionally kept source-level so it can
    be reused by the existing G2 projection/leadfield pipeline.
    """

    SUPPORTED_SIGNAL_TYPES = {
        "resting_jr",
        "spike_jr",
        "alpha_jr",
        "oscillator",
        "wilson_cowan",
        "seizure_epileptor",
    }

    def __init__(self, connectivity_path: str | Path | None = None):
        self.connectivity_path = Path(connectivity_path) if connectivity_path else find_default_connectivity()

    def generate(self, config: TVBSignalConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._validate_config(config)
        if config.signal_type not in self.SUPPORTED_SIGNAL_TYPES:
            raise ValueError(f"Unsupported signal_type {config.signal_type!r}; choose from {sorted(self.SUPPORTED_SIGNAL_TYPES)}")
        if config.connectivity_path is not None:
            self.connectivity_path = Path(config.connectivity_path)
        if not self.connectivity_path.exists():
            raise FileNotFoundError(self.connectivity_path)
        connectivity_metadata = load_connectivity_metadata(self.connectivity_path)

        os.environ.setdefault("TVB_USER_HOME", str(DEFAULT_TVB_USER_HOME))
        DEFAULT_TVB_USER_HOME.mkdir(parents=True, exist_ok=True)

        try:
            from tvb.simulator.lab import connectivity, coupling, integrators, models, monitors, noise, simulator
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TVB is required for G3 TVB simulation. Install it with: "
                ".\\.venv\\Scripts\\python.exe -m pip install tvb-library"
            ) from exc

        conn = connectivity.Connectivity.from_file(source_file=str(self.connectivity_path))
        conn.configure()
        n_regions = int(conn.number_of_regions)
        output_indices = self._output_region_indices(n_regions, config.reshape_to_994)
        active_regions = self._validate_regions(config.active_regions, len(output_indices))
        active_regions_raw = output_indices[active_regions].tolist()

        model, model_output, model_meta = self._build_model(config, n_regions, active_regions_raw, models)
        nsig = self._noise_sigma(config, model_meta)
        random_stream = np.random.mtrand.RandomState(int(config.random_seed))
        noise_class = noise.Additive(random_stream=random_stream, nsig=nsig)

        sim = simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=self._build_coupling(config, coupling),
            integrator=integrators.HeunStochastic(dt=float(config.dt_ms), noise=noise_class),
            monitors=(monitors.Raw(),),
        ).configure()

        total_length_ms = float(config.warmup_ms) + float(config.simulation_length_ms)
        overflow_policy = "raise" if config.signal_type.endswith("_jr") else np.geterr()["over"]
        try:
            with np.errstate(over=overflow_policy, invalid="raise"):
                (time_ms, raw), = sim.run(simulation_length=total_length_ms)
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"TVB {config.signal_type} integration failed at dt={config.dt_ms} ms: {exc}. "
                "Check model parameters and integration step."
            ) from exc
        region_source = self._extract_source(raw, model_output)
        # Raw monitor samples are integration endpoints: retain (start, stop].
        # The exported sample grid starts at zero and spans [0, retained duration).
        time_ms, region_source = self._discard_initial(time_ms, region_source, config.warmup_ms + config.discard_ms)
        retained_duration_ms = config.simulation_length_ms - config.discard_ms
        expected_samples = int(round(retained_duration_ms / config.dt_ms))
        if region_source.shape != (n_regions, expected_samples):
            raise ValueError(f"Expected {(n_regions, expected_samples)} retained TVB samples, got {region_source.shape}")
        region_source = self._reshape_regions(region_source, config.reshape_to_994)
        time_ms, region_source = self._resample(time_ms, region_source, config.output_srate_hz, retained_duration_ms)

        if not np.isfinite(region_source).all():
            bad_count = int(region_source.size - np.count_nonzero(np.isfinite(region_source)))
            raise FloatingPointError(
                f"TVB simulation produced {bad_count} non-finite values for {config.signal_type}. "
                "Try reducing coupling strength, active excitability, or simulation length."
            )
        with np.errstate(over="raise", invalid="raise"):
            if config.normalize:
                region_source = region_source - region_source.mean(axis=1, keepdims=True)
                region_source = region_source / max(float(np.max(np.abs(region_source))), 1e-12)
            region_source = region_source.astype(np.float32)

        manifest = {
            "generator_type": "G3_TVB_NMM",
            "sample_id": config.sample_id,
            "signal_type": config.signal_type,
            "connectivity_path": str(self.connectivity_path),
            "connectivity_metadata": connectivity_metadata,
            "n_regions_raw": n_regions,
            "n_regions_output": int(region_source.shape[0]),
            "active_regions": active_regions,
            "active_regions_raw": active_regions_raw,
            "active_region_index_space": "output_atlas",
            "region_names": np.asarray(getattr(conn, "region_labels", np.arange(n_regions))).astype(str)[output_indices].tolist(),
            "model": model_meta,
            "coupling": {
                "class": "SigmoidalJansenRit" if config.signal_type.endswith("_jr") else "Linear",
                "strength": float(config.coupling_strength),
                "implementation": config.jr_coupling_mode if config.signal_type.endswith("_jr") else "dense",
            },
            "noise_nsig": self._noise_sigma(config, model_meta),
            "integrator": {"class": "HeunStochastic", "dt_ms": float(config.dt_ms)},
            "simulation_length_ms": float(config.simulation_length_ms),
            "simulated_duration_ms": total_length_ms,
            "warmup_ms": float(config.warmup_ms),
            "discard_ms": float(config.discard_ms),
            "discarded_initial_ms": float(config.warmup_ms + config.discard_ms),
            "retained_duration_ms": float(retained_duration_ms),
            "first_retained_raw_time_ms": float(config.warmup_ms + config.discard_ms + config.dt_ms),
            "output_srate_hz": float(config.output_srate_hz),
            "random_seed": int(config.random_seed),
            "region_source_shape": list(region_source.shape),
            "time_shape": list(time_ms.shape),
            "normalize": bool(config.normalize),
        }
        return region_source, time_ms, _json_safe(manifest)

    def save_sample(self, output_dir: str | Path, config: TVBSignalConfig) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        region_source, time_ms, manifest = self.generate(config)
        npz_path = output_dir / f"{config.sample_id}_region_source.npz"
        manifest_path = output_dir / f"{config.sample_id}_manifest.json"
        np.savez_compressed(npz_path, region_source=region_source, time_ms=time_ms)
        manifest["region_source"] = npz_path.name
        manifest["manifest"] = manifest_path.name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _build_model(self, config: TVBSignalConfig, n_regions: int, active_regions: list[int], models: Any) -> tuple[Any, str, dict[str, Any]]:
        if config.signal_type in {"resting_jr", "spike_jr", "alpha_jr"}:
            if __package__ in {None, ""}:
                from tvb_jr import StableJansenRit
            else:
                from .tvb_jr import StableJansenRit
            default_active_a = {
                "resting_jr": config.jr_background_A,
                "alpha_jr": 3.35,
                "spike_jr": 3.6,
            }[config.signal_type]
            active_a = default_active_a if config.jr_active_A is None else config.jr_active_A
            a_values = np.ones(n_regions, dtype=float) * float(config.jr_background_A)
            for region in active_regions:
                a_values[region] = float(active_a)
            default_mu = 0.10 if config.signal_type == "alpha_jr" else 0.087
            mu = default_mu if config.jr_mu is None else float(config.jr_mu)
            model = StableJansenRit(
                A=a_values,
                mu=np.array([mu]),
                v0=np.array([6.0]),
                p_max=np.array([0.15]),
                p_min=np.array([0.03]),
            )
            return model, "jr_pyramidal_difference", {
                "class": "JansenRit",
                "output": "y1_minus_y2",
                "background_A": float(config.jr_background_A),
                "active_A": float(active_a),
                "mu": float(mu),
                "noise_scale": float(config.jr_noise_scale),
                "sigmoid_evaluation": "scipy.special.expit",
            }

        if config.signal_type == "oscillator":
            model = models.Generic2dOscillator()
            return model, "state0", {
                "class": "Generic2dOscillator",
                "output": "state0",
            }

        if config.signal_type == "wilson_cowan":
            model = models.WilsonCowan()
            return model, "state0", {"class": "WilsonCowan", "output": "excitatory_state"}

        if config.signal_type == "seizure_epileptor":
            model = models.Epileptor()
            if hasattr(model, "x0"):
                x0 = np.asarray(model.x0).copy()
                if x0.size == 1:
                    x0 = np.ones(n_regions, dtype=float) * float(x0.item())
                for region in active_regions:
                    x0[region] = -1.6
                model.x0 = x0
            return model, "state0", {"class": "Epileptor", "output": "fast_subsystem_state0", "active_x0": -1.6}

        raise AssertionError("unreachable")

    def _build_coupling(self, config: TVBSignalConfig, coupling: Any) -> Any:
        if config.signal_type in {"resting_jr", "spike_jr", "alpha_jr"}:
            if __package__ in {None, ""}:
                from tvb_jr import StableSigmoidalJansenRit, SparseSigmoidalJansenRit
            else:
                from .tvb_jr import StableSigmoidalJansenRit, SparseSigmoidalJansenRit
            implementation = SparseSigmoidalJansenRit if config.jr_coupling_mode == "sparse" else StableSigmoidalJansenRit
            return implementation(a=np.array([float(config.coupling_strength)]))
        return coupling.Linear(a=np.array([float(config.coupling_strength)]))

    def _noise_sigma(self, config: TVBSignalConfig, model_meta: dict[str, Any]) -> np.ndarray:
        if model_meta["class"] == "JansenRit":
            sigma = np.zeros(6, dtype=float)
            phi_n_scaling = (0.05 * float(config.jr_noise_scale)) ** 2
            sigma[4] = phi_n_scaling
            return sigma
        if model_meta["class"] == "Epileptor":
            sigma = np.zeros(6, dtype=float)
            sigma[0] = 1e-5
            return sigma
        return np.array([1e-4], dtype=float)

    @staticmethod
    def _extract_source(raw: np.ndarray, model_output: str) -> np.ndarray:
        arr = np.asarray(raw)
        required_states = 3 if model_output == "jr_pyramidal_difference" else 1
        if arr.ndim != 4 or arr.shape[1] < required_states or arr.shape[3] != 1:
            raise ValueError(f"Expected TVB time x state x region x one mode, got {arr.shape}")
        if model_output == "jr_pyramidal_difference":
            source = arr[:, 1, :, 0] - arr[:, 2, :, 0]
        else:
            source = arr[:, 0, :, 0]
        return source.T

    @staticmethod
    def _discard_initial(time_ms: np.ndarray, region_source: np.ndarray, discard_ms: float) -> tuple[np.ndarray, np.ndarray]:
        if discard_ms <= 0:
            return time_ms, region_source
        keep = (time_ms > float(discard_ms)) & ~np.isclose(time_ms, discard_ms, rtol=0, atol=1e-7)
        return time_ms[keep], region_source[:, keep]

    @staticmethod
    def _reshape_regions(region_source: np.ndarray, reshape_to_994: bool) -> np.ndarray:
        indices = TVBSourceGenerator._output_region_indices(region_source.shape[0], reshape_to_994)
        return region_source[indices]

    @staticmethod
    def _output_region_indices(n_regions: int, reshape_to_994: bool) -> np.ndarray:
        if not reshape_to_994:
            return np.arange(n_regions)
        if n_regions != 998:
            raise ValueError("The 994-region output atlas requires exactly 998 TVB regions; disable reshape for other atlases")
        indices = np.arange(994)
        indices[[7, 325, 921, 949]] = [994, 997, 996, 995]
        return indices

    @staticmethod
    def _resample(time_ms: np.ndarray, region_source: np.ndarray, output_srate_hz: float, duration_ms: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if time_ms.ndim != 1 or region_source.ndim != 2 or region_source.shape[1] != time_ms.size or not time_ms.size:
            raise ValueError("Expected nonempty region x time data matching the time vector")
        if not np.isfinite(time_ms).all() or not np.isfinite(output_srate_hz) or output_srate_hz <= 0:
            raise ValueError("Time values and positive output sampling rate must be finite")
        if time_ms.size > 1:
            steps = np.diff(time_ms)
            if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=0, atol=1e-7):
                raise ValueError("TVB time samples must be uniformly spaced and increasing")
            measured_duration_ms = float(steps[0] * time_ms.size)
            if duration_ms is not None and not np.isclose(duration_ms, measured_duration_ms, rtol=0, atol=1e-7):
                raise ValueError("Retained TVB samples do not match requested duration")
            duration_ms = measured_duration_ms
        if duration_ms is None or not np.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("A positive duration is required for resampling")
        sample_count = duration_ms * output_srate_hz / 1000.0
        n_samples = int(round(sample_count))
        if n_samples < 1 or not np.isclose(sample_count, n_samples, rtol=0, atol=1e-7):
            raise ValueError("Retained duration must contain a whole number of output samples")
        resampled = resample(region_source, num=n_samples, axis=1)
        new_time = np.arange(n_samples, dtype=float) * (1000.0 / output_srate_hz)
        return new_time, resampled

    @staticmethod
    def _validate_regions(active_regions: list[int], n_regions: int) -> list[int]:
        if any(not isinstance(region, (int, np.integer)) or isinstance(region, (bool, np.bool_)) for region in active_regions):
            raise ValueError("Active regions must be integer atlas indices")
        regions = [int(region) for region in active_regions]
        if len(regions) != len(set(regions)):
            raise ValueError("Active region IDs must be unique")
        for region in regions:
            if region < 0 or region >= n_regions:
                raise ValueError(f"active region {region} outside [0, {n_regions})")
        return regions

    @staticmethod
    def _validate_config(config: TVBSignalConfig) -> None:
        if config.jr_coupling_mode not in {"dense", "sparse"}:
            raise ValueError("jr_coupling_mode must be dense or sparse")
        for name in ("simulation_length_ms", "dt_ms", "output_srate_hz", "jr_background_A"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("warmup_ms", "discard_ms", "jr_noise_scale"):
            value = getattr(config, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("jr_active_A", "jr_mu", "coupling_strength"):
            value = getattr(config, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if config.discard_ms >= config.simulation_length_ms:
            raise ValueError("discard_ms must be shorter than simulation_length_ms")
        for name in ("warmup_ms", "simulation_length_ms", "discard_ms"):
            steps = getattr(config, name) / config.dt_ms
            if not np.isclose(steps, round(steps), rtol=0, atol=1e-7):
                raise ValueError(f"{name} must be a whole multiple of dt_ms")
        samples = (config.simulation_length_ms - config.discard_ms) * config.output_srate_hz / 1000.0
        if samples < 1 or not np.isclose(samples, round(samples), rtol=0, atol=1e-7):
            raise ValueError("Retained duration must contain a whole number of output samples")
        if not isinstance(config.random_seed, (int, np.integer)) or not 0 <= config.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32)")


def config_from_args(args: Any) -> TVBSignalConfig:
    profile = get_atlas_profile(args.atlas) if args.atlas is not None else None
    if profile and args.no_reshape_to_994 and profile['reshape_to_994']:
        raise ValueError('--no-reshape-to-994 conflicts with the named legacy998 export atlas')
    connectivity_path = args.connectivity_path or (str(profile['connectivity_path']) if profile else None)
    if profile:
        validate_atlas_connectivity(profile, connectivity_path)
    return TVBSignalConfig(
        sample_id=args.sample_id,
        signal_type=args.signal_type,
        connectivity_path=connectivity_path,
        active_regions=[int(v) for v in args.active_regions.split(",") if v.strip()],
        simulation_length_ms=args.simulation_length_ms,
        warmup_ms=args.warmup_ms,
        dt_ms=args.dt_ms,
        output_srate_hz=args.output_srate_hz,
        random_seed=args.random_seed,
        reshape_to_994=profile['reshape_to_994'] if profile else not args.no_reshape_to_994,
        normalize=not args.no_normalize,
        discard_ms=args.discard_ms,
        jr_background_A=args.jr_background_A,
        jr_active_A=args.jr_active_A,
        jr_mu=args.jr_mu,
        jr_noise_scale=args.jr_noise_scale,
        coupling_strength=args.coupling_strength,
        jr_coupling_mode=args.jr_coupling_mode,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate region-level source activity with TVB.")
    parser.add_argument("--atlas", choices=ATLAS_CHOICES, default=None,
                        help="Named connectivity/output atlas; omit for legacy defaults or custom inputs")
    parser.add_argument("--signal-type", choices=sorted(TVBSourceGenerator.SUPPORTED_SIGNAL_TYPES), default="spike_jr")
    parser.add_argument("--sample-id", default="G3_tvb_sample")
    parser.add_argument("--active-regions", default="10")
    parser.add_argument("--connectivity-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "Simulation_G3" / "generated_tvb_demo"))
    parser.add_argument("--simulation-length-ms", type=float, default=1000.0,
                        help="Post-warm-up window in ms (default: 1000)")
    parser.add_argument("--warmup-ms", type=float, default=1000.0,
                        help="Extra startup time to simulate and discard before saving (default: 1000 ms)")
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--output-srate-hz", type=float, default=500.0)
    parser.add_argument("--discard-ms", type=float, default=0.0,
                        help="Additional trim after warm-up; shortens the saved window (default: 0)")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--jr-background-A", type=float, default=3.25)
    parser.add_argument("--jr-active-A", type=float, default=None, help="Active JR A; defaults depend on signal type")
    parser.add_argument("--jr-mu", type=float, default=None, help="JR mean drive; defaults depend on signal type")
    parser.add_argument("--jr-noise-scale", type=float, default=1.0)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument("--jr-coupling-mode", choices=("sparse", "dense"), default="sparse",
                        help="Equivalent delayed JR coupling; dense retains the reference implementation")
    parser.add_argument("--no-reshape-to-994", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)

    config = config_from_args(args)
    manifest = TVBSourceGenerator(config.connectivity_path).save_sample(args.output_dir, config)
    print(json.dumps({"sample_id": manifest["sample_id"], "signal_type": manifest["signal_type"], "shape": manifest["region_source_shape"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
