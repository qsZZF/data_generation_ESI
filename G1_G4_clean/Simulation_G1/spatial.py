"""Shared foreground support checks and phenomenological spatial envelopes.

The envelope changes weights inside a fixed support; it is not a diffusion PDE.
Time is elapsed seconds from the first exported sample, before source demeaning.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialDynamicsConfig:
    mode: str = "auto"
    area_threshold_cm2: float | None = 10.0
    hop_threshold: int | None = 5
    start_s: float = 0.0
    duration_s: float = 0.3

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mode not in {"static", "auto", "diffuse", "contract", "pulse"}:
            raise ValueError("Spatial mode must be static, auto, diffuse, contract or pulse")
        if self.area_threshold_cm2 is not None:
            if (isinstance(self.area_threshold_cm2, bool)
                    or not np.isfinite(self.area_threshold_cm2) or self.area_threshold_cm2 <= 0):
                raise ValueError("area_threshold_cm2 must be positive and finite, or None")
        if self.hop_threshold is not None:
            if (isinstance(self.hop_threshold, bool) or not isinstance(self.hop_threshold, (int, np.integer))
                    or self.hop_threshold < 1):
                raise ValueError("hop_threshold must be a positive integer, or None")
        if not np.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("Spatial start_s must be finite and nonnegative")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("Spatial duration_s must be positive and finite")


def resolve_spatial_config(config):
    """Accept dataclasses and JSON mappings at every public generation boundary."""
    if config is None:
        config = SpatialDynamicsConfig()
    if isinstance(config, dict):
        config = SpatialDynamicsConfig(**config)
    if not isinstance(config, SpatialDynamicsConfig):
        raise ValueError("spatial must be a SpatialDynamicsConfig or mapping")
    config.validate()
    return config


def validate_disjoint_supports(supports):
    """Reject intersecting foreground supports without trimming or moving them."""
    occupied = set()
    for index, support in enumerate(supports):
        vertices = set(np.asarray(support, dtype=int).tolist())
        if not vertices:
            raise ValueError(f"Foreground support {index} is empty")
        if len(vertices) != len(support):
            raise ValueError(f"Foreground support {index} contains duplicate vertices")
        overlap = occupied.intersection(vertices)
        if overlap:
            raise ValueError(f"Foreground patches overlap at {len(overlap)} vertices (support {index})")
        occupied.update(vertices)


def gaussian_weights(distances_mm, decay_mm):
    """Half-amplitude radius in mm; zero/None preserves uniform static weights."""
    distances = np.asarray(distances_mm, dtype=float)
    if distances.ndim != 1 or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Spatial distances must be a finite nonnegative vector")
    if decay_mm is None or decay_mm == 0:
        return np.ones_like(distances)
    if not np.isfinite(decay_mm) or decay_mm < 0:
        raise ValueError("Spatial decay must be nonnegative and finite")
    return np.exp(-np.log(2.0) * (distances / decay_mm)**2)


def patch_hop_order(indices, seed, conn):
    """Maximum shortest-path hop distance inside the support, None if disconnected."""
    support = set(np.asarray(indices, dtype=int).tolist())
    if seed not in support:
        raise ValueError("Patch seed must belong to its support")
    remaining = support - {int(seed)}
    frontier = {int(seed)}
    hops = 0
    while remaining:
        following = set()
        for vertex in frontier:
            following.update(conn.indices[conn.indptr[vertex]:conn.indptr[vertex+1]])
        frontier = following.intersection(remaining)
        if not frontier:
            return None
        remaining.difference_update(frontier)
        hops += 1
    return hops


def spatial_envelope(base_weights, times, config, *, area_mm2, hop_order):
    """Return patch×time weights (patch×1 for static) and a replayable schedule."""
    config = resolve_spatial_config(config)
    base = np.asarray(base_weights, dtype=float)
    times = np.asarray(times, dtype=float)
    if (base.ndim != 1 or not base.size or not np.isfinite(base).all()
            or np.any((base < 0) | (base > 1))):
        raise ValueError("Base spatial weights must be a nonempty finite vector in [0, 1]")
    if times.ndim != 1 or not times.size or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Spatial times must be a finite nondecreasing vector")
    if area_mm2 is not None and (not np.isfinite(area_mm2) or area_mm2 < 0):
        raise ValueError("Patch area must be nonnegative and finite")
    area_trigger = (area_mm2 is not None and config.area_threshold_cm2 is not None
                    and area_mm2 >= 100 * config.area_threshold_cm2)
    hop_trigger = (hop_order is not None and config.hop_threshold is not None
                   and hop_order >= config.hop_threshold)
    mode = config.mode
    if mode == "auto":
        mode = "diffuse" if area_trigger or hop_trigger else "static"
    u = np.clip((times - config.start_s) / config.duration_s, 0., 1.)
    smooth = u*u*(3-2*u)
    if mode == "static":
        mixing = np.zeros(1)
    elif mode == "diffuse":
        mixing = smooth
    elif mode == "contract":
        mixing = 1-smooth
    else:
        # Rise and fall with zero derivative at both endpoints and at the peak.
        phase = 1-np.abs(2*u-1)
        mixing = phase*phase*(3-2*phase)
    weights = base[:, None] + (1-base[:, None])*mixing[None, :]
    metadata = {
        "requested_mode": config.mode, "mode": mode,
        "area_mm2": None if area_mm2 is None else float(area_mm2),
        "hop_order": None if hop_order is None else int(hop_order),
        "area_trigger": bool(area_trigger), "hop_trigger": bool(hop_trigger),
        "start_s": float(config.start_s), "duration_s": float(config.duration_s),
        "time_origin": "first_exported_sample",
        "weight_rule": "base + (1-base)*mixing; u=clip((elapsed-start)/duration,0,1); smooth=u*u*(3-2*u)",
        "mixing_rule": {"static": "0", "diffuse": "smooth", "contract": "1-smooth",
                        "pulse": "v*v*(3-2*v), v=1-abs(2*u-1)"}[mode],
        "initial_weights": weights[:, 0].tolist(), "final_weights": weights[:, -1].tolist(),
        "transition_complete": bool(times[-1] >= config.start_s + config.duration_s),
    }
    return weights, metadata
