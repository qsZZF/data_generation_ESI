from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

try:
    from .g2_simulation_core import (
        G2ScenarioSampler,
        G2SimulationConfig,
        G2SourceSimulator,
        SpatialDynamicsConfig,
        NMMRegionSignalRepository,
        load_brainstorm_cortex,
        load_brainstorm_leadfield,
        load_projection_matrix,
        load_region_filter,
    )
except ImportError:
    from g2_simulation_core import (
        G2ScenarioSampler,
        G2SimulationConfig,
        G2SourceSimulator,
        SpatialDynamicsConfig,
        NMMRegionSignalRepository,
        load_brainstorm_cortex,
        load_brainstorm_leadfield,
        load_projection_matrix,
        load_region_filter,
    )

__all__ = [
    "G2ScenarioSampler",
    "G2SimulationConfig",
    "G2SourceSimulator",
    "SpatialDynamicsConfig",
    "NMMRegionSignalRepository",
    "load_brainstorm_cortex",
    "load_brainstorm_leadfield",
    "load_projection_matrix",
    "load_region_filter",
]
