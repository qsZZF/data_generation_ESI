"""Opt-in, evidence-bounded G1 configurations; ordinary defaults are unchanged."""
from copy import deepcopy
from pathlib import Path

from .g1_simulation_core import G1SimulationConfig

PRESET_NAMES = ('rest_calibrated', 'rest_lbo', 'erp_network')
BASIS_PATH = 'generated_simulation_dataset/G1_spectral_validation/fsaverage5_lbo256.npz'
NETWORK_PATH = 'projected_source_priors/fsaverage5_20k/neuro_motifs/neuro_motifs_fsaverage5_20k_source_priors_signed_maxabs.npz'


def make_g1_config(preset, *, resource_root=None, **overrides):
    """Build an independent config; explicit spectral overrides merge with the preset.

    Relative basis/network paths resolve against resource_root (project root by
    default). Geometry, hashes and resource existence are checked by generation.
    """
    if preset not in PRESET_NAMES:
        raise ValueError(f'Unknown G1 preset {preset!r}; choose from {PRESET_NAMES}')
    root = Path(resource_root).resolve() if resource_root is not None else Path(__file__).resolve().parents[1]
    spectral = dict(basis_path=BASIS_PATH, n_modes=256, scenario='rest',
                    spatial_length_mm=40. if preset=='rest_calibrated' else 20.,
                    spatial_exponent=2., temporal_exponent=1.6, alpha_hz=10.5,
                    alpha_width_hz=1.5, alpha_power_ratio=.2, beta_power_ratio=0.,
                    network_fraction=0. if preset=='rest_lbo' else .3)
    if preset != 'rest_lbo':
        spectral.update(network_path=NETWORK_PATH, network_indices=[3,5,6])
    if preset == 'erp_network':
        spectral.update(scenario='erp', erp_spatial_mode='geodesic',
            erp_local_width_mm=35., erp_roi_power_ratio=2., erp_component_interval_s=.2,
            erp_secondary_gain=.2, event_onset_s=.2, erp_latency_s=.15,
            erp_width_s=.06, erp_latency_jitter_s=.015,
            temporal_exponent=1.2, alpha_power_ratio=.5)
    supplied = deepcopy(overrides)
    spectral_overrides = supplied.pop('spectral', {})
    if not isinstance(spectral_overrides, dict):
        raise ValueError('spectral overrides must be a dictionary')
    spectral.update(spectral_overrides)
    for key in ('basis_path', 'network_path'):
        if key in spectral:
            path = Path(spectral[key])
            spectral[key] = str(path if path.is_absolute() else root/path)
    settings = dict(source_model='lbo' if preset=='rest_lbo' else 'lbo_network',
        spectral=spectral, srate=256., time_window=(0.,2.), baseline_duration_s=2.,
        sensor_noise_type='iid_white', sensor_snr_db=20., sensor_snr_mode='active_rms',
        covariance_estimator='oas', average_reference=True)
    settings.update(supplied)
    config = G1SimulationConfig(**settings)
    config.validate_recipe()
    return config
