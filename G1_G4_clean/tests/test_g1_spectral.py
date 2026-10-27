"""Geometry-aware fields, baseline separation and evoked/induced signal contracts."""
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import welch

from Simulation_G1.geometry import compute_basis, load_basis, project_maps, save_basis
from Simulation_G1.g1_simulation_core import G1SimulationConfig
from Simulation_G1.spectral import _fft_process, generate_coefficients
from Simulation_G1.training_recipes import generator_identity
from test_spatial_patches import grid_simulator


class ModalGenerationTests(unittest.TestCase):
    def test_broad_erp_cannot_leak_into_prestimulus_baseline(self):
        settings=dict(scenario='erp',event_onset_s=0.,erp_latency_s=.1,erp_width_s=.5)
        out=generate_coefficients(self.bundle,settings,srate=128,time_window=(-2,3),
            baseline_duration_s=2,rng=np.random.default_rng(3))
        np.testing.assert_array_equal(out['foreground'][:,:256],0.)
        self.assertGreater(np.max(abs(out['foreground'][:,256:])),0.)
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            self.generate({**settings,'erp_latency_s':-.1})

    def setUp(self):
        rng = np.random.default_rng(4)
        self.mass = rng.uniform(.5, 2, 32)
        q, _ = np.linalg.qr(rng.normal(size=(32, 16)))
        self.bundle = dict(basis=q / np.sqrt(self.mass[:, None]), mass=self.mass,
                           eigenvalues=np.linspace(.0001, .02, 16), mesh_sha256='fixture')

    def generate(self, settings=None, *, seed=8, **kwargs):
        return generate_coefficients(self.bundle, settings or {}, srate=128,
            time_window=(0, 1), baseline_duration_s=2, rng=np.random.default_rng(seed), **kwargs)

    def test_mass_energy_and_replay_and_rest_baseline_target(self):
        first = self.generate()
        second = self.generate()
        for key in ('foreground', 'background', 'baseline_background'):
            np.testing.assert_array_equal(first[key], second[key])
        field = self.bundle['basis'] @ first['foreground']
        np.testing.assert_allclose(np.sum(self.mass[:, None] * field ** 2, axis=0),
                                   np.sum(first['foreground'] ** 2, axis=0), atol=1e-14)
        np.testing.assert_array_equal(first['background'], 0)
        np.testing.assert_array_equal(first['baseline_background'], np.zeros((16, 256)))
        self.assertEqual(first['metadata']['target_definition'], 'whole_resting_field')

    def test_spatial_length_suppresses_high_modes(self):
        rough = self.generate({'spatial_length_mm': 0})['foreground']
        smooth = self.generate({'spatial_length_mm': 60})['foreground']
        rough_ratio = np.mean(rough[-4:] ** 2) / np.mean(rough[:4] ** 2)
        smooth_ratio = np.mean(smooth[-4:] ** 2) / np.mean(smooth[:4] ** 2)
        self.assertLess(smooth_ratio, rough_ratio / 10)

    def test_erp_is_phase_locked_and_jitter_changes_latency(self):
        settings = dict(scenario='erp', erp_latency_jitter_s=0)
        first, second = self.generate(settings, seed=1), self.generate(settings, seed=2)
        np.testing.assert_array_equal(first['foreground'], second['foreground'])
        self.assertFalse(np.array_equal(first['background'], second['background']))
        first = self.generate({**settings, 'erp_latency_jitter_s': .02}, seed=1)
        second = self.generate({**settings, 'erp_latency_jitter_s': .02}, seed=2)
        self.assertNotEqual(first['metadata']['erp_latency_s'], second['metadata']['erp_latency_s'])
        for result in (first, second):
            peak = np.linalg.norm(result['foreground'], axis=0).argmax() / 128
            self.assertAlmostEqual(peak, result['metadata']['erp_latency_s'], delta=1 / 128)

    def test_ersp_is_only_modulation_delta_with_unchanged_baseline(self):
        unchanged = self.generate(dict(scenario='ersp', ersp_depth=0))
        enhanced = self.generate(dict(scenario='ersp', ersp_depth=.5))
        reduced = self.generate(dict(scenario='ersp', ersp_depth=-.5))
        for key in ('background', 'baseline_background'):
            np.testing.assert_array_equal(unchanged[key], enhanced[key])
            np.testing.assert_array_equal(unchanged[key], reduced[key])
        np.testing.assert_array_equal(unchanged['foreground'], 0)
        np.testing.assert_allclose(reduced['foreground'], -enhanced['foreground'])
        times = np.arange(128) / 128
        np.testing.assert_array_equal(enhanced['foreground'][:, (times < .2) | (times > .7)], 0)
        trials = np.asarray([self.generate(dict(scenario='ersp'), seed=seed)['foreground'] for seed in range(48)])
        self.assertLess(np.mean(trials.mean(axis=0) ** 2), .15 * np.mean(trials ** 2))

    def test_network_fraction_uses_selected_modal_subspace(self):
        coefficients = np.eye(16)[:, :3]
        result = self.generate(dict(network_fraction=1, network_indices=[0, 2]), network_coefficients=coefficients)
        np.testing.assert_array_equal(result['foreground'][[1, *range(3, 16)]], 0)
        self.assertGreater(np.linalg.norm(result['foreground'][[0, 2]]), 0)
        self.assertEqual(result['metadata']['network_indices'], [0, 2])

    def test_aperiodic_slope_and_alpha_peak_are_generating_spectra(self):
        def spectrum(settings):
            result = generate_coefficients(self.bundle, settings, srate=128,
                time_window=(0, 32), baseline_duration_s=2, rng=np.random.default_rng(18))
            freq, psd = welch(result['foreground'], fs=128, nperseg=512, axis=-1)
            return freq, psd.sum(axis=0)
        frequency, power = spectrum(dict(temporal_exponent=1.5, alpha_power_ratio=0))
        selected = (frequency >= 2) & (frequency <= 30)
        slope = np.polyfit(np.log(frequency[selected]), np.log(power[selected]), 1)[0]
        self.assertAlmostEqual(slope, -1.5, delta=.25)
        frequency, power = spectrum(dict(alpha_hz=10, alpha_width_hz=1, alpha_power_ratio=8))
        selected = (frequency >= 5) & (frequency <= 20)
        peak = frequency[selected][power[selected].argmax()]
        self.assertAlmostEqual(peak, 10, delta=1)

    def test_disabled_beta_preserves_draw_count_and_alpha_output(self):
        for scenario in ('rest', 'erp', 'ersp'):
            with self.subTest(scenario=scenario), patch('Simulation_G1.spectral._fft_process', wraps=_fft_process) as draw:
                original = self.generate(dict(scenario=scenario))
                self.assertEqual(draw.call_count, 2)
            disabled = self.generate(dict(scenario=scenario, beta_power_ratio=0, ersp_beta_depth=-.8))
            for key in ('foreground', 'background', 'baseline_background'):
                np.testing.assert_array_equal(original[key], disabled[key])

    def test_beta_ersp_controls_band_power_without_changing_baseline(self):
        settings = dict(scenario='ersp', event_onset_s=2, event_duration_s=8, spatial_length_mm=0,
                        alpha_power_ratio=0, ersp_depth=0, beta_hz=20,
                        beta_width_hz=1, beta_power_ratio=1000)

        def trial(depth):
            return generate_coefficients(self.bundle, {**settings, 'ersp_beta_depth': depth},
                srate=128, time_window=(0, 12), baseline_duration_s=2, rng=np.random.default_rng(27))

        unchanged, reduced, enhanced = trial(0), trial(-.8), trial(.6)
        for result in (reduced, enhanced):
            np.testing.assert_array_equal(result['background'], unchanged['background'])
            np.testing.assert_array_equal(result['baseline_background'], unchanged['baseline_background'])
            times = np.arange(12 * 128) / 128
            np.testing.assert_array_equal(result['foreground'][:, (times < 2) | (times > 10)], 0)
            self.assertIn('alpha_beta_amplitude_modulation_delta', result['metadata']['target_definition'])

        def beta_power(result):
            signal = result['foreground'] + result['background']
            frequency, power = welch(signal[:, 5 * 128:7 * 128], fs=128, nperseg=256, axis=-1)
            return power[:, (frequency >= 15) & (frequency <= 25)].sum()

        self.assertLess(beta_power(reduced), .2 * beta_power(unchanged))
        self.assertGreater(beta_power(enhanced), 1.8 * beta_power(unchanged))
        frequency, power = welch(reduced['foreground'], fs=128, nperseg=512, axis=-1)
        self.assertAlmostEqual(frequency[power.sum(axis=0).argmax()], 20, delta=1)

    def test_invalid_settings_and_timing_rejected(self):
        for settings in ({'unknown': 1}, {'spatial_length_mm': -1}, {'scenario': 'unknown'},
                         {'n_modes': 99}, {'alpha_hz': 70}, {'ersp_depth': -2},
                         {'beta_power_ratio': -1}, {'beta_width_hz': 0}, {'ersp_beta_depth': -2},
                         {'beta_hz': 70, 'beta_power_ratio': 1},
                         {'scenario': 'erp', 'erp_latency_s': 8},
                         {'scenario': 'ersp', 'event_duration_s': 2}, {'network_fraction': .8}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.generate(settings)


class SpectralSourceIntegrationTests(unittest.TestCase):
    def test_zero_evoked_target_does_not_label_background_as_active(self):
        config=replace(self.config,spectral={**self.config.spectral,'scenario':'erp','erp_amplitude':0.})
        source,meta=self.sim.generate_source(config)
        self.assertGreater(np.linalg.norm(source),0)
        self.assertEqual(meta['active_vertices'],[])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.sim = grid_simulator(5)
        self.bundle = compute_basis(self.sim.vertices * self.sim.coord_to_mm, self.sim.faces, n_modes=8)
        self.basis_path = self.path / 'basis.npz'
        save_basis(self.basis_path, self.bundle)
        self.config = G1SimulationConfig(source_model='lbo', srate=128, time_window=(0, 1),
            random_seed=4, sensor_noise_type='iid_white', normalize_source=True,
            spectral={'basis_path': str(self.basis_path)})

    def test_all_scenarios_replay_with_baseline_covariance(self):
        for scenario in ('rest', 'erp', 'ersp'):
            config = replace(self.config, spectral={**self.config.spectral, 'scenario': scenario})
            first = self.sim.generate_sensor(config, compute_diagnostics=False)
            second = self.sim.generate_sensor(config, compute_diagnostics=False)
            self.assertEqual(len(first), 5)
            for left, right in zip(first[:4], second[:4]):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(first[3].shape, (25, 128))
            np.testing.assert_array_equal(first[3], self.sim.generate_source(config)[0])
            metadata = first[4]
            np.testing.assert_allclose(metadata['noise_covariance'], np.cov(np.asarray(metadata['baseline_sensor'])))
            self.assertEqual(metadata['source_model'], 'lbo')
            self.assertEqual(metadata['srate'], 128)
            self.assertEqual(metadata['time_window'], [0, 1])
            self.assertEqual(metadata['source_units'], 'model_current_arbitrary')
            self.assertEqual(metadata['baseline_contains_source_background'], scenario != 'rest')

    def test_basis_hash_and_mesh_are_checked(self):
        self.sim.generate_source(replace(self.config, spectral={**self.config.spectral,
            'basis_sha256': hashlib.sha256(self.basis_path.read_bytes()).hexdigest()}))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.sim.generate_source(replace(self.config, spectral={**self.config.spectral, 'basis_sha256': 'wrong'}))
        self.sim.vertices[0, 0] += .01
        with self.assertRaises(ValueError):
            self.sim.generate_source(self.config)

    def test_basis_cache_reuses_assets_not_samples_and_checks_hash_on_hits(self):
        with patch('Simulation_G1.geometry.load_basis', wraps=load_basis) as loader:
            first = self.sim.generate_sensor(self.config, compute_diagnostics=False)
            repeated = self.sim.generate_sensor(self.config, compute_diagnostics=False)
            changed = self.sim.generate_sensor(replace(self.config, random_seed=99), compute_diagnostics=False)
            self.assertEqual(loader.call_count, 1)
            for left, right in zip(first[:4], repeated[:4]):
                np.testing.assert_array_equal(left, right)
            self.assertFalse(np.array_equal(first[3], changed[3]))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.sim.generate_source(replace(self.config, spectral={**self.config.spectral, 'basis_sha256': 'wrong'}))
            self.assertEqual(loader.call_count, 1)

    def test_basis_file_change_invalidates_cache_and_old_hash(self):
        old_hash = hashlib.sha256(self.basis_path.read_bytes()).hexdigest()
        with patch('Simulation_G1.geometry.load_basis', wraps=load_basis) as loader:
            original = self.sim.generate_source(self.config)[0]
            modified = {**self.bundle, 'eigenvalues': self.bundle['eigenvalues'] * 2}
            save_basis(self.basis_path, modified)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.sim.generate_source(replace(self.config, spectral={**self.config.spectral, 'basis_sha256': old_hash}))
            refreshed = self.sim.generate_source(self.config)[0]
            self.assertEqual(loader.call_count, 2)
            self.assertFalse(np.array_equal(original, refreshed))

    def test_network_projection_is_cached_and_file_changes_revalidated(self):
        network = self.path / 'network.npz'
        maps = self.bundle['basis'][:, :3].astype(np.float32)

        def save_maps(values):
            np.savez(network, source_prior_weights=values, source_prior_raw_values=values,
                     vertices_brainstorm_mm=(self.sim.vertices * self.sim.coord_to_mm).astype(np.float32),
                     faces=self.sim.faces)

        save_maps(maps)
        old_hash = hashlib.sha256(network.read_bytes()).hexdigest()
        config = replace(self.config, source_model='lbo_network', spectral={**self.config.spectral,
                         'network_path': str(network), 'network_fraction': 1})
        with patch('Simulation_G1.geometry.project_maps', wraps=project_maps) as projector:
            original = self.sim.generate_source(config)[0]
            repeated = self.sim.generate_source(config)[0]
            np.testing.assert_array_equal(original, repeated)
            self.assertEqual(projector.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'Network asset SHA256'):
                self.sim.generate_source(replace(config, spectral={**config.spectral, 'network_sha256': 'wrong'}))
            maps[:, 0] *= -1
            save_maps(maps)
            with self.assertRaisesRegex(ValueError, 'Network asset SHA256'):
                self.sim.generate_source(replace(config, spectral={**config.spectral, 'network_sha256': old_hash}))
            refreshed = self.sim.generate_source(config)[0]
            self.assertEqual(projector.call_count, 2)
            self.assertFalse(np.array_equal(original, refreshed))

    def test_network_mesh_validation_and_signed_map_integration(self):
        network = self.path / 'network_signed_maxabs.npz'
        maps = self.bundle['basis'][:, :3].astype(np.float32)
        np.savez(network, source_prior_weights=maps, source_prior_raw_values=maps,
                 vertices_brainstorm_mm=(self.sim.vertices * self.sim.coord_to_mm).astype(np.float32),
                 faces=self.sim.faces)
        config = replace(self.config, source_model='lbo_network', spectral={**self.config.spectral,
                         'network_path': str(network), 'scenario': 'erp'})
        result = self.sim.generate_sensor(config, compute_diagnostics=False)
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertEqual(result[4]['network_sha256'], hashlib.sha256(network.read_bytes()).hexdigest())
        np.savez(network, source_prior_weights=maps, source_prior_raw_values=maps,
                 vertices_brainstorm_mm=self.sim.vertices[::-1] * self.sim.coord_to_mm, faces=self.sim.faces)
        with self.assertRaisesRegex(ValueError, 'vertex rows'):
            self.sim.generate_source(config)

    def test_legacy_background_controls_rejected_and_recipe_identity_complete(self):
        with self.assertRaisesRegex(ValueError, 'legacy'):
            self.sim.generate_source(replace(self.config, source_snr_db=5))
        identity = generator_identity()['files']
        self.assertIn('geometry.py', identity)
        self.assertIn('spectral.py', identity)


if __name__ == '__main__':
    unittest.main()
