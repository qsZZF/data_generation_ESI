"""Unified orchestration: source/sensor truth, independent rest and channel alignment."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G2.g2_simulation_core import G2SourceSimulator
from nmm_simulation import NMMSimulationConfig, NMMSourceSimulator


class UnifiedNMMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        projection = sparse.csc_matrix([[1, 0, 0], [.2, .7, 0], [0, .3, 1], [0, 0, .2]])
        sparse.save_npz(self.root / 'p.npz', projection)
        self.profile = dict(atlas_id='toy3', n_regions=3, n_regions_raw=3,
            connectivity_path=self.root / 'unused.zip', projection_path=self.root / 'p.npz',
            reshape_to_994=False, region_names=['a', 'b', 'c'])
        cortex = dict(Vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]),
            Faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            VertConn=sparse.csr_matrix(np.ones((4, 4)) - np.eye(4)))
        self.backend = G2SourceSimulator(np.array([[1, 0, .2, 0], [0, 1, 0, .3], [.1, 0, 1, 0]]),
                                         cortex, projection, self.root / 'bank')
        self.sim = NMMSourceSimulator(self.backend, channel_names=['a', 'b', 'c'])
        self.config = NMMSimulationConfig(atlas='aal78', active_regions=[1], source_mode='selected',
            duration_s=1, srate=100, source_snr_db=3, sensor_snr_db=10, random_seed=42)
        self.profile_patch = patch('nmm_simulation.get_atlas_profile', return_value=self.profile)
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        self.resolver = patch('nmm_simulation.resolve_trajectory', side_effect=self.fake_trajectory).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def fake_trajectory(config, **kwargs):
        t = np.arange(int(config.simulation_length_ms * config.output_srate_hz / 1000)) / config.output_srate_hz
        rest = config.signal_type == 'resting_jr'
        source = np.array([2 + np.sin(2*np.pi*(3 if rest else 7)*t + phase) for phase in (0, .3, 1)])
        entry = dict(sample_id=config.sample_id, atlas_id='toy3', source_unit='mV_PSP',
            simulation_group_id=str(config.random_seed),
            manifest=dict(active_regions=config.active_regions, signal_type=config.signal_type,
                random_seed=config.random_seed, normalize=False, region_names=['a', 'b', 'c'],
                model=dict(**{'class': 'JansenRit'}, background_A=3.25, active_A=3.25 if rest else 3.6)))
        return source, t*1000, entry

    def test_both_snrs_and_component_truth_survive_export_precision(self):
        result = self.sim.generate_sample(self.config)
        area = self.backend.vertex_area
        power = lambda x: np.average(np.mean(x.astype(float)**2, axis=1), weights=area)
        measured = 10*np.log10(power(result['source_foreground'])/power(result['source_background']))
        self.assertAlmostEqual(measured, 3, places=5)
        sensor_snr = 10*np.log10(np.mean(result['clean_sensor'].astype(float)**2)/np.mean(result['sensor_noise'].astype(float)**2))
        self.assertAlmostEqual(sensor_snr, 10, places=5)
        np.testing.assert_allclose(result['source'], result['source_foreground']+result['source_background'], atol=1e-7)
        np.testing.assert_allclose(result['source'], self.backend.projection_matrix @ result['region_source'], atol=1e-7)
        np.testing.assert_allclose(result['sensor'], result['clean_sensor']+result['sensor_noise'], atol=1e-7)
        np.testing.assert_allclose(result['clean_sensor'], result['foreground_sensor']+result['background_sensor'], atol=1e-7)
        self.assertEqual(len(result['manifest']['parent_simulation_group_ids']), 2)
        self.assertFalse(result['manifest']['parameter_preset_is_activation_truth'])
        self.assertEqual(len(self.sim.generate_sensor(self.config)), 5)

    def test_rest_background_cannot_share_seed_or_contain_perturbed_regions(self):
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.sim.generate_sample(replace(self.config, background_seed=42))
        def corrupted(config, **kwargs):
            source, times, entry = self.fake_trajectory(config, **kwargs)
            if config.signal_type == 'resting_jr':
                entry['manifest']['active_regions'] = [1]
            return source, times, entry
        self.resolver.side_effect = corrupted
        with self.assertRaisesRegex(ValueError, 'without perturbed'):
            self.sim.generate_sample(self.config)
        def missing_parameters(config, **kwargs):
            source, times, entry = self.fake_trajectory(config, **kwargs)
            if config.signal_type == 'resting_jr':
                entry['manifest']['model'].pop('active_A')
                entry['manifest']['model'].pop('background_A')
            return source, times, entry
        self.resolver.side_effect = missing_parameters
        with self.assertRaisesRegex(ValueError, 'without perturbed'):
            self.sim.generate_sample(self.config)

    def test_no_background_is_zero_and_snr_disabled_is_not_a_fake_number(self):
        result = self.sim.generate_sample(replace(self.config, source_snr_db=None, sensor_snr_db=None))
        self.assertFalse(np.any(result['source_background']))
        self.assertFalse(np.any(result['sensor_noise']))
        self.assertIsNone(result['manifest']['realized_source_snr_db'])
        self.assertIsNone(result['manifest']['realized_sensor_snr_db'])

    def test_sensor_seed_changes_only_measurement_noise(self):
        first = self.sim.generate_sample(replace(self.config, sensor_seed=11))
        second = self.sim.generate_sample(replace(self.config, sensor_seed=12))
        np.testing.assert_array_equal(first['source'], second['source'])
        np.testing.assert_array_equal(first['clean_sensor'], second['clean_sensor'])
        self.assertFalse(np.array_equal(first['sensor_noise'], second['sensor_noise']))

    def test_real_noise_alignment_and_sampling_are_explicit(self):
        config = replace(self.config, sensor_noise_type='real')
        noise = np.random.default_rng(19).normal(size=(3, 200))
        first = self.sim.generate_sample(config, real_sensor_noise=noise,
            real_noise_srate=100, real_noise_channel_names=['a', 'b', 'c'])
        second = self.sim.generate_sample(config, real_sensor_noise=noise[::-1],
            real_noise_srate=100, real_noise_channel_names=['c', 'b', 'a'])
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
        with self.assertRaisesRegex(ValueError, 'sampling rate'):
            self.sim.generate_sample(config, real_sensor_noise=noise, real_noise_srate=200,
                                     real_noise_channel_names=['a', 'b', 'c'])

    def test_real_noise_content_crop_and_shared_recording_are_traceable(self):
        config = replace(self.config, sensor_noise_type='real')
        noise = np.random.default_rng(19).normal(size=(3, 250))
        kwargs = dict(real_noise_srate=100, real_noise_channel_names=['a', 'b', 'c'], real_noise_recording_id='recording_01')
        first = self.sim.generate_sample(config, real_sensor_noise=noise, **kwargs)
        second = self.sim.generate_sample(config, real_sensor_noise=-noise, **kwargs)
        a, b = first['manifest']['real_noise_provenance'], second['manifest']['real_noise_provenance']
        self.assertNotEqual(a['input_sha256'], b['input_sha256'])
        self.assertNotEqual(a['selected_sha256'], b['selected_sha256'])
        self.assertEqual(a['parent_group_id'], b['parent_group_id'])
        self.assertEqual(a['start_sample'], b['start_sample'])
        self.assertEqual(a['n_samples'], 200)
        self.assertEqual(a['baseline_stop_sample'], a['event_start_sample'])
        self.assertEqual(len(first['manifest']['parent_data_group_ids']), 3)

    def test_saved_sample_contains_component_truth_and_refuses_overwrite(self):
        metadata = self.sim.save_sample(self.root / 'out', self.config)
        folder = self.root / 'out' / self.config.sample_id
        with np.load(folder / metadata['data_file']) as arrays:
            for key in ('region_foreground', 'region_background', 'source_foreground',
                        'source_background', 'foreground_sensor', 'background_sensor'):
                self.assertIn(key, arrays.files)
        json.loads((folder / (self.config.sample_id+'_manifest.json')).read_text())
        with self.assertRaises(FileExistsError):
            self.sim.save_sample(self.root / 'out', self.config)


if __name__ == '__main__':
    unittest.main()
