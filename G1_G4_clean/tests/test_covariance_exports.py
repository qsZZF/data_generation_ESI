"""Legacy tuple and standard NPZ consumers receive whitening covariance too."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from Simulation_G2.g2_simulation_core import G2SimulationConfig, G2SourceSimulator
from simulate_source_sensor import save_standard_sample


class CovarianceExportTests(unittest.TestCase):
    def test_noiseless_legacy_baseline_is_explicit_and_json_finite(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with patch.object(self.backend, 'generate_source', side_effect=lambda _: (self.source, self.source, {})):
                *_, metadata = self.backend.generate_sensor(replace(self.config, sensor_snr_db=None))
        np.testing.assert_array_equal(metadata['baseline_sensor'], np.zeros((4, 400)))
        self.assertEqual(metadata['noise_covariance_info']['rank'], 0)
        json.dumps(metadata, allow_nan=False)

    def setUp(self):
        self.backend = object.__new__(G2SourceSimulator)
        self.backend.leadfield = np.random.default_rng(11).normal(size=(4, 6))
        self.source = np.random.default_rng(12).normal(size=(6, 200)).astype(np.float32)
        self.config = G2SimulationConfig(srate=200, time_window=(0., 1.), sensor_noise_type='iid_white', random_seed=14)

    def test_legacy_and_standard_exports_keep_covariance_without_noise_timeseries(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(self.backend, 'generate_source', side_effect=lambda _: (self.source, self.source, {})):
                sensor, clean, noise, source, metadata = self.backend.generate_sensor(self.config)
                np.testing.assert_allclose(metadata['noise_covariance'], np.cov(np.asarray(metadata['baseline_sensor'], dtype=float)))
                np.testing.assert_allclose(metadata['oracle_noise_covariance'], np.cov(noise.astype(float)))
                manifest = self.backend.save_sample(folder, self.config, save_sensor_noise=False)
            with np.load(Path(folder) / manifest['noise_covariance_file']) as saved:
                np.testing.assert_array_equal(saved['noise_covariance'], metadata['noise_covariance'])
                self.assertEqual(saved['noise_covariance'].dtype, np.float64)
            self.assertNotIn('noise_covariance', manifest)
            standard = save_standard_sample(Path(folder), 'standard', sensor, clean, noise,
                                            source, None, metadata, False)
            with np.load(Path(folder) / 'standard' / standard['standard_data']) as saved:
                np.testing.assert_allclose(saved['noise_covariance'], np.cov(saved['baseline_sensor'].astype(float)))
                self.assertFalse(np.array_equal(saved['baseline_sensor'], noise))
            self.assertNotIn('noise_covariance', standard)
            json.dumps(standard)

    def test_standard_export_preserves_supplied_total_nuisance_covariance(self):
        noise = np.random.default_rng(13).normal(size=(4, 200)).astype(np.float32)
        total_covariance = np.cov(3 * noise.astype(float))
        metadata = {'noise_covariance': total_covariance.tolist(),
                    'oracle_noise_covariance': np.cov(noise.astype(float)).tolist(),
                    'sensor_noise_covariance': np.cov(noise.astype(float)).tolist(),
                    'baseline_sensor': (3 * noise).tolist()}
        with tempfile.TemporaryDirectory() as folder:
            manifest = save_standard_sample(Path(folder), 'total', noise, noise * 0, noise,
                                            self.source, None, metadata, False)
            with np.load(Path(folder) / 'total' / manifest['standard_data']) as saved:
                np.testing.assert_array_equal(saved['noise_covariance'], total_covariance)
