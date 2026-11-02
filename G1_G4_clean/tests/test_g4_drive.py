"""Drive semantics, legacy trajectory, and bank/reconstruction identity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Simulation_G4 import simulator
from Simulation_G4.bank import generate_bank, project_bank


class G4DriveTests(unittest.TestCase):
    def test_legacy_shared_trajectory_fixture(self):
        parameters = simulator.sample_parameters(np.random.default_rng(4), 4)
        source, _, _ = simulator.simulate(parameters, np.full((4, 4), .01),
                                         duration_s=.5, warmup_s=.25, seed=23)
        # Captured before the drive change, NumPy float64 / existing Heun path.
        self.assertEqual(hashlib.sha256(source.tobytes()).hexdigest(),
                         '8843c000273f0800c4e012af55203e153b51960035e364ead6a5fed0bac536bc')

    def test_drive_variance_correlation_endpoints_and_prefix_stability(self):
        shared = simulator._generate_drive(7, 100000, 4, 'shared', 1.)
        independent = simulator._generate_drive(7, 100000, 4, 'independent', 0.)
        mixed = simulator._generate_drive(7, 100000, 4, 'mixed', .15)
        np.testing.assert_array_equal(shared[:, 0], np.random.default_rng(7).normal(220, 22, 100000))
        np.testing.assert_array_equal(shared, simulator._generate_drive(7, 100000, 4, 'mixed', 1.))
        np.testing.assert_array_equal(independent, simulator._generate_drive(7, 100000, 4, 'mixed', 0.))
        for mode, rho, drive in [('independent', 0., independent), ('mixed', .15, mixed)]:
            np.testing.assert_array_equal(drive[:100], simulator._generate_drive(7, 100, 4, mode, rho))
            np.testing.assert_allclose(drive.mean(axis=0), 220, atol=.3)
            np.testing.assert_allclose(drive.std(axis=0), 22, atol=.3)
            correlations = np.corrcoef(drive.T)[np.triu_indices(4, 1)]
            np.testing.assert_allclose(correlations, rho, atol=.015)

    def test_invalid_drive_config_rejected(self):
        for mode, rho in [('unknown', None), ('mixed', -1), ('mixed', np.nan),
                          ('mixed', 1.1), ('shared', .5), ('independent', .2)]:
            with self.subTest(mode=mode, rho=rho), self.assertRaises(ValueError):
                simulator.drive_config(mode, rho)
        self.assertEqual(simulator.drive_config(), {'drive_mode': 'shared', 'drive_correlation': 1.})

    def test_zero_coupling_identical_regions_no_longer_forced_identical(self):
        parameters = simulator.sample_parameters(np.random.default_rng(4), 1)
        parameters = {k: np.repeat(v, 10) if k != 'delay_ms' else v for k, v in parameters.items()}
        source, _, _ = simulator.simulate(parameters, np.zeros((10, 10)), duration_s=1, seed=9)
        np.testing.assert_array_equal(source, np.broadcast_to(source[0], source.shape))
        independent, _, metadata = simulator.simulate(parameters, np.zeros((10, 10)),
            duration_s=1, seed=9, drive_mode='independent')
        self.assertGreater(np.linalg.norm(independent - independent[:1]), 1.)
        self.assertEqual(metadata['drive_correlation'], 0.)
        self.assertEqual(metadata['source_unit'], 'mV_PSP')

    def test_bank_drive_identity_and_corrupt_sample_guard(self):
        options = dict(n_draws=10, duration_s=.25, warmup_s=.1, sampling_rate_hz=128,
                       atlas_id='test', atlas_hash='test')
        with tempfile.TemporaryDirectory() as directory:
            config, _ = generate_bank(np.zeros((2, 2)), directory, **options)
            self.assertEqual(config['drive_mode'], 'independent')
            path = Path(directory) / 'sample_000000.npz'
            with np.load(path) as sample:
                values = dict(sample)
                meta = json.loads(str(sample['metadata_json']))
                self.assertEqual(meta['drive_mode'], 'independent')
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                generate_bank(np.zeros((2, 2)), directory, drive_mode='shared', **options)
            meta['drive_mode'] = 'shared'
            meta['drive_correlation'] = 1.
            values['metadata_json'] = json.dumps(meta)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'drive'):
                project_bank(directory, np.eye(2), 32)

    def test_legacy_config_and_samples_without_drive_fields_mean_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = generate_bank(np.zeros((2, 2)), directory, n_draws=10,
                duration_s=.25, warmup_s=.1, sampling_rate_hz=128,
                atlas_id='test', atlas_hash='test', drive_mode='shared')
            expected, _ = project_bank(directory, np.eye(2), 32)
            config.pop('drive_mode')
            config.pop('drive_correlation')
            (Path(directory) / 'config.json').write_text(json.dumps(config))
            for path in Path(directory).glob('sample_*.npz'):
                with np.load(path) as sample:
                    values = dict(sample)
                metadata = json.loads(str(values['metadata_json']))
                metadata.pop('drive_mode')
                metadata.pop('drive_correlation')
                values['metadata_json'] = json.dumps(metadata)
                np.savez(path, **values)
            actual, _ = project_bank(directory, np.eye(2), 32)
            np.testing.assert_array_equal(actual, expected)
            # Canonicalize missing drive fields without bypassing the code hash guard.
            generate_bank(np.zeros((2, 2)), directory, n_draws=10,
                duration_s=.25, warmup_s=.1, sampling_rate_hz=128,
                atlas_id='test', atlas_hash='test', drive_mode='shared')

    def test_mixed_bank_serial_and_process_workers_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            directories = [Path(directory) / str(workers) for workers in (1, 2)]
            for workers, destination in zip((1, 2), directories):
                generate_bank(np.zeros((2, 2)), destination, n_draws=10,
                    duration_s=.25, warmup_s=.1, sampling_rate_hz=128,
                    atlas_id='test', atlas_hash='test', workers=workers,
                    drive_mode='mixed', drive_correlation=.2)
            for index in range(10):
                with np.load(directories[0] / f'sample_{index:06d}.npz') as serial, \
                     np.load(directories[1] / f'sample_{index:06d}.npz') as parallel:
                    np.testing.assert_array_equal(serial['source'], parallel['source'])
                    self.assertEqual(str(serial['metadata_json']), str(parallel['metadata_json']))


if __name__ == '__main__':
    unittest.main()
