"""Resume and split identities must survive montage reuse."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Simulation_G4.bank import generate_bank, project_bank


MATRIX = np.array([[0., .1], [.05, 0.]])
BANK_OPTIONS = dict(n_draws=10, duration_s=1, sampling_rate_hz=100,
                    atlas_id='test', atlas_hash='fixture')


def small_simulator(parameters, connectivity, **kwargs):
    n = round(kwargs['duration_s'] * kwargs['sampling_rate_hz'])
    t = np.arange(n) / kwargs['sampling_rate_hz']
    source = np.array([np.sin(2 * np.pi * f * t) for f in (8, 12)])
    return source, t, {'seed': kwargs['seed']}


class SourceBankTests(unittest.TestCase):
    def test_default_256_hz_bank_and_legacy_rate_guard(self):
        options = {key: value for key, value in BANK_OPTIONS.items() if key != 'sampling_rate_hz'}
        with tempfile.TemporaryDirectory() as directory, \
             patch('Simulation_G4.bank.simulate', side_effect=small_simulator):
            config, _ = generate_bank(MATRIX, directory, **options)
            self.assertEqual(config['sampling_rate_hz'], 256)
            with np.load(Path(directory) / 'sample_000000.npz') as sample:
                self.assertEqual(sample['source'].shape, (2, 256))
                np.testing.assert_allclose(np.diff(sample['time_s']), 1 / 256)
        with tempfile.TemporaryDirectory() as directory, \
             patch('Simulation_G4.bank.simulate', side_effect=small_simulator):
            generate_bank(MATRIX, directory, **BANK_OPTIONS)
            before = (Path(directory) / 'config.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                generate_bank(MATRIX, directory, **options)
            self.assertEqual((Path(directory) / 'config.json').read_bytes(), before)

    def test_resume_and_montage_views_keep_draw_groups(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('Simulation_G4.bank.simulate', side_effect=small_simulator) as simulator:
            _, splits = generate_bank(MATRIX, directory, **BANK_OPTIONS)
            self.assertEqual(simulator.call_count, 10)
            _, repeated = generate_bank(MATRIX, directory, **BANK_OPTIONS)
            self.assertEqual(simulator.call_count, 10)
            self.assertEqual(splits, repeated)
            ids = [i for subset in splits.values() for i in subset]
            self.assertEqual(sorted(ids), list(range(10)))
            first, labels = project_bank(directory, np.eye(2), 100)
            second, other_labels = project_bank(directory, np.eye(2) * 3, 50)
            np.testing.assert_array_equal(labels, other_labels)
            np.testing.assert_allclose(second, first[..., :50] * 3, atol=1e-6)
            with self.assertRaises(ValueError):
                generate_bank(MATRIX * 2, directory, **BANK_OPTIONS)
            self.assertTrue((Path(directory) / 'splits.json').exists())

    def test_orphan_samples_fail_before_creating_configuration_or_generating(self):
        for suffix in ('.npz', '.part'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory, \
                 patch('Simulation_G4.bank.simulate') as simulator:
                path = Path(directory) / ('sample_000000' + suffix)
                path.write_bytes(b'unidentified sample')
                with self.assertRaisesRegex(ValueError, 'Orphan sample cache'):
                    generate_bank(MATRIX, directory, **BANK_OPTIONS)
                self.assertFalse((Path(directory) / 'config.json').exists())
                self.assertEqual(path.read_bytes(), b'unidentified sample')
                simulator.assert_not_called()

    def test_resume_and_projection_reject_wrong_seeds_targets_and_sources(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('Simulation_G4.bank.simulate', side_effect=small_simulator):
            generate_bank(MATRIX, directory, **BANK_OPTIONS)
            path = Path(directory) / 'sample_000000.npz'
            with np.load(path, allow_pickle=False) as data:
                original = {name: data[name] for name in data.files}
            changes = [
                ('draw_seed', original['draw_seed'] + 1, 'draw_seed'),
                ('noise_seed', original['noise_seed'] + 1, 'noise_seed'),
                ('noise_seed', np.array([original['noise_seed']]), 'noise_seed'),
                ('theta', original['theta'] + .001, 'parameter theta'),
                ('tau_e1_ms', original['tau_e1_ms'][:1], 'parameter tau_e1_ms'),
                ('delay_ms', np.array([original['delay_ms']]), 'parameter delay_ms'),
                ('delay_ms', original['delay_ms'] + .001, 'parameter delay_ms'),
                ('source', original['source'][:, :-1], 'source shape'),
                ('source', np.full_like(original['source'], np.nan), 'Nonfinite'),
                ('draw_seed', None, 'Missing'),
            ]
            for name, value, message in changes:
                with self.subTest(name=name, value_shape=np.shape(value)):
                    corrupt = dict(original)
                    if value is None:
                        del corrupt[name]
                    else:
                        corrupt[name] = value
                    np.savez_compressed(path, **corrupt)
                    with self.assertRaisesRegex(ValueError, message):
                        generate_bank(MATRIX, directory, **BANK_OPTIONS)
                    with self.assertRaisesRegex(ValueError, message):
                        project_bank(directory, np.eye(2), 100)
            # A full file copied from a different split/draw is also rejected.
            path.write_bytes((Path(directory) / 'sample_000001.npz').read_bytes())
            with self.assertRaisesRegex(ValueError, 'draw_seed'):
                generate_bank(MATRIX, directory, **BANK_OPTIONS)
            with self.assertRaisesRegex(ValueError, 'draw_seed'):
                project_bank(directory, np.eye(2), 100)


if __name__ == '__main__':
    unittest.main()
