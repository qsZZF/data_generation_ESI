"""Registered measured noise stays aligned, traceable and split-safe in NMM APIs."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from Simulation_G1.real_noise import create_noise_registry, load_noise_window
from Simulation_G2.training_dataset import NMMRecipeDataset
import test_unified_nmm as unified_fixture
import test_nmm_training_dataset as period_fixture


def registry(root, names, *, role='resting_background', splits=('train',), sensor_units='V'):
    records = []
    for index, split in enumerate(splits):
        values = np.random.default_rng(31 + index).normal(size=(len(names), 500))
        values -= values.mean(axis=0, keepdims=True)
        path = root / f'noise_{split}.npz'
        np.savez(path, noise=values)
        records.append(dict(path=str(path), record_id=f'record_{split}', recording_id=f'acquisition_{split}',
            subject_id=f'noise_subject_{split}', split=split, fs_hz=100., channel_names=names,
            sensor_units=sensor_units, reference='average', noise_role=role))
    path = root / 'registry.json'
    create_noise_registry(records, path)
    return path


class RegisteredUnifiedNMMTests(unittest.TestCase):
    setUp = unified_fixture.UnifiedNMMTests.setUp
    fake_trajectory = staticmethod(unified_fixture.UnifiedNMMTests.fake_trajectory)

    def test_registered_crop_and_ancestors_are_preserved(self):
        path = registry(self.root, ['c', 'b', 'a'])
        config = replace(self.config, source_snr_db=None, sensor_noise_type='real', sensor_seed=77,
            real_noise_registry=str(path), real_noise_record_id='record_train', real_noise_split='train')
        first = self.sim.generate_sample(config)
        again = self.sim.generate_sample(config)
        np.testing.assert_array_equal(first['sensor_noise'], again['sensor_noise'])
        info = first['manifest']['real_noise_provenance']
        joint, expected = load_noise_window(path, 'record_train', expected_srate=100.,
            expected_channels=['a', 'b', 'c'], expected_reference='average',
            baseline_samples=100, event_samples=100, seed=77, split='train')
        self.assertEqual(info['selected_sha256'], expected['selected_sha256'])
        self.assertEqual(info['start_sample'], expected['start_sample'])
        self.assertEqual(info['identity_status'], 'registered_verified')
        self.assertTrue(set(info['parent_data_group_ids']).issubset(first['manifest']['parent_data_group_ids']))
        event = joint[:, 100:] - joint[:, 100:].mean(axis=1, keepdims=True)
        event -= event.mean(axis=0, keepdims=True)
        self.assertGreater(np.corrcoef(event.ravel(), first['sensor_noise'].ravel())[0, 1], .999999)
        with self.assertRaisesRegex(ValueError, 'bare'):
            self.sim.generate_sample(config, real_sensor_noise=np.zeros((3, 200)))
        with self.assertRaisesRegex(ValueError, 'split'):
            self.sim.generate_sample(replace(config, real_noise_split='test'))

    def test_registered_neural_background_needs_explicit_combination_opt_in(self):
        path = registry(self.root, ['a', 'b', 'c'])
        config = replace(self.config, sensor_noise_type='real', real_noise_registry=str(path),
                         real_noise_record_id='record_train')
        with self.assertRaisesRegex(ValueError, 'allow_combined'):
            self.sim.generate_sample(config)
        sample = self.sim.generate_sample(replace(config, allow_combined_real_source_background=True))
        self.assertTrue(np.any(sample['source_background']))
        self.assertEqual(sample['manifest']['real_noise_provenance']['noise_role'], 'resting_background')

    def test_physical_registered_noise_uses_units_and_fixed_amplitude(self):
        path = registry(self.root, ['a', 'b', 'c'])
        config = replace(self.config, source_snr_db=None, sensor_noise_type='real', sensor_snr_db=None,
            real_noise_registry=str(path), real_noise_record_id='record_train',
            amplitude_mode='physical', foreground_peak_am=1e-9, leadfield_source_unit='A*m',
            sensor_units='V', sensor_noise_std=1e-6, normalize_source=False)
        first = self.sim.generate_sample(config)
        second = self.sim.generate_sample(replace(config, foreground_peak_am=2e-9))
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
        np.testing.assert_array_equal(first['baseline_sensor'], second['baseline_sensor'])
        np.testing.assert_allclose(second['source_foreground'], first['source_foreground'] * 2, rtol=1e-6)
        for std in (None, 0., [0., 0., 0.]):
            with self.subTest(std=std), self.assertRaisesRegex(ValueError, 'positive sensor_noise_std'):
                self.sim.generate_sample(replace(config, sensor_noise_std=std))

    def test_relative_registered_noise_rejects_disabled_noise_and_mixed_units(self):
        path = registry(self.root, ['a', 'b', 'c'], sensor_units=['V', 'uV', 'V'])
        config = replace(self.config, source_snr_db=None, sensor_noise_type='real',
            real_noise_registry=str(path), real_noise_record_id='record_train')
        with self.assertRaisesRegex(ValueError, 'mixed sensor units'):
            self.sim.generate_sample(config)
        with self.assertRaisesRegex(ValueError, 'finite sensor_snr_db'):
            self.sim.generate_sample(replace(config, sensor_snr_db=None))
        self.sim._channel_names = ['a', 'b']
        with self.assertRaisesRegex(ValueError, 'forward-model channel names'):
            self.sim.generate_sample(config)


class RegisteredPeriodNMMTests(unittest.TestCase):
    setUp = period_fixture.NMMTrainingDatasetTests.setUp
    save = period_fixture.NMMTrainingDatasetTests.save
    build = period_fixture.NMMTrainingDatasetTests.build

    def setup_registry(self, **kwargs):
        self.names = [f'channel_{index}' for index in range(self.backend.n_sensors)]
        return registry(self.root, self.names, **kwargs)

    def test_v5_recipe_replay_carries_ancestry_and_freezes_registry(self):
        path = self.setup_registry(splits=('train', 'val', 'test'))
        manifest = self.build(real_noise_registry=path, sensor_noise_types=('real',))
        self.assertEqual(manifest['schema'], 'nmm-periods-v5')
        with self.assertRaisesRegex(ValueError, 'channel_names'):
            NMMRecipeDataset(self.root / 'index', self.backend)
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=self.names)
        row = dataset.recipe(0)
        self.assertEqual(row['config']['real_noise_record_id'], 'record_train')
        first = dataset[0]
        repeated = dataset[0]
        np.testing.assert_array_equal(first['sensor_noise'], repeated['sensor_noise'])
        np.testing.assert_array_equal(first['baseline_sensor'], repeated['baseline_sensor'])
        provenance = first['manifest']['real_noise_provenance']
        self.assertEqual(provenance['split'], 'train')
        self.assertEqual(provenance['baseline_stop_sample'], provenance['event_start_sample'])
        for parent in provenance['parent_data_group_ids']:
            self.assertIn(parent, row['parent_data_group_ids'])
            self.assertEqual(manifest['parent_splits'][parent], 'train')
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dataset[0]

    def test_noise_data_tamper_and_missing_matching_split_fail(self):
        path = self.setup_registry(splits=('val',))
        with self.assertRaisesRegex(ValueError, 'split train'):
            self.build(real_noise_registry=path, sensor_noise_types=('real',))
        # Correctly declared split, then altered underlying recording.
        other = self.root / 'valid'
        other.mkdir()
        path = registry(other, self.names)
        self.build(real_noise_registry=path, sensor_noise_types=('real',))
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=self.names)
        dataset[0]
        np.savez(other / 'noise_train.npz', noise=np.ones((self.backend.n_sensors, 500)))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dataset[0]

    def test_partition_requires_neural_noise_opt_in_but_empty_room_does_not(self):
        path = self.setup_registry()
        options = dict(real_noise_registry=path, sensor_noise_types=('real',), source_mode='partition', source_snr_db=10.)
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            self.build(**options)
        self.build(**options, allow_combined_real_source_background=True)
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=self.names)
        sample = dataset[0]
        self.assertTrue(np.any(sample['source_background']))
        empty = self.root / 'empty'
        empty.mkdir()
        path = registry(empty, self.names, role='empty_room')
        self.build(folder='empty_index', **dict(options, real_noise_registry=path))
        sample = NMMRecipeDataset(self.root / 'empty_index', self.backend, channel_names=self.names)[0]
        self.assertEqual(sample['manifest']['real_noise_provenance']['noise_role'], 'empty_room')

    def test_each_fold_uses_its_own_noise_ancestors(self):
        path = self.setup_registry(splits=('train', 'val', 'test'))
        self.save(self.bank, 'two', seed=2, group='2' * 64)
        self.save(self.bank, 'three', seed=3, group='3' * 64)
        self.build(real_noise_registry=path, sensor_noise_types=('real',), split_fractions=(1/3, 1/3, 1/3))
        parents = {}
        for split in ('train', 'val', 'test'):
            dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=self.names, split=split)
            sample = dataset[0]
            info = sample['manifest']['real_noise_provenance']
            self.assertEqual(info['record_id'], f'record_{split}')
            parents[split] = set(info['parent_data_group_ids'])
        self.assertFalse(parents['train'] & parents['val'])
        self.assertFalse(parents['train'] & parents['test'])
        self.assertFalse(parents['val'] & parents['test'])

    def test_inherited_real_ancestry_cannot_be_reassigned_to_another_fold(self):
        path = self.setup_registry(splits=('train', 'val'))
        noise_registry = json.loads(path.read_text())
        val_parent = noise_registry['records'][1]['parent_group_id']
        bank_path = self.bank / 'bank.json'
        bank = json.loads(bank_path.read_text())
        bank['samples']['one']['provenance']['parent_data_group_ids'] = [val_parent]
        bank_path.write_text(json.dumps(bank))
        with self.assertRaisesRegex(ValueError, 'ancestry conflicts'):
            self.build(real_noise_registry=path, sensor_noise_types=('real',))
        self.assertFalse((self.root / 'index').exists())

    def test_relative_period_noise_requires_common_units_and_enabled_snr(self):
        self.names = [f'channel_{index}' for index in range(self.backend.n_sensors)]
        path = registry(self.root, self.names, sensor_units=['V'] + ['uV'] * (len(self.names) - 1))
        with self.assertRaisesRegex(ValueError, 'units'):
            self.build(real_noise_registry=path, sensor_noise_types=('real',))
        with self.assertRaisesRegex(ValueError, 'finite sensor_snrs'):
            self.build(real_noise_registry=path, sensor_noise_types=('real',), sensor_snrs=(None,))

    def test_physical_registered_period_noise_cannot_be_silently_disabled(self):
        from unittest.mock import patch
        path = self.setup_registry()
        amplitude = dict(amplitude_mode='physical', foreground_peak_am=1e-9,
                         leadfield_source_unit='A*m', sensor_units='V', sensor_noise_std=1e-6)
        options = dict(real_noise_registry=path, sensor_noise_types=('real',), sensor_snrs=(None,),
                       normalize_source=False, amplitude_settings=amplitude)
        for std in (None, 0., [0.] * self.backend.n_sensors):
            with self.subTest(std=std), self.assertRaisesRegex(ValueError, 'positive sensor_noise_std'):
                self.build(**dict(options, amplitude_settings=dict(amplitude, sensor_noise_std=std)))
        self.build(**options)
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=self.names)
        dataset[0]
        row = dataset.recipe(0)
        dataset.manifest['amplitude']['sensor_noise_std'] = None
        row['config']['amplitude']['sensor_noise_std'] = None
        with patch.object(dataset, 'recipe', return_value=row), self.assertRaisesRegex(ValueError, 'positive sensor_noise_std'):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
