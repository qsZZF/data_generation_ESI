"""Registered noise recipes preserve acquisition splits and frozen resource identity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1.real_noise import create_noise_registry
from Simulation_G1.training_recipes import (RecipeDataset, build_recipe_dataset,
                                            generator_identity, LEVELS, _variant)
from Simulation_G1.g1_simulation_core import G1ScenarioSampler
from test_spatial_patches import grid_simulator


class RegisteredNoiseRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sim = grid_simulator(25)
        self.sim.channel_names = ['A', 'B', 'C', 'D']
        self.levels = tuple((name, fraction, counts, (.10, .25), sensor, source, 2.)
                           for name, fraction, counts, areas, sensor, source, spacing in LEVELS)
        records = []
        for index, split in enumerate(('train', 'val', 'test')):
            path = self.root / f'{split}.npz'
            noise = np.random.default_rng(index).normal(size=(4, 800))
            noise -= noise.mean(axis=0)
            np.savez(path, noise=noise)
            records.append(dict(path=str(path), record_id=f'{split}_noise',
                recording_id=f'acquisition_{index}', subject_id=f'private_subject_{index}',
                split=split, fs_hz=100, channel_names=self.sim.channel_names,
                sensor_units='V', reference='average', noise_role='resting_background'))
        self.registry = self.root / 'noise_registry.json'
        self.registry_manifest = create_noise_registry(records, self.registry)

    def build(self, name='set', **kwargs):
        with patch('Simulation_G1.training_recipes.LEVELS', self.levels):
            return build_recipe_dataset(self.sim, self.root/name, n_supports=10,
                variants_per_support=5, eval_variants_per_support=5, shard_size=13,
                srate=100, duration_s=1, real_noise_registry=self.registry, **kwargs)

    def test_variant_draws_are_replayable_under_production_policy(self):
        rows = [_variant(dict(index=2, support_id='fixture', split='train', difficulty='hard',
                             seed_vertices=[1, 2]), i, {'time_window': [0, 2]}, 42,
                         G1ScenarioSampler().waveforms) for i in range(10)]
        repeated = [_variant(dict(index=2, support_id='fixture', split='train', difficulty='hard',
                                  seed_vertices=[1, 2]), i, {'time_window': [0, 2]}, 42,
                              G1ScenarioSampler().waveforms) for i in range(10)]
        self.assertEqual(rows, repeated)
        self.assertTrue(all(row['config']['sensor_noise_type'] == 'iid_white' for row in rows))

    def test_matching_split_only_and_no_default_double_background(self):
        with patch.object(self.sim, 'generate_sensor', side_effect=AssertionError('signal generation')):
            manifest = self.build()
        self.assertEqual(manifest['schema'], 'g1-recipes-v4')
        self.assertTrue(manifest['real_sensor_noise_enabled'])
        groups = {row['split']: row['parent_group_id'] for row in self.registry_manifest['records']}
        for split in ('train', 'val', 'test'):
            dataset = RecipeDataset(self.root/'set', self.sim, split=split)
            for index in range(len(dataset)):
                config, row = dataset.recipe(index)
                self.assertEqual(config.sensor_noise_type, 'real')
                self.assertEqual(config.real_noise_record_id, f'{split}_noise')
                self.assertEqual(config.real_noise_split, split)
                self.assertIsNone(config.source_background_type)
                self.assertIsNone(config.source_snr_db)
                self.assertEqual(row['real_noise']['parent_group_id'], groups[split])
                self.assertNotIn('private_subject', json.dumps(row))
                for other in set(groups)-{split}:
                    self.assertNotIn(groups[other], json.dumps(row))

    def test_registry_tamper_rejected_after_dataset_open_and_at_open(self):
        self.build()
        dataset = RecipeDataset(self.root/'set', self.sim)
        dataset.recipe(0)
        self.registry.write_bytes(self.registry.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'registry.*checksum'):
            dataset.recipe(0)
        with self.assertRaisesRegex(ValueError, 'registry.*checksum'):
            RecipeDataset(self.root/'set', self.sim)

    def test_combined_background_requires_explicit_opt_in_and_replay_is_deterministic(self):
        self.build(allow_combined_real_source_background=True)
        dataset = RecipeDataset(self.root/'set', self.sim, split=None)
        # Permission does not invent a source-background model when none is supplied.
        self.assertTrue(all(dataset.recipe(i)[0].source_background_type is None
                            for i in range(len(dataset))))
        first = dataset[0]
        dataset[1]
        repeated = dataset[0]
        for key in ('sensor', 'clean_sensor', 'sensor_noise', 'source_foreground', 'source_background', 'source_total'):
            np.testing.assert_array_equal(first[key], repeated[key])
        self.assertEqual(first['manifest'], repeated['manifest'])

    def test_missing_channels_or_compatible_split_fail_before_dataset_write(self):
        self.sim.channel_names = None
        with self.assertRaisesRegex(ValueError, 'channel_names'):
            self.build('missing_channels')
        self.assertFalse((self.root/'missing_channels').exists())
        self.sim.channel_names = ['wrong', 'B', 'C', 'D']
        with self.assertRaisesRegex(ValueError, 'compatible.*train'):
            self.build('missing_split')
        self.assertFalse((self.root/'missing_split').exists())

    def test_new_generators_are_fingerprinted(self):
        self.assertIn('real_noise.py', generator_identity()['files'])
        self.assertIn('multiscale.py', generator_identity()['files'])

    def test_multiscale_passthrough_only_with_enabled_spectral_background(self):
        from Simulation_G1.geometry import compute_basis, save_basis
        basis_path = self.root / 'basis.npz'
        save_basis(basis_path, compute_basis(self.sim.vertices*self.sim.coord_to_mm,
                                           self.sim.faces, n_modes=8))
        spectral = {'basis_path': str(basis_path)}
        multiscale = {'fraction': .2, 'n_components': 2, 'length_mm': 5.}
        self.build(source_background_spectral=spectral, source_background_multiscale=multiscale,
                   allow_combined_real_source_background=True)
        dataset = RecipeDataset(self.root/'set', self.sim, split=None)
        count = 0
        for index in range(len(dataset)):
            config, row = dataset.recipe(index)
            config.validate_recipe()
            if config.source_background_type == 'spectral_rest':
                self.assertEqual(config.source_background_multiscale, multiscale)
                count += 1
            else:
                self.assertEqual(config.source_background_multiscale, {})
        self.assertGreater(count, 0)
        self.build('no_double_background', source_background_spectral=spectral,
                   source_background_multiscale=multiscale)
        dataset = RecipeDataset(self.root/'no_double_background', self.sim, split=None)
        for index in range(len(dataset)):
            config, _ = dataset.recipe(index)
            self.assertEqual(config.source_background_multiscale, {})
            self.assertEqual(config.source_background_spectral, {})
            config.validate_recipe()

    def test_signal_file_tamper_is_detected_at_replay(self):
        self.build()
        dataset = RecipeDataset(self.root/'set', self.sim)
        np.savez(self.root/'train.npz', noise=np.zeros((4, 800)))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dataset[0]

    def test_relative_mixed_units_rejected_without_loading_signals(self):
        records = [dict(record, sensor_units=['V', 'uV', 'V', 'V'])
                   for record in self.registry_manifest['records']]
        self.registry = self.root/'mixed_registry.json'
        create_noise_registry(records, self.registry)
        with patch('Simulation_G1.real_noise._read_array', side_effect=AssertionError('array load')):
            with self.assertRaisesRegex(ValueError, 'compatible.*units'):
                self.build('mixed')
        self.assertFalse((self.root/'mixed').exists())

    def test_physical_dimensions_checked_in_simulator_channel_order(self):
        from Simulation_G1.training_recipes import _registered_noise_options
        records = [dict(record, sensor_units=['uV', 'fT', 'V', 'fT/cm'])
                   for record in self.registry_manifest['records']]
        self.registry = self.root/'physical_registry.json'
        create_noise_registry(records, self.registry)
        self.sim.channel_names = ['D', 'A', 'B', 'C']
        amplitude = dict(amplitude_mode='physical', foreground_peak_am=1e-9,
                         leadfield_source_unit='A*m', sensor_units=['T/m', 'V', 'T', 'V'],
                         sensor_noise_std=[1e-13, 1e-6, 1e-15, 1e-6])
        with patch('Simulation_G1.real_noise._read_array', side_effect=AssertionError('array load')):
            _, records = _registered_noise_options(self.registry, self.sim, srate=100,
                duration_s=1, required_splits=['train', 'val', 'test'], amplitude=amplitude)
            self.assertEqual(set(records), {'train', 'val', 'test'})
            with self.assertRaisesRegex(ValueError, 'identical destination'):
                self.build('mixed_destination', amplitude_settings=amplitude)
            with self.assertRaisesRegex(ValueError, 'compatible.*units'):
                self.build('bad_dimensions', amplitude_settings={**amplitude, 'sensor_units': 'V'})
        self.assertFalse((self.root/'bad_dimensions').exists())

    def test_physical_registered_noise_requires_nonzero_explicit_amplitude(self):
        amplitude = dict(amplitude_mode='physical', foreground_peak_am=1e-9,
                         leadfield_source_unit='A*m', sensor_units='V')
        for index, value in enumerate((None, 0., [0., 0., 0., 0.])):
            with self.assertRaisesRegex(ValueError, 'positive sensor_noise_std'):
                self.build(f'zero_{index}', amplitude_settings={**amplitude, 'sensor_noise_std': value})
            self.assertFalse((self.root/f'zero_{index}').exists())
        self.build('physical_positive', amplitude_settings={**amplitude, 'sensor_noise_std': 1e-6})
        dataset = RecipeDataset(self.root/'physical_positive', self.sim)
        self.assertGreater(np.linalg.norm(dataset[0]['sensor_noise']), 0)


if __name__ == '__main__':
    unittest.main()
