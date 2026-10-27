"""Configuration-only generation, frozen assets and deterministic indexed replay."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1.training_recipes import (RecipeDataset, build_recipe_dataset, generator_identity,
                                           _counts, LEVELS, PROFILES)
from test_spatial_patches import grid_simulator


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sim = grid_simulator(25)
        # Keep many small supports feasible while covering all requested area bands.
        self.levels = tuple((name, fraction, counts, (.10, .25), sensor, source, 2.)
                           for name, fraction, counts, areas, sensor, source, spacing in LEVELS)

    def build(self, name='set', **kwargs):
        with patch('Simulation_G1.training_recipes.LEVELS', self.levels):
            return build_recipe_dataset(self.sim, self.root/name, n_supports=20,
                variants_per_support=10, eval_variants_per_support=5, shard_size=13,
                srate=100, duration_s=1, **kwargs)

    def test_generation_does_not_compute_signals_and_split_grouping_is_fixed(self):
        with patch.object(self.sim, 'generate_source', side_effect=AssertionError('waveform generation')):
            manifest = self.build()
        self.assertEqual(manifest['split_support_counts'], {'train':16,'val':2,'test':2})
        self.assertEqual(manifest['n_samples'], 180)
        self.assertEqual(set(manifest['counts']['spatial_profile']), set(PROFILES))
        dataset = RecipeDataset(self.root/'set', self.sim, split=None)
        seen = {}
        for index in range(len(dataset)):
            config, row = dataset.recipe(index)
            self.assertEqual(len(config.patch_vertices), config.n_patch)
            self.assertIsNotNone(config.component_seeds)
            self.assertNotEqual(config.sensor_noise_type, 'real')
            if row['support_id'] in seen:
                self.assertEqual(row['split'], seen[row['support_id']])
            seen[row['support_id']] = row['split']
        self.assertEqual(len(seen), 20)
        train = RecipeDataset(self.root/'set', self.sim)
        self.assertEqual(len(train), 160)
        self.assertTrue(all(train.recipe(i)[1]['split']=='train' for i in range(len(train))))
        for label in ('val','test'):
            subset=RecipeDataset(self.root/'set',self.sim,split=label)
            self.assertEqual(len(subset),10)
            self.assertTrue(all(subset.recipe(i)[1]['split']==label for i in range(len(subset))))

    def test_short_evaluation_variants_can_include_causal(self):
        from Simulation_G1.training_recipes import _variant
        from Simulation_G1.g1_simulation_core import G1ScenarioSampler
        row = _variant(dict(index=2, support_id='test', split='val',difficulty='hard',seed_vertices=[1,2]),
                       0, {'time_window':[0,2]}, 42, G1ScenarioSampler().waveforms)
        self.assertEqual(row['config']['connectivity_mode'], 'causal')

    def test_build_is_deterministic_and_indexed_replay_is_order_independent(self):
        self.build('one'); self.build('two')
        for path in (self.root/'one').rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (self.root/'two'/path.relative_to(self.root/'one')).read_bytes())
        dataset = RecipeDataset(self.root/'one', self.sim, split=None)
        first = dataset[0]
        dataset[1]
        repeated = dataset[0]
        for actual, expected in zip(first[:4], repeated[:4]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(first[4], repeated[4])
        self.assertNotIn('source_psd_summary', first[4])
        self.assertTrue(np.isfinite(first[0]).all())
        for invalid in (-1, len(dataset), True, 1.5):
            with self.assertRaises(IndexError):
                dataset.recipe(invalid)

    def test_shared_temporal_group_keeps_component_seeds_and_gains(self):
        self.build()
        dataset = RecipeDataset(self.root/'set', self.sim, split=None)
        configs = [dataset.recipe(i)[0] for i in range(5)]
        self.assertEqual([c.spatial['mode'] for c in configs], ['static','static','diffuse','contract','pulse'])
        self.assertEqual(configs[0].decay_mm, [0.]*configs[0].n_patch)
        for config in configs[1:]:
            self.assertEqual(config.waveform_params, configs[0].waveform_params)
            self.assertEqual(config.patch_gains, configs[0].patch_gains)
            for stream in ('waveform','gain','background'):
                self.assertEqual(config.component_seeds[stream],configs[0].component_seeds[stream])

    def test_changed_assets_code_and_corrupt_shard_are_rejected(self):
        self.build()
        other = grid_simulator(25)
        other.leadfield[0,0] += 1
        with self.assertRaisesRegex(ValueError,'mesh/leadfield'):
            RecipeDataset(self.root/'set', other)
        with patch('Simulation_G1.training_recipes.generator_identity', return_value={}):
            with self.assertRaisesRegex(ValueError,'version'):
                RecipeDataset(self.root/'set', self.sim)
        dataset = RecipeDataset(self.root/'set', self.sim, split=None)
        path = self.root/'set'/dataset.manifest['shards'][0]['file']
        path.write_text('corrupt',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'checksum'):
            dataset.recipe(0)

    def test_existing_dataset_is_not_overwritten(self):
        self.build()
        before=(self.root/'set/dataset.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before,(self.root/'set/dataset.json').read_bytes())

    def test_plan_counts_and_invalid_settings(self):
        self.assertEqual(_counts(25000,[.8,.1,.1]).tolist(), [20000,2500,2500])
        self.assertEqual(20000*32+5000*16,720000)
        with self.assertRaisesRegex(ValueError, 'srate'):
            build_recipe_dataset(self.sim,self.root/'bad',srate=50)
        self.assertFalse((self.root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
