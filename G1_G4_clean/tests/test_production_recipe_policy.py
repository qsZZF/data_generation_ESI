"""Production recipes use explicit cortical backgrounds and audit fixed selections."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Simulation_G1.g1_simulation_core import G1ScenarioSampler
from Simulation_G1.training_recipes import (_variant, _parse_args, _read_settings,
    build_recipe_dataset, RecipeDataset, LEVELS, SCHEMA)
from scripts.audit_training_recipes import stratified_indices, audit_dataset
from test_spatial_patches import grid_simulator


class ProductionRecipePolicyTests(unittest.TestCase):
    def test_default_and_explicit_background_policies(self):
        families = G1ScenarioSampler().waveforms
        supports = [dict(index=i, support_id=str(i), split='train', difficulty='hard', seed_vertices=[1, 2])
                    for i in range(30)]
        for support in supports:
            row = _variant(support, 0, {'time_window': [0, 2]}, 42, families)
            self.assertEqual(row['config']['sensor_noise_type'], 'iid_white')
            self.assertIsNone(row['config']['source_background_type'])
            self.assertIsNone(row['config']['source_snr_db'])
        types = set()
        for support in supports:
            row = _variant(support, 0, {'time_window': [0, 2],
                           'source_background_spectral': {'basis_path': 'explicit.npz'}}, 42, families)
            types.add(row['config']['source_background_type'])
        self.assertEqual(types, {None, 'spectral_rest'})
        self.assertEqual(SCHEMA, 'g1-recipes-v4')

    def test_production_json_resolves_root_and_cli_overrides(self):
        root = Path(__file__).resolve().parents[1]
        config = root/'Simulation_G1/training_recipe.production.json'
        args = _parse_args(['--config', str(config)])
        self.assertEqual((args.n_supports, args.variants_per_support, args.eval_variants_per_support), (25000, 32, 16))
        self.assertEqual(20000*args.variants_per_support+5000*args.eval_variants_per_support, 720000)
        self.assertEqual((args.srate, args.duration_s, args.baseline_duration_s), (256, 2, 2))
        self.assertEqual(args.path_root, root)
        settings = _read_settings(args.background_spectral_config, args.path_root)
        self.assertTrue(Path(settings['basis_path']).is_absolute())
        self.assertEqual(args.background_multiscale_config['fraction'], .1)
        changed = _parse_args(['--config', str(config), '--n-supports', '60', '--variants-per-support', '20'])
        self.assertEqual((changed.n_supports, changed.variants_per_support), (60, 20))

    def test_stratified_sample_covers_marginal_states_and_is_not_prefix(self):
        strata = {(split, difficulty, profile, arrangement, count, family): [index, index+100]
                  for index, (split, difficulty, profile, arrangement, count, family) in enumerate([
                      ('train', 'easy', 'uniform', 'single', 1, 'erp'),
                      ('val', 'medium', 'decay', 'mixed', 2, 'spike'),
                      ('test', 'hard', 'diffuse', 'synchronous', 4, 'ersp'),
                      ('train', 'extreme', 'contract', 'same_family', 8, 'oscillatory'),
                      ('train', 'hard', 'pulse', 'mixed', 3, 'erp,spike')])}
        selected = stratified_indices(strata, 5, 42)
        self.assertEqual(selected, stratified_indices(strata, 5, 42))
        self.assertEqual(len(set(selected)), 5)
        self.assertTrue(any(index >= 100 for index in selected))
        self.assertTrue(all(set(indices)&set(selected) for indices in strata.values()))

    def test_audit_records_failure_without_replacing_selected_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sim = grid_simulator(25)
            levels = tuple((name, fraction, counts, (.10, .25), sensor, source, 2.)
                           for name, fraction, counts, areas, sensor, source, spacing in LEVELS)
            with patch('Simulation_G1.training_recipes.LEVELS', levels):
                build_recipe_dataset(sim, root/'recipes', n_supports=10, variants_per_support=5,
                                     eval_variants_per_support=5, srate=100, duration_s=1)
            dataset = RecipeDataset(root/'recipes', sim, split=None)
            original = sim.generate_sample
            calls = []

            def injected(config, **kwargs):
                calls.append(config.sample_id)
                if len(calls) == 1:
                    raise ValueError('deliberate audit fixture failure')
                return original(config, **kwargs)

            with patch.object(sim, 'generate_sample', side_effect=injected):
                report = audit_dataset(dataset, root/'audit', count=8)
            records = [json.loads(line) for line in (root/'audit/samples.jsonl').read_text().splitlines()]
            self.assertEqual([record['recipe_index'] for record in records], report['selected_indices'])
            self.assertEqual(len(calls), 8)
            self.assertEqual(records[0]['status'], 'failed')
            self.assertIn('deliberate', records[0]['exception'])
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(sum(report['results_count'].values()), 8)
            self.assertTrue(report['support_splits_disjoint'])


if __name__ == '__main__':
    unittest.main()
