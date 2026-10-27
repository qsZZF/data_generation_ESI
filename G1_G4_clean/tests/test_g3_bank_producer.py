"""Bank plans fail before simulation and resume only verified, identical inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import h5py
import numpy as np

from Simulation_G3.generate_nmm_bank import generate_bank


class BankProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with h5py.File(self.root / 'projection.mat', 'w') as f:
            g = f.create_group('Wnmm2source')
            g.attrs['MATLAB_sparse'] = 2
            g['data'] = [1., 1.]
            g['ir'] = [0, 1]
            g['jc'] = [0, 1, 2]
        with ZipFile(self.root / 'connectivity.zip', 'w') as z:
            z.writestr('weights.txt', '0 1\n1 0\n')
            z.writestr('tract_lengths.txt', '0 10\n10 0\n')
            z.writestr('centres.txt', 'A 0 0 0\nB 1 1 1\n')
        self.plan = {
            'atlas_id': 'toy2', 'projection_path': 'projection.mat',
            'defaults': {'connectivity_path': 'connectivity.zip',
                         'reshape_to_994': False, 'active_regions': [0],
                         'simulation_length_ms': 10, 'warmup_ms': 10,
                         'output_srate_hz': 500},
            'samples': [{'sample_id': 'rest', 'signal_type': 'resting_jr'},
                        {'sample_id': 'alpha', 'signal_type': 'alpha_jr', 'random_seed': 1}],
        }
        self.plan_path = self.root / 'plan.json'
        self.output = self.root / 'bank'

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf-8')

    @staticmethod
    def fake_generate(config):
        return np.ones((2, 5), np.float32), np.arange(5) * 2., {
            'sample_id': config.sample_id, 'signal_type': config.signal_type,
            'active_regions': config.active_regions, 'output_srate_hz': 500.,
            'normalize': config.normalize,
        }

    def test_dry_run_resolves_paths_without_constructing_generator_or_writes(self):
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.__init__',
                   side_effect=AssertionError('dry run instantiated simulator')):
            result = generate_bank(self.plan_path, self.output, dry_run=True)
        self.assertEqual(result['pending'], ['rest', 'alpha'])
        self.assertFalse(self.output.exists())

    def test_named_atlas_resolves_both_graph_and_projection(self):
        self.plan = {'atlas': 'aal78', 'defaults': {'active_regions': [0],
                     'simulation_length_ms': 10, 'warmup_ms': 10, 'output_srate_hz': 500},
                     'samples': [{'sample_id': 'rest', 'signal_type': 'resting_jr'}]}
        self.write_plan()
        profile = dict(atlas_id='toy2', n_regions=2, n_regions_raw=2, region_names=['A', 'B'],
                       connectivity_path=self.root / 'connectivity.zip',
                       projection_path=self.root / 'projection.mat', reshape_to_994=False)
        with patch('Simulation_G3.generate_nmm_bank.get_atlas_profile', return_value=profile), \
             patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate', side_effect=self.fake_generate) as generate:
            result = generate_bank(self.plan_path, self.output)
            self.assertEqual(result['atlas_id'], 'toy2')
            config = generate.call_args.args[0]
            self.assertFalse(config.reshape_to_994)
            self.assertEqual(config.connectivity_path, str((self.root / 'connectivity.zip').resolve()))
            profile['region_names'] = ['B', 'A']
            with self.assertRaisesRegex(ValueError, 'region order'):
                generate_bank(self.plan_path, self.output, dry_run=True)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            generate_bank(self.plan_path, self.output, atlas='legacy998')

    def test_invalid_later_config_prevents_any_generation(self):
        self.plan['samples'][1]['active_regions'] = [2]
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate') as generate:
            with self.assertRaises(ValueError):
                generate_bank(self.plan_path, self.output)
            generate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_duplicate_and_unsafe_ids_rejected(self):
        for bad_id in ('rest', 'REST', '../escape', 'folder/name', 'bad:name', 'CON', 'x' * 101):
            with self.subTest(bad_id=bad_id):
                self.plan['samples'][1]['sample_id'] = bad_id
                self.write_plan()
                with self.assertRaises(ValueError):
                    generate_bank(self.plan_path, self.output, dry_run=True)

    def test_resume_checks_provenance_before_any_new_generation(self):
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate',
                   side_effect=self.fake_generate) as generate:
            first = generate_bank(self.plan_path, self.output)
            self.assertEqual(first['generated'], ['rest', 'alpha'])
            self.assertTrue(all(not call.args[0].normalize for call in generate.call_args_list))
            self.assertEqual(generate.call_args_list[0].args[0].connectivity_path,
                             str((self.root / 'connectivity.zip').resolve()))
            generate.reset_mock()
            second = generate_bank(self.plan_path, self.output)
            self.assertEqual(second['reused'], ['rest', 'alpha'])
            generate.assert_not_called()
            self.plan['samples'].insert(0, {'sample_id': 'new'})
            self.plan['samples'][-1]['random_seed'] = 99
            self.write_plan()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                generate_bank(self.plan_path, self.output)
            generate.assert_not_called()

    def test_changed_connectivity_is_not_reused(self):
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate',
                   side_effect=self.fake_generate):
            generate_bank(self.plan_path, self.output)
        with ZipFile(self.root / 'connectivity.zip', 'a') as z:
            z.writestr('note.txt', 'changed connectivity archive')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            generate_bank(self.plan_path, self.output, dry_run=True)

    def test_damaged_source_and_changed_tvb_version_are_not_reused(self):
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate',
                   side_effect=self.fake_generate):
            generate_bank(self.plan_path, self.output)
        with patch('Simulation_G3.generate_nmm_bank.version', return_value='changed-version'):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                generate_bank(self.plan_path, self.output, dry_run=True)
        (self.output / 'rest.npz').write_bytes(b'damaged')
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate') as generate:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                generate_bank(self.plan_path, self.output)
            generate.assert_not_called()

    def test_projection_identity_mismatch_prevents_generation(self):
        self.write_plan()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate',
                   side_effect=self.fake_generate):
            generate_bank(self.plan_path, self.output)
        with h5py.File(self.root / 'projection.mat', 'r+') as f:
            f['Wnmm2source/data'][0] = 2.
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate') as generate:
            with self.assertRaisesRegex(ValueError, 'projection'):
                generate_bank(self.plan_path, self.output)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
