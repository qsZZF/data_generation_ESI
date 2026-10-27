"""Unified providers share provenance while replay remains independent of TVB."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from scipy.sparse import eye, save_npz

from Simulation_G2.nmm_bank import read_bank
from Simulation_G3.g3_tvb_core import TVBSignalConfig
from Simulation_G3.nmm_sources import PRESETS, resolve_trajectory


class SourceProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank = self.root / 'bank'
        projection = self.root / 'projection.npz'
        save_npz(projection, eye(2, format='csc'))
        connectivity = self.root / 'connectivity.zip'
        with ZipFile(connectivity, 'w') as archive:
            archive.writestr('weights.txt', '0 1\n1 0\n')
            archive.writestr('tract_lengths.txt', '0 10\n10 0\n')
            archive.writestr('centres.txt', 'A 0 0 0\nB 1 1 1\n')
        self.profile = dict(atlas_id='toy2', n_regions=2, n_regions_raw=2,
                            connectivity_path=connectivity, projection_path=projection,
                            reshape_to_994=False, region_names=['A', 'B'])
        self.config = TVBSignalConfig(connectivity_path=str(connectivity), reshape_to_994=False,
                                     active_regions=[0], simulation_length_ms=10, warmup_ms=10,
                                     output_srate_hz=500, normalize=False)
        for name in ('Simulation_G3.nmm_sources', 'Simulation_G3.generate_nmm_bank'):
            patcher = patch(name + '.get_atlas_profile', return_value=self.profile)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.generate',
                        side_effect=self.fake_generate)
        self.generate = patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def fake_generate(config):
        return np.full((2, 5), config.random_seed + 1., dtype=np.float32), np.arange(5) * 2., {
            'sample_id': config.sample_id, 'signal_type': config.signal_type,
            'active_regions': config.active_regions, 'output_srate_hz': 500.,
            'normalize': config.normalize, 'random_seed': config.random_seed,
            'model': {'class': 'JansenRit'},
        }

    def resolve(self, provider='cache', config=None, **kwargs):
        return resolve_trajectory(config or self.config, provider=provider, atlas='aal78',
                                  bank_dir=self.bank, **kwargs)

    def test_cache_miss_hit_and_parameter_seed_changes(self):
        first = self.resolve()
        second = self.resolve(config=replace(self.config, sample_id='different_label'))
        self.assertEqual(first[2], second[2])
        self.assertEqual(self.generate.call_count, 1)
        self.resolve(config=replace(self.config, jr_noise_scale=.2))
        self.resolve(config=replace(self.config, random_seed=1))
        self.assertEqual(self.generate.call_count, 3)
        self.assertEqual(len(read_bank(self.bank)['samples']), 3)

    def test_integer_and_float_physics_share_cache_and_provenance(self):
        integer = replace(self.config, simulation_length_ms=10, warmup_ms=10,
                          output_srate_hz=500, discard_ms=0, coupling_strength=1,
                          jr_noise_scale=1, jr_background_A=3, jr_active_A=4, jr_mu=0)
        decimal = replace(integer, simulation_length_ms=10., warmup_ms=10.,
                          output_srate_hz=500., discard_ms=0., coupling_strength=1.,
                          jr_noise_scale=1., jr_background_A=3., jr_active_A=4., jr_mu=0.)
        first = self.resolve(config=integer)[2]
        second = self.resolve(config=decimal)[2]
        self.assertEqual(first['sample_id'], second['sample_id'])
        self.assertEqual(first['simulation_group_id'], second['simulation_group_id'])
        self.assertEqual(first['provenance'], second['provenance'])
        self.assertEqual(self.generate.call_count, 1)
        canonical = self.generate.call_args.args[0]
        self.assertIs(type(canonical.coupling_strength), float)
        self.assertIs(type(canonical.output_srate_hz), float)
        self.assertIs(type(canonical.random_seed), int)
        self.assertIs(type(canonical.active_regions[0]), int)
        self.assertIs(canonical.normalize, False)

    def test_invalid_seed_and_region_types_are_not_coerced(self):
        for invalid in (replace(self.config, random_seed=True),
                        replace(self.config, random_seed=2.0),
                        replace(self.config, active_regions=[0.0]),
                        replace(self.config, active_regions=[True])):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                self.resolve(config=invalid)
        self.generate.assert_not_called()

    def test_online_always_runs_and_has_same_physical_group_as_cache(self):
        cached = self.resolve()
        first = self.resolve('online')
        second = self.resolve('online')
        self.assertEqual(self.generate.call_count, 3)
        np.testing.assert_array_equal(cached[0], first[0])
        self.assertEqual(cached[2]['simulation_group_id'], first[2]['simulation_group_id'])
        self.assertEqual(first[2]['simulation_group_id'], second[2]['simulation_group_id'])
        self.assertNotIn('source_file', first[2])
        self.assertNotIn('source_sha256', first[2])
        self.assertEqual(first[2]['source_unit'], 'mV_PSP')

    def test_views_have_distinct_cache_ids_but_one_split_group(self):
        first = self.resolve()[2]
        changed = self.resolve(config=replace(self.config, warmup_ms=20))[2]
        self.assertNotEqual(first['sample_id'], changed['sample_id'])
        self.assertEqual(first['simulation_group_id'], changed['simulation_group_id'])

    def test_tvb_version_changes_create_new_cache_records(self):
        first = self.resolve()[2]
        with patch('Simulation_G3.nmm_sources.version', return_value='different-version'), \
             patch('Simulation_G3.generate_nmm_bank.version', return_value='different-version'):
            changed = self.resolve()[2]
        self.assertNotEqual(first['sample_id'], changed['sample_id'])
        self.assertEqual(changed['provenance']['tvb_version'], 'different-version')
        self.assertEqual(len(read_bank(self.bank)['samples']), 2)

    def test_bank_replay_does_not_construct_generator_or_use_recipe(self):
        source, times, entry = self.resolve()
        with patch('Simulation_G3.generate_nmm_bank.TVBSourceGenerator.__init__',
                   side_effect=AssertionError('replay constructed TVB generator')):
            replay = self.resolve('bank', config=replace(self.config, normalize=True),
                                  sample_id=entry['sample_id'])
            explicit_cache = self.resolve('cache', sample_id=entry['sample_id'])
        np.testing.assert_array_equal(source, replay[0])
        np.testing.assert_array_equal(times, explicit_cache[1])

    def test_corrupt_record_is_not_regenerated_or_overwritten(self):
        entry = self.resolve()[2]
        target = self.bank / entry['source_file']
        target.write_bytes(b'damaged')
        for provider in ('cache', 'bank'):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.resolve(provider, **({'sample_id': entry['sample_id']} if provider == 'bank' else {}))
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(target.read_bytes(), b'damaged')

    def test_wrong_atlas_header_rejected_on_replay(self):
        entry = self.resolve()[2]
        path = self.bank / 'bank.json'
        bank = json.loads(path.read_text())
        bank['projection_sha256'] = '0' * 64
        path.write_text(json.dumps(bank))
        with self.assertRaisesRegex(ValueError, 'atlas/projection'):
            self.resolve('bank', sample_id=entry['sample_id'])

    def test_invalid_generation_and_provider_modes_fail_before_running(self):
        for provider, config, kwargs in [
            ('missing', self.config, {}), ('bank', self.config, {}),
            ('online', self.config, {'sample_id': 'explicit'}),
            ('cache', replace(self.config, normalize=True), {}),
            ('cache', replace(self.config, reshape_to_994=True), {}),
        ]:
            with self.subTest(provider=provider, config=config):
                with self.assertRaises(ValueError):
                    self.resolve(provider, config=config, **kwargs)
        self.generate.assert_not_called()

    def test_presets_separate_physics_and_validation_claims(self):
        self.assertEqual(PRESETS['rest_control']['parameters']['signal_type'], 'resting_jr')
        self.assertEqual(PRESETS['rest_control']['parameters']['jr_active_A'], 3.25)
        self.assertEqual(PRESETS['spike_origin']['parameters']['jr_active_A'], 3.6)
        self.assertIn('candidate', PRESETS['alpha_candidate']['validation_status'])


if __name__ == '__main__':
    unittest.main()
