"""Production NMM SNR calibration, replay semantics and target aliases."""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1.g1_simulation_core import G1SourceSimulator
from Simulation_G2.nmm_mixing import make_sensors
from Simulation_G2.training_dataset import NMMRecipeDataset
from nmm_simulation import NMMSimulationConfig
import test_unified_nmm as unified_fixture
import test_nmm_training_dataset as period_fixture
import test_registered_nmm_noise as registered_fixture


class ProductionSensorSNRTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(G1SourceSimulator)
        self.backend.leadfield = np.eye(3)
        time = np.arange(128) / 128.
        pulse = np.exp(-.5 * ((time - .5) / .025) ** 2)
        foreground = np.array([pulse, -pulse * .5, pulse * .2])
        self.sources = dict(source_foreground=foreground, source_background=np.zeros_like(foreground))
        self.options = dict(srate=128., sensor_snr_db=10., sensor_seed=7, average_reference=False)

    def test_active_reference_uses_global_event_noise_denominator(self):
        arrays, metadata = make_sensors(self.backend, self.sources, **self.options)
        self.assertEqual(metadata['sensor_snr_mode'], 'active_rms')
        clean, noise = arrays['clean_sensor'], arrays['sensor_noise']
        mask = np.sqrt(np.mean(clean ** 2, axis=0)) >= np.quantile(np.sqrt(np.mean(clean ** 2, axis=0)), .95)
        ratio = 10 * np.log10(np.mean(clean[:, mask] ** 2) / np.mean(noise ** 2))
        self.assertAlmostEqual(ratio, 10., places=11)
        self.assertAlmostEqual(metadata['realized_sensor_snr_reference_db'], ratio)
        self.assertAlmostEqual(metadata['realized_sensor_snr_db'],
                               10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2)))
        self.assertAlmostEqual(metadata['effective_target_active_snr_db'], ratio)
        self.assertAlmostEqual(metadata['effective_target_active_window_snr_db'],
                               10 * np.log10(np.mean(clean[:, mask] ** 2) / np.mean(noise[:, mask] ** 2)))

    def test_global_robust_and_invalid_calibration_controls(self):
        for mode in ('global_rms', 'robust_global_rms', 'active_rms'):
            arrays, meta = make_sensors(self.backend, self.sources, sensor_snr_mode=mode, **self.options)
            reference = self.backend._snr_reference_power(arrays['clean_sensor'], mode, .95)
            self.assertAlmostEqual(10*np.log10(reference/np.mean(arrays['sensor_noise'] ** 2)), 10., places=11)
            self.assertAlmostEqual(meta['realized_sensor_snr_reference_db'], 10., places=11)
        for settings in ({'sensor_snr_mode': 'bad'}, {'sensor_snr_clip_quantile': .1},
                         {'sensor_snr_clip_quantile': np.nan}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_sensors(self.backend, self.sources, **self.options, **settings)

    def test_physical_fixed_noise_does_not_change_with_selected_snr_mode(self):
        options = dict(self.options, sensor_snr_db=None, amplitude_mode='physical',
                       sensor_units='V', sensor_noise_std=1e-6)
        first, _ = make_sensors(self.backend, self.sources, sensor_snr_mode='active_rms', **options)
        second, _ = make_sensors(self.backend, self.sources, sensor_snr_mode='global_rms', **options)
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
        np.testing.assert_array_equal(first['baseline_sensor'], second['baseline_sensor'])

    def test_target_metrics_include_source_background_as_nuisance(self):
        background = .1 * np.vstack([np.cos(np.arange(128) * frequency) for frequency in (.11, .17, .23)])
        sources = dict(self.sources, source_background=background)
        arrays, meta = make_sensors(self.backend, sources, baseline_source=np.tile(background, (1, 2)), **self.options)
        target = arrays['foreground_sensor']
        nuisance = arrays['background_sensor'] + arrays['sensor_noise']
        gfp = np.sqrt(np.mean(target ** 2, axis=0))
        active = gfp >= np.quantile(gfp, .95)
        self.assertAlmostEqual(meta['effective_target_global_snr_db'],
                               10 * np.log10(np.mean(target ** 2) / np.mean(nuisance ** 2)))
        self.assertAlmostEqual(meta['effective_target_active_snr_db'],
                               10 * np.log10(np.mean(target[:, active] ** 2) / np.mean(nuisance ** 2)))


class ProductionUnifiedSNRTests(unittest.TestCase):
    setUp = unified_fixture.UnifiedNMMTests.setUp
    fake_trajectory = staticmethod(unified_fixture.UnifiedNMMTests.fake_trajectory)

    def test_alias_and_invalid_mode_before_trajectory_generation(self):
        self.assertEqual(NMMSimulationConfig().sensor_snr_mode, 'active_rms')
        sample = self.sim.generate_sample(self.config)
        self.assertIs(sample['source_total'], sample['source'])
        self.assertIn('source_foreground', sample['manifest']['target_definition'])
        self.resolver.reset_mock()
        with self.assertRaises(ValueError):
            self.sim.generate_sample(replace(self.config, sensor_snr_mode='invalid'))
        self.resolver.assert_not_called()

    def test_transactional_save_verified_resume_and_corruption(self):
        output = self.root / 'transactional'
        first = self.sim.save_sample(output, self.config)
        folder = output / self.config.sample_id
        self.assertTrue((folder / 'commit.json').is_file())
        self.assertIn('source_total', first['arrays'])
        self.assertIn('source_total', first['array_shapes'])
        self.resolver.reset_mock()
        with self.assertRaises(FileExistsError):
            self.sim.save_sample(output, self.config)
        self.resolver.assert_not_called()
        repeated = self.sim.save_sample(output, self.config, resume=True)
        self.assertEqual(first, repeated)
        self.assertTrue(self.resolver.called)  # Explicitly post-generation/replay resume.
        before = (folder / 'commit.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.sim.save_sample(output, replace(self.config, sensor_seed=123), resume=True)
        self.assertEqual((folder / 'commit.json').read_bytes(), before)
        payload = folder / first['data_file']
        payload.write_bytes(payload.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.sim.save_sample(output, self.config, resume=True)

    def test_supplied_noise_reader_reuses_validated_record(self):
        import Simulation_G1.real_noise as real_noise
        path = registered_fixture.registry(self.root, ['a', 'b', 'c'])
        config = replace(self.config, source_snr_db=None, sensor_noise_type='real',
                         real_noise_registry=str(path), real_noise_record_id='record_train')
        self.sim.noise_reader = real_noise.NoiseWindowReader()
        with patch.object(real_noise, '_validated_record_array', wraps=real_noise._validated_record_array) as read:
            first = self.sim.generate_sample(config)
            second = self.sim.generate_sample(config)
            self.assertEqual(read.call_count, 1)
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
        self.assertGreater(self.sim.noise_reader.cache_bytes, 0)

    def test_resume_ignores_unrelated_bank_append_but_binds_selected_file(self):
        bank = self.root / 'cache'
        bank.mkdir()
        index = bank / 'bank.json'
        index.write_text(json.dumps({'samples': {}}))
        config = replace(self.config, bank_dir=str(bank), source_snr_db=None)
        trajectory = bank / 'selected.npz'
        np.savez(trajectory, fixture_signal=np.arange(10.))
        def resolve(simulation_config, **kwargs):
            source, times, entry = self.fake_trajectory(simulation_config, **kwargs)
            entry.update(source_file=trajectory.name,
                         source_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest())
            return source, times, entry
        self.resolver.side_effect = resolve
        output = self.root / 'append_safe'
        first = self.sim.save_sample(output, config)
        index.write_text(json.dumps({'samples': {'unrelated_sample': {'source_file': 'unrelated.npz'}}}))
        np.savez(bank / 'unrelated.npz', fixture_signal=np.arange(20.))
        self.assertEqual(first, self.sim.save_sample(output, config, resume=True))
        np.savez(trajectory, fixture_signal=np.arange(10.) + 1.)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.sim.save_sample(output, config, resume=True)

    def test_cli_resume_flag_forwards_explicit_resume(self):
        from dataclasses import asdict
        import nmm_simulation
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(asdict(self.config)))
        output = self.root / 'cli'
        (output / self.config.sample_id).mkdir(parents=True)
        with patch('nmm_simulation.NMMSourceSimulator') as constructor:
            constructor.return_value.save_sample.return_value = {'sample_id': self.config.sample_id}
            with self.assertRaises(FileExistsError):
                nmm_simulation.main(['--config', str(config_path), '--output-dir', str(output)])
            constructor.assert_not_called()
            nmm_simulation.main(['--config', str(config_path), '--output-dir', str(output), '--resume'])
            self.assertTrue(constructor.return_value.save_sample.call_args.kwargs['resume'])


class ProductionPeriodSNRTests(unittest.TestCase):
    setUp = period_fixture.NMMTrainingDatasetTests.setUp
    save = period_fixture.NMMTrainingDatasetTests.save
    build = period_fixture.NMMTrainingDatasetTests.build

    def test_v5_declares_snr_and_legacy_absence_means_global(self):
        manifest = self.build(sensor_snrs=(10.,))
        self.assertEqual(manifest['schema'], 'nmm-periods-v5')
        dataset = NMMRecipeDataset(self.root / 'index', self.backend)
        sample = dataset[0]
        self.assertIs(sample['source_total'], sample['source'])
        self.assertEqual(sample['manifest']['sensor_snr_mode'], 'active_rms')
        self.assertAlmostEqual(sample['manifest']['realized_sensor_snr_reference_db'], 10., places=5)
        row = dataset.recipe(0)
        row['config'].pop('sensor_snr_mode')
        row['config'].pop('sensor_snr_clip_quantile')
        with patch.object(dataset, 'recipe', return_value=row), self.assertRaisesRegex(ValueError, 'SNR'):
            dataset[0]
        for schema in ('nmm-periods-v1', 'nmm-periods-v2', 'nmm-periods-v3', 'nmm-periods-v4'):
            dataset.manifest['schema'] = schema
            if schema == 'nmm-periods-v4':
                continue  # Registry-backed v4 compatibility has its own integration fixture.
            with self.subTest(schema=schema), patch.object(dataset, 'recipe', return_value=row):
                legacy = dataset[0]
                self.assertEqual(legacy['manifest']['sensor_snr_mode'], 'global_rms')
                self.assertAlmostEqual(legacy['manifest']['realized_sensor_snr_db'], 10., places=5)

    def test_registry_backed_v4_without_mode_replays_global(self):
        names = [f'channel_{index}' for index in range(self.backend.n_sensors)]
        path = registered_fixture.registry(self.root, names)
        self.build(real_noise_registry=path, sensor_noise_types=('real',))
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=names)
        row = dataset.recipe(0)
        row['config'].pop('sensor_snr_mode')
        row['config'].pop('sensor_snr_clip_quantile')
        dataset.manifest['schema'] = 'nmm-periods-v4'
        with patch.object(dataset, 'recipe', return_value=row):
            sample = dataset[0]
        self.assertEqual(sample['manifest']['sensor_snr_mode'], 'global_rms')
        self.assertAlmostEqual(sample['manifest']['realized_sensor_snr_db'], 10., places=5)
        self.assertEqual(sample['manifest']['real_noise_provenance']['identity_status'], 'registered_verified')

    def test_default_period_reader_caches_registered_recording(self):
        import Simulation_G1.real_noise as real_noise
        names = [f'channel_{index}' for index in range(self.backend.n_sensors)]
        path = registered_fixture.registry(self.root, names)
        self.build(real_noise_registry=path, sensor_noise_types=('real',))
        dataset = NMMRecipeDataset(self.root / 'index', self.backend, channel_names=names)
        with patch.object(real_noise, '_validated_record_array', wraps=real_noise._validated_record_array) as read:
            first = dataset[0]
            second = dataset[0]
            self.assertEqual(read.call_count, 1)
        self.assertGreater(dataset.noise_reader.cache_bytes, 0)
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
