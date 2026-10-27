"""Whitening covariance follows the realized sensor nuisance and survives export."""
import json
import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np

from Simulation_G2.nmm_mixing import make_sensors, project_baseline
from Simulation_G2.training_dataset import NMMRecipeDataset
import test_nmm_mixing
import test_nmm_training_dataset
import test_unified_nmm
import nmm_simulation


class NMMNoiseCovarianceTests(unittest.TestCase):
    def test_default_two_second_unified_baseline(self):
        fixture = self.fixture(test_unified_nmm.UnifiedNMMTests)
        config = vars(fixture.config).copy()
        config.pop('baseline_duration_s')
        sample = fixture.sim.generate_sample(nmm_simulation.NMMSimulationConfig(**config))
        self.assertEqual(sample['baseline_sensor'].shape, (3, 200))
        self.assertEqual(sample['sensor'].shape, (3, 100))
        self.assert_covariance(sample, tolerance=2e-7)

    def fixture(self, case):
        fixture = case()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        return fixture

    def assert_covariance(self, arrays, *, tolerance=1e-12):
        nuisance = arrays['background_sensor'].astype(float) + arrays['sensor_noise'].astype(float)
        for key, values in (('noise_covariance', arrays['baseline_sensor'].astype(float)),
                            ('oracle_noise_covariance', nuisance),
                            ('sensor_noise_covariance', arrays['sensor_noise'])):
            self.assertEqual(arrays[key].dtype, np.dtype('float64'))
            np.testing.assert_allclose(arrays[key], np.cov(values), atol=tolerance, rtol=tolerance)

    def test_actual_correlated_background_and_measurement_noise_include_cross_terms(self):
        fixture = self.fixture(test_nmm_mixing.NMMMixingTests)
        sources, _ = fixture.mix()
        baseline, _ = fixture.sensors(sources, srate=500)
        # A separate baseline segment followed by event noise correlated with event background.
        stream=np.concatenate([np.random.default_rng(801).normal(size=(3,500)),baseline['background_sensor']],axis=1)
        with patch.object(fixture.backend, '_make_sensor_noise', return_value=stream):
            arrays, metadata = fixture.sensors(sources, srate=500, sensor_snr_db=3)
        self.assert_covariance(arrays)
        independent_sum = np.cov(arrays['background_sensor']) + arrays['sensor_noise_covariance']
        self.assertFalse(np.allclose(arrays['oracle_noise_covariance'], independent_sum))
        self.assertFalse(np.allclose(arrays['noise_covariance'], arrays['oracle_noise_covariance']))
        for key in ('noise_covariance_info', 'sensor_noise_covariance_info'):
            self.assertIn(key, metadata)
        np.testing.assert_allclose(arrays['noise_covariance'] @ np.ones(3), 0, atol=1e-7)

    def test_background_only_and_fully_noiseless_samples(self):
        fixture = self.fixture(test_nmm_mixing.NMMMixingTests)
        sources, _ = fixture.mix()
        for reference in (False, True):
            arrays, _ = fixture.sensors(sources, srate=500, average_reference=reference)
            self.assert_covariance(arrays)
            self.assertGreater(np.trace(arrays['noise_covariance']), 0)
            np.testing.assert_array_equal(arrays['sensor_noise_covariance'], np.zeros((3, 3)))
        sources['source_background'] = np.zeros_like(sources['source_background'])
        arrays, _ = make_sensors(fixture.backend, sources, srate=500)
        for key in ('noise_covariance', 'sensor_noise_covariance'):
            np.testing.assert_array_equal(arrays[key], np.zeros((3, 3)))

    def test_unified_sample_five_return_and_npz_export_keep_covariance(self):
        fixture = self.fixture(test_unified_nmm.UnifiedNMMTests)
        sample = fixture.sim.generate_sample(fixture.config)
        self.assert_covariance(sample, tolerance=2e-7)
        self.assertEqual(sample['sensor'].dtype, np.dtype('float32'))
        result = fixture.sim.generate_sensor(fixture.config)
        self.assertEqual(len(result), 5)
        metadata = result[-1]
        json.dumps(metadata, allow_nan=False)
        saved = fixture.sim.save_sample(fixture.root / 'export', fixture.config)
        path = fixture.root / 'export' / fixture.config.sample_id / saved['data_file']
        with np.load(path, allow_pickle=False) as archive:
            for key in ('noise_covariance', 'oracle_noise_covariance', 'sensor_noise_covariance', 'baseline_sensor'):
                np.testing.assert_array_equal(metadata[key], sample[key])
                np.testing.assert_array_equal(archive[key], sample[key])
                self.assertEqual(archive[key].dtype, np.dtype('float32' if key=='baseline_sensor' else 'float64'))

    def test_period_batches_include_total_covariance_by_default(self):
        fixture = self.fixture(test_nmm_training_dataset.NMMTrainingDatasetTests)
        fixture.build(source_mode='partition', source_snr_db=10, sensor_snrs=(5,))
        dataset = NMMRecipeDataset(fixture.root / 'index', fixture.backend)
        sample = dataset[0]
        self.assert_covariance(sample, tolerance=2e-7)
        channels = fixture.backend.leadfield.shape[0]
        batch = dataset.get_batch([0, 1], workers=2)
        self.assertEqual(set(batch), {'sensor', 'source_foreground', 'noise_covariance', 'manifests'})
        self.assertEqual(batch['noise_covariance'].shape, (2, channels, channels))
        self.assertEqual(batch['noise_covariance'].dtype, np.dtype('float64'))
        np.testing.assert_array_equal(batch['noise_covariance'][0], sample['noise_covariance'])
        prefetched = next(dataset.iter_batches(2, prefetch=1))
        np.testing.assert_array_equal(prefetched['noise_covariance'], batch['noise_covariance'])
        explicit = dataset.get_batch([0], fields=('noise_covariance', 'sensor_noise_covariance'))
        np.testing.assert_array_equal(explicit['sensor_noise_covariance'][0], sample['sensor_noise_covariance'])

    def test_period_baseline_same_parent_nonoverlap_and_shared_source_gain(self):
        fixture = self.fixture(test_nmm_training_dataset.NMMTrainingDatasetTests)
        fixture.build(source_mode='partition',source_snr_db=8,sensor_snrs=(None,))
        dataset=NMMRecipeDataset(fixture.root/'index',fixture.backend)
        for index in range(len(dataset)):
            sample=dataset[index]; row=dataset.recipe(index)
            baseline_ref=row['baseline_background']; event_ref=row['foreground']
            self.assertEqual(baseline_ref['simulation_group_id'],event_ref['simulation_group_id'])
            self.assertTrue(baseline_ref['stop_sample']<=event_ref['start_sample'] or
                            baseline_ref['start_sample']>=event_ref['stop_sample'])
            region,_=dataset._window(baseline_ref,100)
            region=region.copy();region[row['config']['selected_regions']]=0
            expected=fixture.backend.leadfield @ project_baseline(region,fixture.backend.projection_matrix,sample['manifest'])
            expected-=expected.mean(axis=1,keepdims=True)
            expected-=expected.mean(axis=0,keepdims=True)
            np.testing.assert_allclose(sample['baseline_sensor'],expected,atol=1e-7)
            self.assertEqual(sample['manifest']['baseline']['split'],row['split'])

    def test_background_without_distinct_baseline_is_rejected(self):
        fixture=self.fixture(test_nmm_mixing.NMMMixingTests)
        sources,_=fixture.mix()
        with self.assertRaisesRegex(ValueError,'nonoverlapping baseline_source'):
            make_sensors(fixture.backend,sources,srate=500)

    def test_unified_bank_rejects_missing_baseline_samples(self):
        fixture=self.fixture(test_unified_nmm.UnifiedNMMTests)
        def truncated(config,**kwargs):
            values,times,entry=fixture.fake_trajectory(config,**kwargs)
            return values[:,:100],times[:100],entry
        fixture.resolver.side_effect=truncated
        with self.assertRaisesRegex(ValueError,'nonoverlapping baseline'):
            fixture.sim.generate_sample(fixture.config)

    def test_measurement_baseline_uses_shared_event_gain_without_power_matching(self):
        fixture=self.fixture(test_nmm_mixing.NMMMixingTests)
        sources,_=fixture.mix();sources['source_background']*=0
        rng=np.random.default_rng(901)
        baseline=4*rng.normal(size=(3,500));event=rng.normal(size=(3,200))
        with patch.object(fixture.backend,'_make_sensor_noise',return_value=np.concatenate([baseline,event],axis=1)):
            arrays,meta=make_sensors(fixture.backend,sources,srate=500,sensor_snr_db=10,average_reference=False,baseline_duration_s=1.)
        gain=meta['baseline']['shared_noise_gain']
        np.testing.assert_allclose(arrays['baseline_sensor'],(baseline-baseline.mean(axis=1,keepdims=True))*gain,rtol=1e-6)
        np.testing.assert_allclose(arrays['sensor_noise'],(event-event.mean(axis=1,keepdims=True))*gain,rtol=1e-12)
        self.assertGreater(np.trace(arrays['noise_covariance']),8*np.trace(arrays['sensor_noise_covariance']))

    def test_external_background_baseline_retains_same_parent_and_custom_length(self):
        fixture=self.fixture(test_nmm_training_dataset.NMMTrainingDatasetTests)
        bank=fixture.root/'bg'
        test_nmm_training_dataset.initialize_bank(bank,atlas_id='test',n_regions=4,
            projection_sha256=fixture.projection_hash,schema_version=2)
        fixture.save(bank,'rest',seed=2,rest=True)
        fixture.build(source_mode='selected',background_bank_dirs=[bank],source_snr_db=10)
        dataset=NMMRecipeDataset(fixture.root/'index',fixture.backend)
        row=dataset.recipe(0); sample=dataset[0]
        self.assertEqual(row['baseline_background']['simulation_group_id'],row['background']['simulation_group_id'])
        self.assertEqual(row['baseline_background']['start_sample'],row['background']['stop_sample'])
        self.assertEqual(sample['manifest']['baseline']['split'],row['split'])
        fixture.build(folder='short_baseline',source_mode='partition',source_snr_db=8,
                      sensor_snrs=(None,),baseline_duration_s=.5)
        custom=NMMRecipeDataset(fixture.root/'short_baseline',fixture.backend)[0]
        self.assertEqual(custom['baseline_sensor'].shape[1],50)
        self.assert_covariance(custom,tolerance=2e-7)

    def test_period_index_rejects_record_without_disjoint_baseline(self):
        fixture=self.fixture(test_nmm_training_dataset.NMMTrainingDatasetTests)
        with self.assertRaisesRegex(ValueError,'nonoverlapping baseline'):
            fixture.build(source_mode='partition',source_snr_db=8,period_s=2.)
        self.assertFalse((fixture.root/'index').exists())

    def test_unified_baseline_uses_native_bank_clock_before_resampling(self):
        fixture=self.fixture(test_unified_nmm.UnifiedNMMTests)
        def native_trajectory(config,**kwargs):
            return fixture.fake_trajectory(replace(config,output_srate_hz=100),**kwargs)
        fixture.resolver.side_effect=native_trajectory
        result=fixture.sim.generate_sample(replace(fixture.config,srate=256))
        self.assertEqual(result['baseline_sensor'].shape,(3,256))
        baseline=result['manifest']['baseline']
        self.assertEqual(baseline['background_native_srate_hz'],100)
        self.assertEqual(baseline['background_start_sample'],100)
        self.assertEqual(baseline['background_stop_sample'],200)

    def test_cli_baseline_override_and_rest_trajectory_extension(self):
        fixture=self.fixture(test_unified_nmm.UnifiedNMMTests)
        path=fixture.root/'recipes.json'
        path.write_text(json.dumps({'duration_s':1.,'srate':100.,'baseline_duration_s':.25}))
        with patch('nmm_simulation.NMMSourceSimulator') as simulator, patch('builtins.print'):
            simulator.return_value.save_sample.return_value={}
            self.assertEqual(nmm_simulation.main(['--config',str(path),'--output-dir',str(fixture.root/'out'),
                                                  '--baseline-duration-s','.5']),0)
            config=simulator.return_value.save_sample.call_args.args[1]
            self.assertEqual(config.baseline_duration_s,.5)
        config=replace(fixture.config,trajectory_duration_s=1.,baseline_duration_s=.5)
        trajectory=fixture.sim._simulation_config(config,fixture.profile,background=True)
        self.assertEqual(trajectory.simulation_length_ms,1500.)


if __name__ == '__main__':
    unittest.main()
