"""Primary training samples contain named arrays and complete source truth."""
from dataclasses import replace
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator


class ProductionSampleTests(unittest.TestCase):
    def setUp(self):
        self.sim=grid_simulator(10)
        self.config=G1SimulationConfig(srate=128,time_window=(0,1),patch_area_mm2=10,
            random_seed=19,sensor_noise_type='iid_white',sensor_snr_db=10)

    def test_primary_arrays_manifest_and_foreground_supervision(self):
        for background in (None,'pink'):
            config=replace(self.config,source_background_type=background,
                           source_snr_db=5 if background else None)
            sample=self.sim.generate_sample(config,compute_diagnostics=False)
            for key in ('sensor','clean_sensor','sensor_noise','source_foreground',
                        'source_background','source_total','foreground_sensor','background_sensor',
                        'baseline_sensor','noise_covariance'):
                self.assertIsInstance(sample[key],np.ndarray)
                self.assertNotIn(key,sample['manifest'])
            json.dumps(sample['manifest'],allow_nan=False)
            self.assertIs(sample['source'],sample['source_total'])
            np.testing.assert_allclose(sample['source_foreground']+sample['source_background'],
                                       sample['source_total'],atol=2e-7)
            np.testing.assert_allclose(sample['foreground_sensor']+sample['background_sensor'],
                                       sample['clean_sensor'],atol=2e-7)
            np.testing.assert_allclose(sample['noise_covariance'],np.cov(sample['baseline_sensor']),atol=1e-12)
            self.assertEqual(sample['manifest']['supervision_target'],'source_foreground')

    def test_plain_patch_component_request_is_never_silently_empty(self):
        for background in (None,'pink'):
            parts={}
            config=replace(self.config,source_background_type=background,
                           source_snr_db=5 if background else None)
            sample=self.sim.generate_sample(config,source_components=parts)
            self.assertIn('foreground_coarse',parts)
            np.testing.assert_array_equal(parts['foreground_residual'],0)
            np.testing.assert_allclose(parts['foreground_coarse'],sample['source_foreground'],atol=1e-7)
            if background:
                np.testing.assert_allclose(parts['background_coarse'],sample['source_background'],atol=1e-7)

    def test_selected_snr_is_recorded_separately_from_global_and_effective(self):
        sample=self.sim.generate_sample(self.config)
        meta=sample['manifest']
        self.assertEqual(meta['sensor_snr_mode'],'active_rms')
        self.assertAlmostEqual(meta['realized_sensor_snr_reference_db'],10,places=5)
        self.assertLess(meta['realized_sensor_snr_db'],10)
        self.assertIn('effective_target_snr_db',meta)
        self.assertIn('effective_target_event_snr_db',meta)
        self.assertIn('source_temporal_representation',meta)

    def test_atomic_save_complete_patch_truth_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            saved=self.sim.save_sample(directory,self.config,save_source=True,save_components=True)
            folder=Path(directory)/self.config.sample_id
            self.assertTrue((folder/'commit.json').is_file())
            with np.load(folder/saved['source_components_file']) as data:
                np.testing.assert_array_equal(data['foreground_residual'],0)
                np.testing.assert_allclose(data['foreground_coarse'],data['source_foreground'])
            with patch.object(self.sim,'generate_sample',side_effect=AssertionError('regenerated')):
                repeated=self.sim.save_sample(directory,self.config,save_source=True,save_components=True,resume=True)
                self.assertEqual(saved,repeated)
                with self.assertRaises(FileExistsError):
                    self.sim.save_sample(directory,self.config,save_source=True,save_components=True)
                with self.assertRaisesRegex(ValueError,'identity'):
                    self.sim.save_sample(directory,replace(self.config,random_seed=20),
                                         save_source=True,save_components=True,resume=True)
            with (folder/saved['standard_data']).open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.sim.save_sample(directory,self.config,save_source=True,save_components=True,resume=True)

    def test_reused_support_distances_are_cached_and_mask_change_invalidates(self):
        seeds,supports,_=self.sim._resolve_spatial_patches(self.config,np.random.default_rng(9))
        config=replace(self.config,seed_vertices=seeds.tolist(),patch_vertices=[v.tolist() for v in supports])
        with patch('Simulation_G1.geodesic.geodesic_distances',wraps=__import__(
                'Simulation_G1.geodesic',fromlist=['geodesic_distances']).geodesic_distances) as solve:
            self.sim.generate_sample(config,compute_diagnostics=False)
            first=solve.call_count
            self.sim.generate_sample(replace(config,random_seed=22),compute_diagnostics=False)
            self.assertEqual(solve.call_count,first)
            old_graph=self.sim.geodesic_graph
            outside=next(i for i in range(self.sim.n_sources) if i not in supports[0])
            self.sim.cortex_mask[outside]=False
            self.assertIsNot(old_graph,self.sim.geodesic_graph)


if __name__=='__main__':
    unittest.main()
