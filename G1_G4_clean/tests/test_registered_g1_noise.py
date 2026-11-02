"""Registered nuisance has a verified sensor layout and a replayable baseline."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np

from Simulation_G1.g1_simulation_core import G1SimulationConfig
from Simulation_G1.real_noise import create_noise_registry
from test_spatial_patches import grid_simulator


class RegisteredG1NoiseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sim = grid_simulator(10)
        self.sim.channel_names = [f'CH{i}' for i in range(self.sim.n_sensors)]
        data = np.random.default_rng(8).normal(size=(self.sim.n_sensors, 1000))
        data -= data.mean(axis=0)
        np.savez(self.root/'noise.npz', noise=data)
        create_noise_registry([dict(path=str(self.root/'noise.npz'), record_id='rest',
            recording_id='session1', subject_id='S1', split='train', fs_hz=100.,
            channel_names=self.sim.channel_names, sensor_units='V', reference='average',
            noise_role='resting_background')], self.root/'registry.json')
        self.config = G1SimulationConfig(srate=100, time_window=(0, 1),
            n_patch=1, patch_area_mm2=10, random_seed=17, sensor_noise_type='real',
            real_noise_registry=str(self.root/'registry.json'), real_noise_record_id='rest',
            real_noise_split='train')

    def test_replay_covariance_and_ancestry(self):
        first = self.sim.generate_sensor(self.config)
        second = self.sim.generate_sensor(self.config)
        for a,b in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(first[4],second[4])
        meta = first[4]
        info = meta['real_noise_metadata']
        self.assertEqual(info['baseline_stop_sample'], info['event_start_sample'])
        self.assertEqual(info['baseline_stop_sample']-info['baseline_start_sample'],200)
        self.assertEqual(meta['parent_data_group_ids'],info['parent_data_group_ids'])
        np.testing.assert_allclose(meta['noise_covariance'], np.cov(meta['baseline_sensor']),atol=1e-12)

    def test_reject_wrong_split_ambiguous_input_and_combined_background(self):
        for config in (replace(self.config,real_noise_split='test'),
                       replace(self.config,sensor_noise_type='white'),
                       replace(self.config,source_background_type='pink',source_snr_db=10)):
            with self.assertRaises(ValueError):
                self.sim.generate_sensor(config)
        with self.assertRaises(ValueError):
            self.sim.generate_sensor(self.config,real_sensor_noise=np.ones((self.sim.n_sensors,300)))
        combined = replace(self.config,source_background_type='pink',source_snr_db=10,
                           allow_combined_real_source_background=True)
        self.assertTrue(self.sim.generate_sensor(combined)[4]['real_noise_metadata']['combined_source_background'])
        self.sim.channel_names=None
        with self.assertRaises(ValueError):
            self.sim.generate_sensor(self.config)

    def test_raw_array_remains_supported_and_marked_unverified(self):
        config=replace(self.config,real_noise_registry=None,real_noise_record_id=None,real_noise_split=None)
        result=self.sim.generate_sensor(config,real_sensor_noise=np.random.default_rng(2).normal(size=(self.sim.n_sensors,300)))
        self.assertEqual(result[4]['real_noise_metadata']['validation'],'unverified_array_metadata')

    def test_relative_mixed_units_and_disabled_noise_are_rejected(self):
        import json
        record=json.loads((self.root/'registry.json').read_text())['records'][0]
        record['sensor_units']=['V']*self.sim.n_sensors
        record['sensor_units'][0]='uV'
        create_noise_registry([record],self.root/'mixed.json')
        with self.assertRaisesRegex(ValueError,'common sensor units'):
            self.sim.generate_sensor(replace(self.config,real_noise_registry=str(self.root/'mixed.json')))
        with self.assertRaisesRegex(ValueError,'requires sensor_snr_db'):
            self.sim.generate_sensor(replace(self.config,sensor_snr_db=None))

    def test_registered_physical_noise_cannot_be_silently_disabled(self):
        config=replace(self.config,amplitude_mode='physical',normalize_source=False,
            sensor_snr_db=None,foreground_peak_am=1e-9,leadfield_source_unit='A*m',sensor_units='V')
        for value in (None,0,[0]*self.sim.n_sensors):
            with self.assertRaisesRegex(ValueError,'positive sensor_noise_std'):
                self.sim.generate_sensor(replace(config,sensor_noise_std=value))
        result=self.sim.generate_sensor(replace(config,sensor_noise_std=1e-6))
        self.assertGreater(np.linalg.norm(result[2]),0)

    def test_resume_verifies_registered_recording_content_before_reusing_output(self):
        self.sim.save_sample(self.root/'samples',self.config,save_source=True)
        np.savez(self.root/'noise.npz',noise=np.zeros((self.sim.n_sensors,1000)))
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.sim.save_sample(self.root/'samples',self.config,save_source=True,resume=True)


if __name__=='__main__':
    unittest.main()
