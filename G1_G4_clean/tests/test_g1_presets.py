"""Preset isolation, explicit overrides and external asset paths."""
from pathlib import Path
import tempfile
import unittest

from Simulation_G1.presets import make_g1_config


class PresetTests(unittest.TestCase):
    def test_calibrated_parameters_and_assets(self):
        with tempfile.TemporaryDirectory() as root:
            config=make_g1_config('rest_calibrated',resource_root=root,random_seed=9)
            self.assertEqual(config.srate,256)
            self.assertEqual(config.time_window,(0,2))
            self.assertEqual(config.baseline_duration_s,2)
            self.assertEqual(config.covariance_estimator,'oas')
            self.assertEqual(config.source_model,'lbo_network')
            self.assertEqual(config.spectral['spatial_length_mm'],40)
            self.assertEqual(config.spectral['temporal_exponent'],1.6)
            self.assertEqual(config.spectral['network_indices'],[3,5,6])
            self.assertTrue(Path(config.spectral['basis_path']).is_relative_to(Path(root)))

    def test_nested_overrides_and_isolation(self):
        overrides={'alpha_power_ratio':.1,'network_indices':[0]}
        one=make_g1_config('rest_calibrated',spectral=overrides,sensor_snr_db=10)
        overrides['network_indices'].append(2)
        self.assertEqual(one.spectral['network_indices'],[0])
        one.spectral['network_indices'].append(1)
        two=make_g1_config('rest_calibrated')
        self.assertEqual(two.spectral['network_indices'],[3,5,6])
        self.assertEqual(one.spectral['alpha_power_ratio'],.1)
        self.assertEqual(one.spectral['alpha_hz'],10.5)
        self.assertEqual(one.sensor_snr_db,10)

    def test_other_profiles_and_rejected_settings(self):
        rest=make_g1_config('rest_lbo')
        self.assertEqual(rest.source_model,'lbo')
        self.assertNotIn('network_path',rest.spectral)
        erp=make_g1_config('erp_network')
        self.assertEqual(erp.spectral['erp_spatial_mode'],'geodesic')
        self.assertEqual(erp.spectral['erp_local_width_mm'],35)
        with self.assertRaises(ValueError):
            make_g1_config('unknown')
        with self.assertRaises(ValueError):
            make_g1_config('rest_lbo',spectral=[])
        with self.assertRaises(TypeError):
            make_g1_config('rest_lbo',unknown_parameter=1)


if __name__=='__main__':
    unittest.main()
