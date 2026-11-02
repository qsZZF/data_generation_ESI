"""End-to-end physical amplitudes, fixed measurement noise and sample labels."""
from dataclasses import replace
import unittest
import numpy as np

from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator
import test_unified_nmm as nmm_fixture


class PhysicalG1Tests(unittest.TestCase):
    def configuration(self):
        return G1SimulationConfig(srate=128, time_window=(0, 1), seed_vertices=[40],
            patch_area_mm2=80, random_seed=19, normalize_source=False,
            amplitude_mode='physical', foreground_peak_am=1e-9, leadfield_source_unit='A*m',
            sensor_units='V', sensor_noise_std=1e-6, sensor_snr_db=None,
            sensor_noise_type='iid_white', record_observability=True)

    def test_fixed_noise_and_physical_scaling_without_visibility_equalization(self):
        sim, config = grid_simulator(), self.configuration()
        first = sim.generate_sensor(config, compute_diagnostics=False)
        doubled = sim.generate_sensor(replace(config, foreground_peak_am=2e-9), compute_diagnostics=False)
        np.testing.assert_array_equal(first[2], doubled[2])
        np.testing.assert_allclose(doubled[3], 2*first[3], rtol=2e-7, atol=0)
        np.testing.assert_array_equal(first[-1]['baseline_sensor'], doubled[-1]['baseline_sensor'])
        self.assertAlmostEqual(doubled[-1]['realized_sensor_snr_db']-first[-1]['realized_sensor_snr_db'], 20*np.log10(2), places=5)
        self.assertAlmostEqual(np.max(abs(first[3]))/1e-9, 1., places=6)
        weak = grid_simulator()
        weak.leadfield *= .1
        weaker = weak.generate_sensor(config, compute_diagnostics=False)
        np.testing.assert_array_equal(first[2], weaker[2])
        self.assertAlmostEqual(weaker[-1]['realized_sensor_snr_db']-first[-1]['realized_sensor_snr_db'], -20., places=5)

    def test_background_and_baseline_share_current_gain(self):
        sim = grid_simulator()
        config = replace(self.configuration(), source_background_type='pink', source_snr_db=12)
        a = sim._generate_source_components(config)
        b = sim._generate_source_components(replace(config, foreground_peak_am=2e-9))
        for index in (0, 2, 3):
            np.testing.assert_allclose(b[index], 2*a[index], rtol=2e-7, atol=0)
        foreground = a[0].astype(float)-a[2]
        self.assertAlmostEqual(np.max(abs(foreground))/1e-9, 1., places=5)
        self.assertAlmostEqual(10*np.log10(np.mean(foreground**2)/np.mean(a[2]**2)), 12., places=5)

    def test_labels_reproduce_and_zero_noise_is_not_fake_information(self):
        sim, config = grid_simulator(), self.configuration()
        first = sim.generate_sensor(config, compute_diagnostics=False)[-1]
        again = sim.generate_sensor(config, compute_diagnostics=False)[-1]
        self.assertEqual(first['observability'], again['observability'])
        labels = first['observability']
        self.assertAlmostEqual(labels['geometric_observable_energy_fraction']+labels['geometric_null_energy_fraction'], 1.)
        clean = sim.generate_sensor(replace(config, sensor_noise_std=0), compute_diagnostics=False)[-1]
        self.assertIsNone(clean['realized_sensor_snr_db'])
        self.assertEqual(clean['observability']['noise_status'], 'zero_noise_covariance')
        self.assertIsNone(clean['observability']['white_null_energy_fraction'])

    def test_physical_mode_requires_unambiguous_contract(self):
        config = self.configuration()
        for invalid in (replace(config, normalize_source=True), replace(config, sensor_snr_db=10),
                        replace(config, leadfield_source_unit=None), replace(config, sensor_units=None)):
            with self.assertRaises(ValueError):
                grid_simulator().generate_sensor(invalid)

    def test_mixed_units_have_no_pooled_power_or_psd(self):
        config = replace(self.configuration(), modality='MEG', average_reference=False,
            sensor_units=['T', 'T', 'T/m', 'T/m'], sensor_noise_std=[1e-12]*4)
        for std in ([1e-12]*4, [0.]*4):
            result = grid_simulator().generate_sensor(replace(config, sensor_noise_std=std))[-1]
            self.assertIsNone(result['realized_sensor_snr_db'])
            self.assertIsNone(result['clean_sensor_power'])
            self.assertIsNone(result['noise_sensor_power'])
            self.assertNotIn('clean_sensor_psd_summary', result)
            self.assertEqual(result['sensor_psd_status'], 'mixed_units_no_pooled_diagnostic')


class PhysicalNMMTests(unittest.TestCase):
    def test_unified_physical_noise_is_independent_of_current_gain(self):
        fixture = nmm_fixture.UnifiedNMMTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = replace(fixture.config, amplitude_mode='physical', foreground_peak_am=1e-9,
            normalize_source=False, sensor_snr_db=None, sensor_units='V', sensor_noise_std=1e-6,
            leadfield_source_unit='A*m', record_observability=True)
        first = fixture.sim.generate_sample(config)
        second = fixture.sim.generate_sample(replace(config, foreground_peak_am=2e-9))
        np.testing.assert_array_equal(first['sensor_noise'], second['sensor_noise'])
        for key in ('source_foreground', 'source_background', 'region_source'):
            np.testing.assert_allclose(second[key], 2*first[key], rtol=2e-7, atol=0)
        self.assertEqual(first['manifest']['source_unit'], 'A*m')
        self.assertEqual(first['manifest']['native_temporal_model_unit'], 'mV_PSP')
        self.assertAlmostEqual(np.max(abs(first['source_foreground']))/1e-9, 1., places=6)
        self.assertIn('observability', first['manifest'])
