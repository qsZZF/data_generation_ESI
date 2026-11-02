"""Multiscale truth follows every source gain and is exported outside JSON."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from Simulation_G1.geometry import compute_basis, save_basis
from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator


class ComponentExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sim=grid_simulator(6)
        basis=compute_basis(self.sim.vertices*self.sim.coord_to_mm,self.sim.faces,n_modes=5)
        save_basis(self.root/'basis.npz',basis)
        self.spectral=dict(basis_path=str(self.root/'basis.npz'),scenario='erp',n_modes=5)
        self.residual=dict(fraction=.2,n_components=3,length_mm=2.,seed=100)

    def test_spectral_physical_and_normalized_export_reconstruct_source(self):
        base=G1SimulationConfig(srate=128,time_window=(0,1),random_seed=5,
            source_model='lbo',spectral=self.spectral,multiscale=self.residual)
        for config in (base,replace(base,amplitude_mode='physical',normalize_source=False,
                foreground_peak_am=1e-9,leadfield_source_unit='A*m',sensor_units='V',
                sensor_noise_std=1e-6,sensor_snr_db=None)):
            components={}
            *_,source,meta=self.sim.generate_sensor(config,source_components=components)
            total=sum(components[key] for key in ('foreground_coarse','foreground_residual',
                                                  'background_coarse','background_residual'))
            np.testing.assert_allclose(total,source,rtol=3e-6,atol=1e-16)
            for value in components.values():
                self.assertTrue(np.isfinite(value).all())
            output=self.root/config.amplitude_mode
            manifest=self.sim.save_sample(output,config,save_source=True)
            json.dumps(manifest)
            with np.load(output/config.sample_id/manifest['source_components_file']) as data:
                for key in components:
                    np.testing.assert_array_equal(data[key],components[key])

    def test_patch_background_uses_shared_gain_and_exports_baseline_components(self):
        config=G1SimulationConfig(srate=128,time_window=(0,1),random_seed=8,patch_area_mm2=8,
            source_snr_db=10,source_background_type='spectral_rest',
            source_background_spectral={**self.spectral,'scenario':'rest'},
            source_background_multiscale=self.residual)
        components={}
        source,meta,background,baseline=self.sim._generate_source_components(config,components=components)
        np.testing.assert_array_equal(components['foreground_residual'],0)
        np.testing.assert_allclose(components['background_coarse']+components['background_residual'],background)
        np.testing.assert_allclose(components['baseline_background_coarse']+components['baseline_background_residual'],baseline)
        np.testing.assert_allclose(components['foreground_coarse']+background,source)
        self.assertEqual(baseline.shape[1],256)
        power=lambda x:np.mean(np.asarray(x,dtype=float)**2)
        self.assertAlmostEqual(10*np.log10(power(components['foreground_coarse'])/power(background)),10,places=5)

    def test_disabled_feature_leaves_sensor_arrays_identical(self):
        config=G1SimulationConfig(srate=128,time_window=(0,1),random_seed=5,
            source_model='lbo',spectral=self.spectral)
        a=self.sim.generate_sensor(config)
        components={}
        b=self.sim.generate_sensor(replace(config,multiscale={'fraction':0}),source_components=components)
        for x,y in zip(a[:4],b[:4]):
            np.testing.assert_array_equal(x,y)


if __name__=='__main__':
    unittest.main()
