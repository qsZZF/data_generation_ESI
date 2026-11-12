import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from Simulation_G1.geometry import compute_basis,save_basis
from Simulation_G1.g1_simulation_core import G1SimulationConfig
from test_spatial_patches import grid_simulator


class NetworkERPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.sim=grid_simulator(9)
        vertices=self.sim.vertices*self.sim.coord_to_mm
        self.bundle=compute_basis(vertices,self.sim.faces,24)
        save_basis(self.path/'basis.npz',self.bundle)
        maps=np.column_stack([np.exp(-((vertices[:,0]-center)**2+(vertices[:,1]-120)**2)/300)
                              for center in (110,125,135)])
        maps[:,1]*=-1
        np.savez(self.path/'maps.npz',source_prior_weights=maps,source_prior_raw_values=maps,
                 vertices_brainstorm_mm=vertices,faces=self.sim.faces)
        self.config=G1SimulationConfig(source_model='lbo_network',srate=128,time_window=(-.2,1.5),
            random_seed=1,normalize_source=False,component_seeds=dict(spatial=10,waveform=20,background=30,gain=40,sensor=50),
            spectral=dict(basis_path=str(self.path/'basis.npz'),network_path=str(self.path/'maps.npz'),
                scenario='erp',erp_spatial_mode='geodesic',spatial_length_mm=8,erp_local_width_mm=12,
                event_onset_s=0.,erp_latency_s=.2,erp_width_s=.06,erp_latency_jitter_s=.01,
                erp_component_interval_s=.2,erp_roi_power_ratio=2.,network_indices=[0,1,2]))
        # The public clock requires an integer number of event samples.
        self.config=replace(self.config,time_window=(-.25,1.5))

    def test_full_sample_contract_and_replay(self):
        one=self.sim.generate_sample(self.config,compute_diagnostics=False)
        two=self.sim.generate_sample(self.config,compute_diagnostics=False)
        for key in ('sensor','source_foreground','source_background','baseline_sensor','noise_covariance'):
            np.testing.assert_array_equal(one[key],two[key])
        self.assertEqual(one['baseline_sensor'].shape,(4,256))
        np.testing.assert_allclose(one['source_foreground']+one['source_background'],one['source_total'],atol=1e-6)
        self.assertGreater(np.linalg.matrix_rank(one['source_foreground'],tol=1e-5),1)
        meta=one['manifest']['network_erp']
        self.assertAlmostEqual(meta['roi_power_ratio_realized'],2,places=5)
        json.dumps(one['manifest'],allow_nan=False)
        covariance=one['noise_covariance']
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0],-1e-10)
        self.assertLess(np.linalg.norm(covariance@np.ones(4)),1e-8)
        source=one['source_foreground']
        roi=np.asarray(meta['roi_vertices'])
        t=self.config.time_window[0]+np.arange(source.shape[1])/self.config.srate
        active=(t>=meta['roi_window_s'][0])&(t<=meta['roi_window_s'][1])
        power=lambda x:np.average(np.mean(x[roi][:,active]**2,axis=1),weights=self.bundle['mass'][roi])
        self.assertAlmostEqual(power(source)/power(one['source_background']),2,places=5)
        # Centering introduces only a constant pre-onset offset.
        self.assertLess(np.max(np.ptp(source[:,t<0],axis=1)),1e-6)

    def test_streams_and_single_entry_cache(self):
        original=self.sim.generate_sample(self.config,compute_diagnostics=False)
        cached=self.sim._network_erp_weights_cache
        config=replace(self.config,component_seeds={**self.config.component_seeds,'waveform':21})
        changed=self.sim.generate_sample(config,compute_diagnostics=False)
        self.assertIs(cached,self.sim._network_erp_weights_cache)
        self.assertEqual(original['manifest']['network_erp']['weights_sha256'],changed['manifest']['network_erp']['weights_sha256'])
        self.assertFalse(np.array_equal(original['source_foreground'],changed['source_foreground']))
        _,_,_,baseline1=self.sim._generate_source_components(self.config)
        _,_,_,baseline2=self.sim._generate_source_components(config)
        np.testing.assert_array_equal(baseline1,baseline2)
        config=replace(self.config,component_seeds={**self.config.component_seeds,'spatial':11})
        changed=self.sim.generate_sample(config,compute_diagnostics=False)
        self.assertNotEqual(original['manifest']['network_erp']['weights_sha256'],changed['manifest']['network_erp']['weights_sha256'])

    def test_invalid_options_and_mask(self):
        for settings in ({'erp_roi_power_ratio':0},{'erp_secondary_gain':2},{'erp_component_interval_s':4},
                         {'network_indices':[0,0]},{'scenario':'rest'},{'erp_amplitude':3}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                self.sim.generate_sample(replace(self.config,spectral={**self.config.spectral,**settings}),compute_diagnostics=False)
        with self.assertRaises(ValueError):
            self.sim.generate_sample(replace(self.config,multiscale={'fraction':.1}),compute_diagnostics=False)
        # Masks are part of cache identity and are applied to every component.
        from Simulation_G1.g1_simulation_core import G1SourceSimulator
        mask=np.ones(self.sim.n_sources,dtype=bool);mask[:9]=False
        masked=G1SourceSimulator(self.sim.leadfield,self.sim.vertices,self.sim.faces,self.sim.vert_conn,cortex_mask=mask)
        sample=masked.generate_sample(self.config,compute_diagnostics=False)
        np.testing.assert_array_equal(sample['source_total'][~mask],0)


if __name__=='__main__':
    unittest.main()
