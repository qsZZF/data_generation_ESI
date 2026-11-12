"""Small mechanism tests, not empirical realism certification."""
import unittest
import numpy as np

from Simulation_G1.geometry import compute_basis
from Simulation_G1.wx_pilot import build_weights, generate_timecourses


class WXPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x,y = np.meshgrid(np.arange(9)*10., np.arange(7)*10.)
        vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        faces = []
        for j in range(6):
            for i in range(8):
                a = j*9+i
                faces.extend([[a,a+1,a+9],[a+1,a+10,a+9]])
        cls.bundle = compute_basis(vertices,np.array(faces),24)
        cls.maps = np.column_stack([np.exp(-((x.ravel()-cx)**2+(y.ravel()-30)**2)/500)
                                    for cx in [20,40,60]])
        cls.mask = np.ones(x.size,dtype=bool)
        cls.mask[:2] = False
        cls.weights = build_weights(cls.bundle,cls.maps,cls.mask,length_mm=10)

    def generate(self, scenario='erp', variant='wx', **kw):
        return generate_timecourses(self.weights,scenario=scenario,variant=variant,seed=17,**kw)

    def test_spatial_mask_normalization_and_reproducibility(self):
        repeated = build_weights(self.bundle,self.maps,self.mask,length_mm=10)
        np.testing.assert_array_equal(self.weights['W'],repeated['W'])
        np.testing.assert_array_equal(self.weights['W'][~self.mask],0)
        np.testing.assert_allclose(np.sum(self.weights['mass'][:,None]*self.weights['W']**2,axis=0),1)

    def test_rank_and_causal_erp(self):
        old,new = self.generate(variant='synchronous'),self.generate()
        self.assertEqual(np.linalg.matrix_rank(old['X_foreground']),1)
        self.assertGreater(np.linalg.matrix_rank(self.weights['W']@new['X_foreground']),1)
        np.testing.assert_array_equal(new['X_foreground'][:,new['time']<.2],0)
        np.testing.assert_array_equal(old['X_background'],new['X_background'])
        self.assertAlmostEqual(new['metadata']['roi_component_power_ratio'],2)

    def test_all_scenarios_replay_and_additivity(self):
        W = self.weights['W']
        for scenario in ('rest','erp','ersp'):
            one,two = self.generate(scenario),self.generate(scenario)
            for key in ('X_foreground','X_background','X_baseline'):
                np.testing.assert_array_equal(one[key],two[key])
                self.assertTrue(np.isfinite(one[key]).all())
            np.testing.assert_allclose(W@(one['X_foreground']+one['X_background']),
                                       W@one['X_foreground']+W@one['X_background'],atol=1e-14)
            self.assertEqual(one['X_baseline'].shape,(9,512))

    def test_ersp_is_delta_and_rest_has_no_target_baseline(self):
        result = self.generate('ersp')
        np.testing.assert_array_equal(result['X_foreground'][:,result['time']<.3],0)
        self.assertGreaterEqual(result['metadata']['min_alpha_amplitude_multiplier'],0)
        self.assertIn(-.5,result['metadata']['modulation_depths'])
        rest = self.generate('rest')
        np.testing.assert_array_equal(rest['X_baseline'],0)
        np.testing.assert_array_equal(rest['X_background'],0)

    def test_invalid_inputs(self):
        for kw in ({'rho':-1},{'srate':20},{'duration_s':1},{'baseline_s':float('nan')},
                   {'duration_s':2.000001}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.generate(**kw)
        with self.assertRaises(ValueError):
            build_weights(self.bundle,self.maps,self.mask,length_mm=-1)
        with self.assertRaises(ValueError):
            build_weights(self.bundle,self.maps,self.mask,n_networks=4)


if __name__ == '__main__':
    unittest.main()
