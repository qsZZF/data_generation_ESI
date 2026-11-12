import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Simulation_G1.geometry import compute_basis,save_basis
from Simulation_G1.network_recipes import build_network_recipes
from Simulation_G1.training_recipes import RecipeDataset
from test_spatial_patches import grid_simulator


class NetworkRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.sim=grid_simulator(9)
        coords=self.sim.vertices*self.sim.coord_to_mm
        bundle=compute_basis(coords,self.sim.faces,16)
        save_basis(self.path/'basis.npz',bundle)
        maps=np.column_stack([np.exp(-np.sum((coords[:,:2]-center)**2,axis=1)/200)
                              for center in ((110,110),(120,120),(130,130))])
        np.savez(self.path/'networks.npz',vertices_brainstorm_mm=coords,faces=self.sim.faces,
                 source_prior_weights=maps,source_prior_raw_values=maps)
        self.overrides=dict(spectral=dict(basis_path=str(self.path/'basis.npz'),
            network_path=str(self.path/'networks.npz'),network_indices=[0,1,2],n_modes=16))

    def build(self,name='recipes',**kwargs):
        options=dict(n_groups=3,variants_per_group=2,split_fractions=(1/3,1/3,1/3),
                     presets=('rest_calibrated','erp_network'),config_overrides=self.overrides)
        options.update(kwargs)
        return build_network_recipes(self.path/name,self.sim,**options)

    def test_group_split_streams_and_replay(self):
        manifest=self.build()
        self.assertEqual(manifest['n_samples'],12)
        dataset=RecipeDataset(self.path/'recipes',self.sim,split=None)
        groups={}
        for index in range(len(dataset)):
            config,row=dataset.recipe(index)
            groups.setdefault(row['support_id'],[]).append((config,row))
        for rows in groups.values():
            self.assertEqual(len({r['split'] for _,r in rows}),1)
            self.assertEqual(len({c.component_seeds['spatial'] for c,_ in rows}),1)
            self.assertEqual(len({c.component_seeds['waveform'] for c,_ in rows}),4)
        self.assertEqual({rows[0][1]['split'] for rows in groups.values()},{'train','val','test'})
        first=dataset[0]
        again=dataset[0]
        np.testing.assert_array_equal(first['sensor'],again['sensor'])
        self.assertEqual(first['baseline_sensor'].shape,(4,512))
        self.assertEqual(first['noise_covariance'].shape,(4,4))
        np.testing.assert_allclose(first['source_total'],first['source_foreground']+first['source_background'],atol=1e-7)
        erp=dataset[2]
        self.assertEqual(erp['sensor'].shape,(4,512))
        self.assertGreater(np.linalg.matrix_rank(erp['source_foreground']),1)

    def test_resource_change_is_rejected_before_replay(self):
        self.build()
        dataset=RecipeDataset(self.path/'recipes',self.sim,split=None)
        with (self.path/'basis.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Spectral resource checksum'):
            dataset.recipe(0)

    def test_deterministic_metadata_and_snapshot(self):
        one=self.build('one')
        two=self.build('two')
        self.assertEqual((self.path/'one/recipes-00000.jsonl').read_bytes(),
                         (self.path/'two/recipes-00000.jsonl').read_bytes())
        self.assertEqual(one['split_support_counts'],two['split_support_counts'])
        self.assertIn('network_erp.py',one['generator']['files'])
        self.assertTrue((self.path/'one/generator_snapshot/Simulation_G1/presets.py').exists())

    def test_invalid_inputs_do_not_create_output(self):
        for kwargs in ({'n_groups':0},{'variants_per_group':True},{'sensor_snrs':[]},
                       {'split_fractions':(.4,.4,.4)},{'presets':['rest_lbo','rest_lbo']},
                       {'config_overrides':{'random_seed':1}}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.build(**kwargs)
        self.assertFalse((self.path/'recipes').exists())


if __name__=='__main__':
    unittest.main()
