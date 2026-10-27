"""End-to-end period replay, component SNR, deterministic batches and worker reset."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G2.g2_simulation_core import G2SourceSimulator
from Simulation_G2.nmm_bank import initialize_bank,projection_hash,save_bank_sample,load_bank_sample
from Simulation_G2.period_index import build_period_index
from Simulation_G2.training_dataset import NMMRecipeDataset
from test_spatial_patches import grid_simulator


class NMMTrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.bank=self.root/'bank'
        grid=grid_simulator(4)
        projection=sparse.csc_matrix(np.repeat(np.eye(4),4,axis=0))
        cortex=dict(Vertices=grid.vertices,Faces=grid.faces,VertConn=grid.vert_conn)
        self.backend=G2SourceSimulator(grid.leadfield,cortex,projection,self.bank)
        self.projection_hash=projection_hash(projection)
        initialize_bank(self.bank,atlas_id='test',n_regions=4,projection_sha256=self.projection_hash,schema_version=2)
        self.t=np.arange(250)/100
        self.signal=np.array([np.sin(2*np.pi*(5+i)*self.t+.3*i)+.3*np.cos(2*np.pi*2*self.t) for i in range(4)])
        self.save(self.bank,'one',seed=1)

    def save(self,bank,name,*,seed,rest=False,group=None):
        manifest=dict(signal_type='resting_jr' if rest else 'spike_jr',output_srate_hz=100.,
            active_regions=[] if rest else [0],normalize=False,random_seed=seed,
            model={'class':'JansenRit','active_A':3.25 if rest else 3.6,'background_A':3.25})
        data=self.signal if not rest else np.roll(self.signal,seed,axis=1)
        return save_bank_sample(bank,name,data,self.t*1000,manifest,{},
            storage={'codec':'lowrank','rank':2},simulation_group_id=group)

    def build(self,folder='index',**kwargs):
        return build_period_index([self.bank],self.root/folder,**({'split_fractions':(1.,0.,0.)}|kwargs))

    def test_periods_match_canonical_slices_and_repeat_after_random_access(self):
        manifest=self.build(sensor_snrs=(None,),normalize_source=False)
        ds=NMMRecipeDataset(self.root/'index',self.backend)
        self.assertEqual(len(ds),2)
        self.assertEqual(manifest['statistics']['discarded_tail_samples'],50)
        decoded,_,_=load_bank_sample(self.bank,'one')
        first=ds[0]; ds[1]; again=ds[0]
        expected=decoded[:,:100]-decoded[:,:100].mean(axis=1,keepdims=True)
        np.testing.assert_allclose(first['region_source'],expected,atol=1e-7)
        for key in first:
            if key!='manifest':np.testing.assert_array_equal(first[key],again[key])
        self.assertGreater(ds.cache_info()['banks'][0]['hits'],0)
        self.assertTrue(np.all(first['sensor_noise']==0))
        self.assertEqual(first['manifest']['sampling_rate_hz'],100.)
        self.assertFalse(first['manifest']['parameter_preset_is_activation_truth'])

    def test_partition_spatial_and_both_snrs_survive_float32_export(self):
        self.build(source_mode='partition',source_snr_db=20,sensor_snrs=(10,),
                   spatial_profiles=('diffuse',),sensor_noise_types=('iid_white',))
        sample=NMMRecipeDataset(self.root/'index',self.backend)[0]
        areas=self.backend.vertex_area
        power=lambda value:np.average(np.mean(value.astype(float)**2,axis=1),weights=areas)
        ratio=10*np.log10(power(sample['source_foreground'])/power(sample['source_background']))
        self.assertAlmostEqual(ratio,20,places=5)
        sensor_snr=10*np.log10(np.mean(sample['clean_sensor'].astype(float)**2)/np.mean(sample['sensor_noise'].astype(float)**2))
        self.assertAlmostEqual(sensor_snr,10,places=5)
        np.testing.assert_allclose(sample['source'],sample['source_foreground']+sample['source_background'],atol=1e-7)
        np.testing.assert_allclose(sample['sensor'],sample['clean_sensor']+sample['sensor_noise'],atol=2e-6)
        self.assertEqual(sample['manifest']['background_policy'],'same_trajectory_partition_reweighted')
        self.assertFalse(sample['manifest']['source_matches_static_projection'])

    def test_spatial_modulation_does_not_turn_psp_offset_into_activity(self):
        for profile in ('uniform','decay','diffuse','contract','pulse'):
            self.build(folder=profile,source_mode='partition',source_snr_db=20,
                       spatial_profiles=(profile,),sensor_snrs=(None,))
            ds=NMMRecipeDataset(self.root/profile,self.backend)
            row=ds.recipe(0)
            window,entry=ds._window(row['foreground'])
            expected=ds[0]
            with patch.object(ds,'_window',return_value=(window.astype(float)+23.,entry)):
                actual=ds[0]
            for field in ('source_foreground','source_background','sensor'):
                with self.subTest(profile=profile,field=field):
                    np.testing.assert_allclose(actual[field],expected[field],atol=2e-6)

    def test_independent_background_period_and_parent_identity(self):
        bg=self.root/'bg'
        initialize_bank(bg,atlas_id='test',n_regions=4,projection_sha256=self.projection_hash,schema_version=2)
        self.save(bg,'rest',seed=2,rest=True)
        self.build(source_mode='selected',background_bank_dirs=[bg],source_snr_db=10)
        sample=NMMRecipeDataset(self.root/'index',self.backend)[0]
        self.assertEqual(sample['manifest']['background_policy'],'independent_rest')
        self.assertEqual(len(sample['manifest']['parent_data_group_ids']),2)
        self.assertAlmostEqual(sample['manifest']['realized_source_snr_db'],10)

    def test_threaded_batch_and_bounded_prefetch_equal_serial(self):
        self.build(variants_per_period=4,sensor_snrs=(0.,10.),sensor_noise_types=('iid_white','pink'))
        ds=NMMRecipeDataset(self.root/'index',self.backend,cache_bytes=100000)
        sequential=ds.get_batch([2,0,3],fields=('sensor','region_source'))
        threaded=ds.get_batch([2,0,3],fields=('sensor','region_source'),workers=2)
        np.testing.assert_array_equal(threaded['sensor'],sequential['sensor'])
        expected=list(ds.iter_batches(3,shuffle=True,seed=12,fields=('sensor',)))
        actual=list(ds.iter_batches(3,shuffle=True,seed=12,fields=('sensor',),workers=2,prefetch=2))
        for left,right in zip(expected,actual):
            np.testing.assert_array_equal(left['sensor'],right['sensor'])
            self.assertEqual([m['recipe_id'] for m in left['manifests']],[m['recipe_id'] for m in right['manifests']])
        self.assertLessEqual(ds.cache_info()['bytes'],100000)
        self.assertEqual(sum(len(batch['sensor']) for batch in actual),len(ds))
        ds.clear_cache(); self.assertEqual(ds.cache_info()['bytes'],0)

    def test_pickle_resets_cache_and_replays_same_values(self):
        self.build()
        ds=NMMRecipeDataset(self.root/'index',self.backend)
        first=ds[0]
        other=pickle.loads(pickle.dumps(ds))
        self.assertEqual(other.cache_info()['bytes'],0)
        np.testing.assert_array_equal(other[0]['sensor'],first['sensor'])

    def test_fold_selection_and_changed_inputs_fail(self):
        self.save(self.bank,'two',seed=2,group='2'*64)
        self.save(self.bank,'three',seed=3,group='3'*64)
        self.build(split_fractions=(.8,.1,.1))
        for split in ('train','val','test'):
            ds=NMMRecipeDataset(self.root/'index',self.backend,split=split)
            self.assertTrue(all(ds.recipe(i)['split']==split for i in range(len(ds))))
        index=self.bank/'bank.json'
        content=json.loads(index.read_text()); content['atlas_id']='modified'
        index.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError,'frozen'):
            ds[0]


if __name__=='__main__':
    unittest.main()
