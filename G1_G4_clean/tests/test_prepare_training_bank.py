"""Offline rate/protection/codec views preserve parents and calibrated evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import eye

from Simulation_G2.nmm_bank import initialize_bank,load_bank_sample,projection_hash,save_bank_sample
from Simulation_G2.prepare_training_bank import convert_bank,select_protected,comparison_metrics


class TrainingBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.bank=self.root/'original'
        self.projection=projection_hash(eye(4,format='csc'))
        initialize_bank(self.bank,atlas_id='test4',n_regions=4,projection_sha256=self.projection)
        t=np.arange(1000)/500
        self.source=np.array([11*np.exp(-((t-.6)/.025)**2),10*np.exp(-((t-.63)/.025)**2),
                              np.sin(2*np.pi*10*t)+np.sin(2*np.pi*200*t),np.sin(2*np.pi*7*t)])
        manifest=dict(signal_type='spike_jr',output_srate_hz=500,active_regions=[0],n_regions_output=4,
                      normalize=False,model={'class':'JansenRit'})
        self.entry=save_bank_sample(self.bank,'sample',self.source,t*1000,manifest,{'config':{'random_seed':7}})

    def test_conversion_protects_propagation_and_preserves_parent_group(self):
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.bank.iterdir()}
        result=convert_bank(self.bank,self.root/'views',ranks=[1],forward=np.eye(4))
        self.assertEqual(len(result['records']),3)
        dense,clock,entry=load_bank_sample(self.root/'views/dense','sample')
        self.assertEqual(dense.shape,(4,512))
        self.assertEqual(entry['simulation_group_id'],self.entry['simulation_group_id'])
        self.assertEqual(entry['storage']['protected_regions'],[0,1])
        np.testing.assert_allclose(clock,np.arange(512)*1000/256)
        for label in ('int16','lowrank_k001'):
            source,_,other=load_bank_sample(self.root/'views'/label,'sample')
            np.testing.assert_array_equal(source[:2],dense[:2])
            self.assertEqual(other['simulation_group_id'],self.entry['simulation_group_id'])
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.bank.iterdir()})
        report=result['records'][-1]['metrics']
        self.assertGreater(report['source_background_relative_error'],0)
        contexts=report['source_snr_context']
        self.assertAlmostEqual(contexts[1]['same_gain_sensor_error_relative_to_foreground']/contexts[0]['same_gain_sensor_error_relative_to_foreground'],1/np.sqrt(10))

    def test_whole_record_antialias_and_physical_duration(self):
        convert_bank(self.bank,self.root/'rate200',rate=200,codecs=['dense'])
        source,_,entry=load_bank_sample(self.root/'rate200/dense','sample')
        self.assertEqual(source.shape,(4,400))
        t=np.arange(400)/200
        self.assertLess(float(np.sqrt(np.mean((source[2,30:-30]-np.sin(2*np.pi*10*t[30:-30]))**2))),.01)
        self.assertEqual(entry['manifest']['training_view']['scope'],'whole_recording_before_cropping')

    def test_protection_native_and_resampled_union_and_non_psp_policy(self):
        native=np.zeros((4,500)); output=np.zeros((4,256))
        native[1,200]=9; output[2,100]=10
        ids,metadata=select_protected(native,output,self.entry,256,manual=[3])
        self.assertEqual(ids,[0,1,2,3])
        self.assertTrue(metadata['event_detection_applied'])
        inherited={**self.entry,'storage':{'protected_regions':[3]}}
        ids,metadata=select_protected(native,output,inherited,256,policy='configured')
        self.assertEqual(ids,[0,3])
        self.assertEqual(metadata['inherited_regions'],[3])
        _,metadata=select_protected(native,output,{**self.entry,'source_unit':'model_state'},256)
        self.assertFalse(metadata['event_detection_applied'])
        with self.assertRaisesRegex(ValueError,'mV'):
            select_protected(native,output,{**self.entry,'source_unit':'model_state'},256,policy='events')

    def test_conversion_flattens_declared_subject_ancestry(self):
        source,times,entry=load_bank_sample(self.bank,'sample')
        save_bank_sample(self.bank,'related',source,times,entry['manifest'],
            {'original_provenance':{'parent_data_group_ids':['subject-17']}},simulation_group_id='b'*64)
        convert_bank(self.bank,self.root/'parents',sample_ids=['related'],codecs=['dense'])
        _,_,converted=load_bank_sample(self.root/'parents/dense','related')
        self.assertIn('subject-17',converted['provenance']['parent_data_group_ids'])
        self.assertIn('b'*64,converted['provenance']['parent_data_group_ids'])

    def test_invalid_plan_fails_and_existing_output_is_not_overwritten(self):
        for args in (dict(rate=500),dict(ranks=[0]),dict(sample_ids=[]),dict(protected_regions=[True])):
            with self.subTest(args=args),self.assertRaises(ValueError):
                convert_bank(self.bank,self.root/'bad',**args)
        self.assertFalse((self.root/'bad').exists())
        convert_bank(self.bank,self.root/'existing',codecs=['dense'])
        with self.assertRaises(FileExistsError):
            convert_bank(self.bank,self.root/'existing',codecs=['dense'])

    def test_lossless_metrics_and_sensor_nullspace_are_explicit(self):
        x=self.source.astype(np.float32)
        report=comparison_metrics(x,x,500,[0],forward=np.eye(4))
        self.assertEqual(report['sensor_total_relative_error'],0)
        self.assertEqual(report['source_background_relative_error'],0)
        report=comparison_metrics(x,x*.9,500,[],forward=np.eye(4))
        self.assertEqual(report['source_snr_context'],[])


if __name__=='__main__':
    unittest.main()
