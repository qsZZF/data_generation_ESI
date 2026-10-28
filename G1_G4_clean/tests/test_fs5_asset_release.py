import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_fs5_asset_release import export_network_archive,inspect_coactive,verified_copy,digest
from scripts.project_network_priors_to_fsaverage5 import save_prior_files


class AssetReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)

    def test_legacy_labels_are_recovered_without_unpickling(self):
        source=self.folder/'legacy.npz';csv_path=self.folder/'qc.csv';output=self.folder/'safe.npz'
        weights=np.array([[1.,-2.],[3.,4.]],dtype=np.float32)
        np.savez(source,source_prior_weights=weights,network_names=np.array(['unread','unread'],dtype=object),
                 source_paths=np.array(['unread','unread'],dtype=object))
        with csv_path.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['name','source_file']);writer.writeheader()
            writer.writerows([dict(name='A',source_file='a.nii.gz'),dict(name='B',source_file='b.nii.gz')])
        before=digest(source)
        info=export_network_archive(source,csv_path,output)
        self.assertEqual(digest(source),before)
        self.assertTrue(info['numeric_arrays_preserved'])
        with np.load(output,allow_pickle=False) as data:
            np.testing.assert_array_equal(data['source_prior_weights'],weights)
            np.testing.assert_array_equal(data['network_names'],['A','B'])
            for key in data.files:self.assertFalse(data[key].dtype.hasobject)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            export_network_archive(source,csv_path,source)

    def test_projection_writer_now_emits_unicode_labels(self):
        weights=np.ones((3,2),dtype=np.float32);vertices=np.zeros((3,3));faces=np.array([[0,1,2]])
        save_prior_files(self.folder,'test','signed_maxabs',weights,weights,['A','B'],
                         [Path('a.nii.gz'),Path('b.nii.gz')],vertices,vertices,faces,{})
        with np.load(self.folder/'test_fsaverage5_20k_source_priors_signed_maxabs.npz',allow_pickle=False) as data:
            for key in data.files:self.assertFalse(data[key].dtype.hasobject)
            self.assertEqual(data['network_names'].tolist(),['A','B'])

    def test_wrong_csv_column_names_are_rejected_against_matlab(self):
        from scipy.io import savemat
        source=self.folder/'network.npz';qc=self.folder/'qc.csv'
        np.savez(source,source_prior_weights=np.ones((3,2)))
        savemat(source.with_suffix('.mat'),dict(networkNames=np.array(['A','B'],dtype=object),
                                               sourcePaths=np.array(['a','b'],dtype=object)))
        qc.write_text('name,source_file\nB,b\nA,a\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'label order'):
            export_network_archive(source,qc,self.folder/'released.npz')

    def test_streamed_matrix_check_rejects_nonfinite_and_wrong_coordinates(self):
        path=self.folder/'coactive.npz'
        matrix=np.arange(16,dtype=np.float32).reshape(4,4)
        np.savez_compressed(path,M=matrix,coords=np.zeros((4,3)))
        report=inspect_coactive(path)
        self.assertEqual(report['shape'],[4,4]);self.assertEqual(report['maximum'],15)
        matrix[3,3]=np.nan;np.savez_compressed(path,M=matrix,coords=np.zeros((4,3)))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):inspect_coactive(path)
        np.savez_compressed(path,M=np.eye(4),coords=np.zeros((3,3)))
        with self.assertRaisesRegex(ValueError,'coordinates'):inspect_coactive(path)

    def test_verified_copy_refuses_bad_checksum(self):
        source=self.folder/'source';source.write_bytes(b'asset')
        destination=self.folder/'release'/'copy'
        with self.assertRaisesRegex(ValueError,'checksum'):
            verified_copy(source,destination,'0'*64)
        self.assertFalse(destination.exists())
        self.assertEqual(verified_copy(source,destination),digest(source))
        self.assertEqual(destination.read_bytes(),b'asset')
