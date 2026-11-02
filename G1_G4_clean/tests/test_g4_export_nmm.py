"""G4 exports preserve fitted-model identity, physical PSP units and ancestry."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from Simulation_G4.export_nmm import export_reconstructions
from Simulation_G4.simulator import sample_parameters
from Simulation_G2.nmm_bank import initialize_bank, projection_hash, save_bank_sample, load_bank_sample
from Simulation_G2.bank_reader import BankWindowReader
from Simulation_G2.g2_simulation_core import G2SourceSimulator
from Simulation_G2.period_index import build_period_index
from Simulation_G2.training_dataset import NMMRecipeDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class G4ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.atlas_dir = self.root / 'atlas'
        self.atlas_dir.mkdir()
        self.vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        self.projection = sparse.csc_matrix([[1., 0], [1., 0], [0, 1.], [0, 1.]])
        self.gain = np.array([[1., .2, 0, .1], [0, 1, .3, .1], [.2, 0, 1, .4]])
        self.names = np.array(['roi_left', 'roi_right'])
        np.savez(self.atlas_dir / 'atlas.npz', labels_vertex=[1, 1, 2, 2], region_ids=[1, 2],
            vertex_area_mm2=np.ones(4), vertex_to_region=[0, 0, 1, 1], projection_mode='membership',
            vertices=self.vertices, faces=self.faces, region_names=self.names, aal_roi_numbers=[1, 2],
            leadfield=self.gain @ self.projection)
        sparse.save_npz(self.atlas_dir / 'projection.npz', self.projection)
        self.atlas_manifest = dict(atlas_id='toy_g4_membership', projection_mode='membership',
            bundle_sha256=sha(self.atlas_dir / 'atlas.npz'),
            projection_file_sha256=sha(self.atlas_dir / 'projection.npz'))
        (self.atlas_dir / 'manifest.json').write_text(json.dumps(self.atlas_manifest))
        self.run = self.root / 'run'
        (self.run / 'reconstructed_sources').mkdir(parents=True)
        self.sc_hash = hashlib.sha256(np.eye(2).tobytes()).hexdigest()
        self.run_manifest = dict(dataset='toy_dataset', atlas_manifest=self.atlas_manifest,
            bank_config=dict(atlas_id=self.atlas_manifest['atlas_id'], atlas_hash=self.atlas_manifest['bundle_sha256'],
                             connectivity_sha256=self.sc_hash, drive_mode='independent', drive_correlation=0.),
            sampling_rate_hz=128.)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_manifest))
        self.parameters = sample_parameters(np.random.default_rng(5), 2)
        self.time = np.arange(256) / 128.
        self.source = np.array([[np.sin(2*np.pi*frequency*self.time + phase) for frequency in (8, 11)]
                                for phase in (0., .3)]).astype(np.float32)
        self.metadata = dict(dataset='toy_dataset', subject='subject_1', condition='rest',
            parent_data_group_ids=['actual_subject_1', 'actual_recording_1'],
            fitting_recording_ids=['recording_1'], evaluation_recording_ids=['recording_1'],
            atlas_id=self.atlas_manifest['atlas_id'], atlas_bundle_sha256=self.atlas_manifest['bundle_sha256'],
            projection_sha256=projection_hash(self.projection), connectivity_sha256=self.sc_hash,
            source_unit='mV_PSP', simulations=[dict(model='paper_dual_timescale_jr', seed=10 + index,
                source_unit='mV_PSP', sampling_rate_hz=128., drive_mode='independent', drive_correlation=0.) for index in range(2)])
        self.artifact = self.run / 'reconstructed_sources/group_0000.npz'
        self.write_group()

    def write_group(self):
        fingerprint = 'a' * 64
        np.savez(self.artifact, region_source_mV=self.source, time_s=self.time,
            fingerprint=fingerprint, region_names=self.names, metadata_json=json.dumps(self.metadata),
            fitting_window_ids=['fit_1', 'fit_2'], evaluation_window_ids=['eval_1', 'eval_2'], **self.parameters)
        group = dict(subject=self.metadata['subject'], condition=self.metadata['condition'], status='forward_checked',
            fitting_windows=2, evaluation_windows=2,
            file=str(Path('reconstructed_sources') / self.artifact.name))
        self.artifact.with_suffix('.json').write_text(json.dumps(dict(fingerprint=fingerprint,
            artifact_sha256=sha(self.artifact), result=group)))
        (self.run / 'reconstruction_report.json').write_text(json.dumps(dict(dataset='toy_dataset', groups=[group])))

    def test_export_replays_through_bank_and_period_interfaces(self):
        output = self.root / 'export'
        summary = export_reconstructions(self.run, output, atlas_dir=self.atlas_dir)
        self.assertEqual(summary['n_records'], 2)
        reader = BankWindowReader(output)
        ids = sorted(reader.bank['samples'])
        groups = []
        for repeat, sample_id in enumerate(ids):
            source, entry = reader.read(sample_id, 0, 256)
            np.testing.assert_array_equal(source, self.source[repeat])
            self.assertEqual(entry['source_unit'], 'mV_PSP')
            self.assertEqual(entry['signal_type'], 'data_guided_jr')
            self.assertEqual(entry['active_regions'], [])
            self.assertEqual(entry['manifest']['model']['class'], 'DualTimescaleJansenRit')
            groups.append(entry['simulation_group_id'])
        self.assertEqual(len(set(groups)), 1)
        index = self.root / 'periods'
        manifest = build_period_index([output], index, split_fractions=(1., 0., 0.), sensor_snrs=(None,))
        self.assertEqual(manifest['parent_splits']['actual_subject_1'], 'train')
        cortex = dict(Vertices=self.vertices, Faces=self.faces, VertConn=sparse.csr_matrix(np.ones((4, 4)) - np.eye(4)))
        backend = G2SourceSimulator(self.gain, cortex, self.projection, output)
        sample = NMMRecipeDataset(index, backend)[0]
        self.assertEqual(sample['manifest']['signal_type'], 'data_guided_jr')
        self.assertIn('actual_subject_1', sample['manifest']['parent_data_group_ids'])
        with self.assertRaises(FileExistsError):
            export_reconstructions(self.run, output, atlas_dir=self.atlas_dir)

    def test_missing_ancestry_wrong_atlas_and_corrupt_artifact_fail_before_export(self):
        for key, value in [('parent_data_group_ids', []), ('connectivity_sha256', 'b' * 64),
                           ('atlas_bundle_sha256', 'b' * 64), ('source_unit', 'A*m')]:
            original = self.metadata[key]
            self.metadata[key] = value
            self.write_group()
            with self.subTest(key=key), self.assertRaises(ValueError):
                export_reconstructions(self.run, self.root / 'invalid', atlas_dir=self.atlas_dir)
            self.assertFalse((self.root / 'invalid').exists())
            self.metadata[key] = original
        self.write_group()
        self.artifact.write_bytes(self.artifact.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            export_reconstructions(self.run, self.root / 'invalid', atlas_dir=self.atlas_dir)

    def test_explicit_bank_unit_boundary_and_legacy_fallback(self):
        bank = self.root / 'units'
        initialize_bank(bank, atlas_id='toy', n_regions=2, projection_sha256=projection_hash(self.projection), schema_version=2)
        base = dict(signal_type='data_guided_jr', output_srate_hz=128., normalize=False,
                    active_regions=[], model={'class': 'DualTimescaleJansenRit'})
        save_bank_sample(bank, 'explicit', self.source[0], self.time * 1000,
                         dict(base, source_unit='mV_PSP'), {})
        self.assertEqual(load_bank_sample(bank, 'explicit')[2]['source_unit'], 'mV_PSP')
        save_bank_sample(bank, 'legacy', self.source[0], self.time * 1000,
                         dict(base, model={'class': 'JansenRit'}), {})
        self.assertEqual(load_bank_sample(bank, 'legacy')[2]['source_unit'], 'mV_PSP')
        for index, settings in enumerate((dict(source_unit='A*m'), dict(source_unit='mV_PSP', normalize=True),
                                           dict(source_unit='normalized_model_state', normalize=False))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                save_bank_sample(bank, f'invalid_{index}', self.source[0], self.time * 1000, dict(base, **settings), {})

    def test_conditions_share_subject_ancestry_but_keep_distinct_parameter_fits(self):
        report_path = self.run / 'reconstruction_report.json'
        original = json.loads(report_path.read_text())['groups'][0]
        self.artifact = self.run / 'reconstructed_sources/group_0001.npz'
        self.metadata['condition'] = 'task'
        self.write_group()
        second = json.loads(report_path.read_text())['groups'][0]
        report_path.write_text(json.dumps(dict(dataset='toy_dataset', groups=[original, second])))
        output = self.root / 'conditions'
        summary = export_reconstructions(self.run, output, atlas_dir=self.atlas_dir)
        self.assertEqual(summary['n_records'], 4)
        self.assertEqual(len({row['simulation_group_id'] for row in summary['records']}), 2)
        # All conditions/draws depend on the same known subject and cannot be split apart.
        with self.assertRaisesRegex(ValueError, 'independent parent groups'):
            build_period_index([output], self.root / 'leaky_splits', split_fractions=(.5, .5, 0.))

    def test_reordered_regions_and_sidecar_report_mismatch_are_rejected(self):
        self.names = self.names[::-1]
        self.write_group()
        with self.assertRaisesRegex(ValueError, 'region order'):
            export_reconstructions(self.run, self.root / 'wrong_order', atlas_dir=self.atlas_dir)
        self.names = self.names[::-1]
        self.write_group()
        sidecar_path = self.artifact.with_suffix('.json')
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['result']['condition'] = 'different'
        sidecar_path.write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(ValueError, 'report differs'):
            export_reconstructions(self.run, self.root / 'wrong_report', atlas_dir=self.atlas_dir)


if __name__ == '__main__':
    unittest.main()
