"""Dataset-qualified measured ancestry follows G4 simulations into bank export."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G2.nmm_bank import projection_hash
from Simulation_G4.pipeline import reconstruct_groups
from Simulation_G4.provenance import reconstruction_ancestry
from Simulation_G4.simulator import sample_parameters


def group_id(kind, value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return f'real_noise_{kind}_' + hashlib.sha256(encoded).hexdigest()


class G4ExportAncestryTests(unittest.TestCase):
    def setUp(self):
        self.provenance = dict(recording=np.array(['fit_record'] * 2 + ['held_record'] * 2),
            window_id=np.array(['fit:0', 'fit:1', 'held:0', 'held:1']))

    def ancestry(self, dataset='dataset-a', condition='rest', fitting=(0, 1), evaluation=(2, 3)):
        return reconstruction_ancestry(dataset, 'S001', condition, self.provenance,
                                       np.asarray(fitting), np.asarray(evaluation))

    def test_fitting_parents_are_distinct_from_evaluation_only_provenance(self):
        metadata = self.ancestry()
        self.assertEqual(metadata['dataset'], 'dataset-a')
        parents = set(metadata['parent_data_group_ids'])
        self.assertIn(group_id('subject', 'dataset-a:S001'), parents)
        self.assertIn(group_id('recording', 'dataset-a:S001:fit_record'), parents)
        self.assertNotIn(group_id('recording', 'dataset-a:S001:held_record'), parents)
        self.assertEqual(metadata['fitting_recording_ids'], ['dataset-a:S001:fit_record'])
        self.assertEqual(metadata['evaluation_recording_ids'], ['dataset-a:S001:held_record'])
        self.assertEqual(metadata['fitting_window_ids'], ['fit:0', 'fit:1'])
        self.assertEqual(metadata['evaluation_window_ids'], ['held:0', 'held:1'])
        reversed_metadata = self.ancestry(fitting=(2, 3), evaluation=(0, 1))
        fit_windows = {value for value in parents if value.startswith('g4_fitwindow_')}
        evaluation_windows = {value for value in reversed_metadata['parent_data_group_ids']
                              if value.startswith('g4_fitwindow_')}
        self.assertEqual(len(fit_windows), 2)
        self.assertFalse(fit_windows & evaluation_windows)

    def test_same_subject_and_record_names_are_isolated_between_datasets(self):
        first, second = self.ancestry(), self.ancestry(dataset='dataset-b')
        self.assertFalse(set(first['parent_data_group_ids']) & set(second['parent_data_group_ids']))
        self.assertEqual(first, self.ancestry())

    def test_same_subject_links_conditions_and_matches_qualified_noise_registry_id(self):
        first, second = self.ancestry(), self.ancestry(condition='task')
        shared = set(first['parent_data_group_ids']) & set(second['parent_data_group_ids'])
        self.assertIn(group_id('subject', 'dataset-a:S001'), shared)
        self.assertIn(group_id('recording', 'dataset-a:S001:fit_record'), shared)

    def test_missing_dataset_does_not_invent_exportable_ancestry(self):
        metadata = self.ancestry(dataset=None)
        self.assertIsNone(metadata['dataset'])
        self.assertEqual(metadata['parent_data_group_ids'], [])

    def test_reconstruction_artifact_preserves_group_ancestry_and_model_atlas_identity(self):
        rng = np.random.default_rng(1201)
        rate, n_times = 100, 200
        eeg = rng.normal(size=(8, 8, n_times)).astype(np.float32) * 1e-6
        provenance = dict(subject=np.full(8, 'S001'), condition=np.full(8, 'rest'),
            recording=np.array(['fit_record'] * 4 + ['held_record'] * 4),
            trial=np.array(['trial-a'] * 4 + ['trial-b'] * 4),
            window_id=np.array([f'window:{index}' for index in range(8)]))
        prior = sample_parameters(rng, 4)
        parameters = {key: np.tile(np.atleast_1d(value), (8, 1)) for key, value in prior.items()}
        projection = sparse.eye(4, format='csc')
        atlas = dict(region_names=np.array(['r1', 'r2', 'r3', 'r4']), aal_roi_numbers=np.arange(1, 5),
                     projection=projection, manifest=dict(atlas_id='fixture-membership', bundle_sha256='fixture-hash',
                                                         projection_mode='membership'))
        leadfield = rng.normal(size=(8, 4))

        def simulate(parameters, connectivity, *, sampling_rate_hz, duration_s, seed, **kwargs):
            time = np.arange(round(duration_s * sampling_rate_hz)) / sampling_rate_hz
            source = np.sin(2 * np.pi * np.arange(6, 10)[:, None] * time)
            source += np.random.default_rng(seed).normal(size=source.shape) * .1
            return source, time, dict(model='paper_dual_timescale_jr', source_unit='mV_PSP', seed=seed)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            options = dict(sampling_rate_hz=rate, prior_parameters=prior, device='cpu', dataset_id='dataset-a')
            with patch('Simulation_G4.pipeline.predict', return_value=parameters), \
                 patch('Simulation_G4.pipeline.simulate', side_effect=simulate) as simulator, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = reconstruct_groups(None, eeg, provenance, atlas, np.zeros((4, 4)), leadfield,
                                            output, **options)
                self.assertEqual(simulator.call_count, 6)
                repeated = reconstruct_groups(None, eeg, provenance, atlas, np.zeros((4, 4)), leadfield,
                                              output, **options)
                self.assertEqual(simulator.call_count, 6)
                self.assertEqual(report, repeated)
            with np.load(output / report[0]['file'], allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata_json']))
                self.assertEqual(archive['region_source_mV'].shape, (3, 4, n_times))
                self.assertEqual(metadata['dataset'], 'dataset-a')
                self.assertTrue(metadata['parent_data_group_ids'])
                self.assertIn(group_id('subject', 'dataset-a:S001'), metadata['parent_data_group_ids'])
                self.assertEqual(metadata['atlas_id'], atlas['manifest']['atlas_id'])
                self.assertEqual(metadata['projection_sha256'], projection_hash(projection))
                self.assertEqual(metadata['source_unit'], 'mV_PSP')
                model = metadata['model']
                self.assertEqual(model['class'],'DualTimescaleJansenRit')
                self.assertEqual(model['implementation'],'paper_dual_timescale_jr')
                self.assertEqual(len(metadata['simulations']), 3)
                self.assertEqual(metadata['fitting_window_ids'], archive['fitting_window_ids'].tolist())
                self.assertEqual(metadata['evaluation_window_ids'], archive['evaluation_window_ids'].tolist())


if __name__ == '__main__':
    unittest.main()
