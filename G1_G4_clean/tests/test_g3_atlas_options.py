"""Atlas presets retain the legacy route and load real sparse projection files."""
from pathlib import Path
import tempfile
import unittest
import hashlib
import json
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G2.g2_simulation_core import load_projection_matrix
from Simulation_G3.atlas_profiles import get_atlas_profile, load_connectivity_metadata


class AtlasOptionsTests(unittest.TestCase):
    def test_named_aal_rejects_replaced_default_projection_or_atlas(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            anatomy = root / 'Anatomy/G4_AAL_cortical'
            anatomy.mkdir(parents=True)
            graph = root / 'Anatomy/G3_AAL78_TVB'
            graph.mkdir()
            atlas = anatomy / 'atlas.npz'
            np.savez(atlas, region_names=[f'R{i}' for i in range(78)])
            projection = sparse.eye(78, format='csc')
            sparse.save_npz(anatomy / 'projection.npz', projection)
            path = graph / 'connectivity_78.zip'
            path.write_bytes(b'graph-fixture')
            from Simulation_G2.nmm_bank import projection_hash
            (graph / 'manifest.json').write_text(json.dumps(dict(connectivity_file=path.name,
                connectivity_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                projection_sha256=projection_hash(projection),
                source_files_sha256={str(atlas): hashlib.sha256(atlas.read_bytes()).hexdigest()})))
            with patch('Simulation_G3.atlas_profiles.ROOT', root):
                self.assertEqual(get_atlas_profile('aal78')['n_regions'], 78)
                order = np.r_[1, 0, np.arange(2, 78)]
                sparse.save_npz(anatomy / 'projection.npz', projection[:, order])
                with self.assertRaisesRegex(ValueError, 'projection differs'):
                    get_atlas_profile('aal78')
                sparse.save_npz(anatomy / 'projection.npz', projection)
                np.savez(atlas, region_names=[f'Changed{i}' for i in range(78)])
                with self.assertRaisesRegex(ValueError, 'atlas bundle differs'):
                    get_atlas_profile('aal78')

    def test_legacy_profile_is_default(self):
        profile = get_atlas_profile()
        self.assertEqual(profile, get_atlas_profile('legacy998'))
        self.assertEqual((profile['n_regions_raw'], profile['n_regions']), (998, 994))
        self.assertTrue(profile['reshape_to_994'])
        with self.assertRaises(ValueError):
            get_atlas_profile('aal90')

    def test_sparse_npz_projection_preserves_values_and_checks_vertex_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'projection.npz'
            original = sparse.csr_matrix([[0., .2], [.5, 0.], [.5, .8]])
            sparse.save_npz(path, original)
            loaded = load_projection_matrix(path, n_vertices=3)
            self.assertEqual(loaded.format, 'csc')
            np.testing.assert_array_equal(loaded.toarray(), original.toarray())
            with self.assertRaisesRegex(ValueError, 'vertices'):
                load_projection_matrix(path, n_vertices=4)
            sparse.save_npz(path, sparse.csr_array(original))
            array_loaded = load_projection_matrix(path)
            self.assertIsInstance(array_loaded, sparse.csc_matrix)
            np.testing.assert_array_equal(array_loaded[:, 0].indices, [1, 2])

    def test_connectivity_assumptions_must_match_bundle_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'connectivity.zip'
            path.write_bytes(b'fixture')
            self.assertIsNone(load_connectivity_metadata(path))
            sidecar = path.parent / 'manifest.json'
            sidecar.write_text(json.dumps(dict(connectivity_file=path.name,
                connectivity_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tract_lengths='centroid distance proxy')))
            self.assertEqual(load_connectivity_metadata(path)['tract_lengths'], 'centroid distance proxy')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_connectivity_metadata(path)
