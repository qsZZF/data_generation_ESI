"""Projection conventions are explicit identities, never silent migrations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from Simulation_G4.atlas import parcel_projection, load_atlas


def small_bundle(directory):
    directory.mkdir()
    labels, ids, areas = np.array([3, 0, 3, 8]), np.array([3, 8]), np.array([1., 2., 3., 5.])
    projection, indices = parcel_projection(labels, ids, areas)
    gain = np.array([[1., 10., 3., 4.], [2., 0., 5., 6.]])
    np.savez_compressed(directory / 'atlas.npz', labels_vertex=labels, region_ids=ids,
                        vertex_area_mm2=areas, vertex_to_region=indices, leadfield=gain @ projection,
                        channel_names=np.array(['a', 'b']), region_names=np.array(['left', 'right']))
    sparse.save_npz(directory / 'projection.npz', projection)
    (directory / 'parcels.csv').write_text('original parcel table\n', encoding='utf-8')
    manifest = {'atlas_id': 'example', 'bundle_sha256': hashlib.sha256((directory / 'atlas.npz').read_bytes()).hexdigest()}
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return gain


class MembershipTests(unittest.TestCase):
    def test_membership_is_constant_within_parcel_and_projects_consistently(self):
        labels, areas = np.array([3, 0, 3, 8]), np.array([1., 2., 3., 5.])
        membership, indices = parcel_projection(labels, [3, 8], areas, mode='membership')
        area, _ = parcel_projection(labels, [3, 8], areas)
        np.testing.assert_array_equal(membership.toarray(), [[1, 0], [0, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(indices, [0, -1, 0, 1])
        np.testing.assert_allclose(area.toarray(), [[.25, 0], [0, 0], [.75, 0], [0, 1]])
        regional = np.array([[2., 4.], [3., 7.]])
        leadfield = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal((membership @ regional)[0], (membership @ regional)[2])
        np.testing.assert_allclose((leadfield @ membership) @ regional, leadfield @ (membership @ regional))
        with self.assertRaisesRegex(ValueError, 'projection mode'):
            parcel_projection(labels, [3, 8], areas, mode='unknown')

    def test_derive_keeps_source_unchanged_and_refuses_overwrite(self):
        from Simulation_G4.atlas import derive_projection_view
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gain = small_bundle(root / 'source')
            before = {p.name: p.read_bytes() for p in (root / 'source').iterdir()}
            view = derive_projection_view(root / 'source', root / 'membership', gain)
            self.assertEqual(str(view['projection_mode']), 'membership')
            self.assertEqual(view['manifest']['projection_mode'], 'membership')
            np.testing.assert_allclose(view['leadfield'], gain @ view['projection'])
            for name, content in before.items():
                self.assertEqual((root / 'source' / name).read_bytes(), content)
            self.assertEqual((root / 'membership' / 'parcels.csv').read_bytes(), before['parcels.csv'])
            with self.assertRaises(FileExistsError):
                derive_projection_view(root / 'source', root / 'membership', gain)
            with self.assertRaises(FileExistsError):
                derive_projection_view(root / 'source', root / 'source', gain)

    def test_mode_identity_mismatch_rejected(self):
        from Simulation_G4.atlas import derive_projection_view
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gain = small_bundle(root / 'source')
            self.assertEqual(load_atlas(root / 'source')['manifest'].get('projection_mode', 'area_unit_sum'), 'area_unit_sum')
            derive_projection_view(root / 'source', root / 'new', gain)
            path = root / 'new' / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['projection_mode'] = 'area_unit_sum'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'projection mode'):
                load_atlas(root / 'new')

    def test_assignment_corruption_rejected_even_without_projection_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            small_bundle(path)
            projection = sparse.load_npz(path / 'projection.npz')
            projection.data[:] = 1
            sparse.save_npz(path / 'projection.npz', projection)
            with self.assertRaisesRegex(ValueError, 'Projection'):
                load_atlas(path)

    def test_wrong_leadfield_and_cross_mode_rebuild_rejected_before_writing(self):
        from Simulation_G4.atlas import build_atlas, derive_projection_view
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gain = small_bundle(root / 'source')
            with self.assertRaisesRegex(ValueError, 'reproduce'):
                derive_projection_view(root / 'source', root / 'new', gain[:, ::-1])
            self.assertFalse((root / 'new').exists())
            with self.assertRaises(FileExistsError):
                build_atlas(root / 'source', projection_mode='membership')

    def test_named_membership_checks_mode_in_sidecar(self):
        from Simulation_G2.nmm_bank import projection_hash
        from Simulation_G3.atlas_profiles import get_atlas_profile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / 'Anatomy/G4_AAL_cortical_membership'
            graph = root / 'Anatomy/G3_AAL78_membership_TVB'
            atlas.mkdir(parents=True)
            graph.mkdir()
            np.savez(atlas / 'atlas.npz', region_names=[f'R{i}' for i in range(78)],
                     projection_mode=np.array('membership'))
            (atlas / 'manifest.json').write_text(json.dumps({'projection_mode': 'membership'}))
            projection = sparse.eye(78, format='csc')
            sparse.save_npz(atlas / 'projection.npz', projection)
            connectivity = graph / 'connectivity_78.zip'
            connectivity.write_bytes(b'fixture')
            metadata = dict(connectivity_file=connectivity.name, projection_mode='membership',
                connectivity_sha256=hashlib.sha256(connectivity.read_bytes()).hexdigest(),
                projection_sha256=projection_hash(projection),
                source_files_sha256={str(atlas / 'atlas.npz'): hashlib.sha256((atlas / 'atlas.npz').read_bytes()).hexdigest()})
            (graph / 'manifest.json').write_text(json.dumps(metadata))
            with patch('Simulation_G3.atlas_profiles.ROOT', root):
                self.assertEqual(get_atlas_profile('aal78_membership')['projection_mode'], 'membership')
                metadata['projection_mode'] = 'area_unit_sum'
                (graph / 'manifest.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'projection mode'):
                    get_atlas_profile('aal78_membership')


if __name__ == '__main__':
    unittest.main()
