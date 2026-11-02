"""Deterministic spatial source selection for the offline template EEG forward."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prepare_nullspace_hdeeg import farthest_point_indices, select_sources, prepare_forward


class SourceSpacesFixture(list):
    def copy(self):
        return copy.deepcopy(self)


class TemplateHdEEGTests(unittest.TestCase):
    def test_fps_is_reproducible_unique_and_nested(self):
        points = np.random.default_rng(14).normal(size=(100, 3))
        short = farthest_point_indices(points, 10)
        long = farthest_point_indices(points, 40)
        np.testing.assert_array_equal(short, long[:10])
        np.testing.assert_array_equal(long, farthest_point_indices(points, 40))
        self.assertEqual(len(set(long)), 40)
        self.assertEqual(long[0], np.argmax(np.sum((points-points.mean(axis=0))**2, axis=1)))
        self.assertEqual(len(set(farthest_point_indices(np.zeros((5, 3)), 5))), 5)
        for invalid in (0, 101, True, 1.5):
            with self.assertRaises(ValueError):
                farthest_point_indices(points, invalid)

    def test_source_selection_preserves_geometry_clears_stale_patch_data(self):
        rng = np.random.default_rng(15)
        sources = SourceSpacesFixture()
        for _ in range(2):
            sources.append(dict(type='surf', rr=rng.normal(size=(20, 3)), nn=rng.normal(size=(20, 3)),
                vertno=np.arange(0,20,2), inuse=np.tile([1,0],10), nuse=10,
                tris=np.array([[0,1,2]]), use_tris=np.array([[0,2,4]]), nuse_tri=1,
                patch_inds=np.arange(10), pinfo=[np.array([0])], nearest=np.array([0]),
                nearest_dist=np.ones(20), dist=np.eye(20), dist_limit=np.array([.1])))
        selected, orders = select_sources(sources, 4)
        for original, reduced, order in zip(sources, selected, orders):
            self.assertEqual(original['nuse'], 10)
            self.assertEqual(reduced['nuse'], 4)
            np.testing.assert_array_equal(reduced['vertno'], np.sort(order))
            np.testing.assert_array_equal(np.flatnonzero(reduced['inuse']), reduced['vertno'])
            for key in ('rr','nn','tris'):
                np.testing.assert_array_equal(reduced[key], original[key])
            self.assertEqual(reduced['nuse_tri'], 0)
            for key in ('use_tris','patch_inds','pinfo','nearest','nearest_dist','dist','dist_limit'):
                self.assertIsNone(reduced[key])

    def test_missing_cached_assets_fail_without_import_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict('sys.modules', {'mne': None}):
                with self.assertRaisesRegex(FileNotFoundError, 'no download'):
                    prepare_forward(Path(temporary)/'output', Path(temporary)/'missing')
            self.assertFalse((Path(temporary)/'output').exists())


if __name__ == '__main__':
    unittest.main()
