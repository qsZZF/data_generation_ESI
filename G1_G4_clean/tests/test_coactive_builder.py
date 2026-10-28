"""Import safety, explicit coordinate registration and association provenance."""
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import build_fs_coactive as builder


class CoactiveBuilderTests(unittest.TestCase):
    def test_import_never_loads_optional_packages_downloads_or_reads_matrix(self):
        original_import = builtins.__import__
        forbidden = {'nimare', 'mne', 'nibabel', 'matplotlib', 'joblib'}

        def checked_import(name, *args, **kwargs):
            if name.split('.')[0] in forbidden:
                raise AssertionError(f'Optional package imported eagerly: {name}')
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location('_coactive_import_test', builder.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.__import__', side_effect=checked_import), \
                patch('numpy.load', side_effect=AssertionError('matrix read at import')), \
                patch('builtins.input', side_effect=AssertionError('interactive import')), \
                patch('logging.basicConfig', side_effect=AssertionError('global logging changed')):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.build_macm_surface_z_matrix))

    def test_transform_required_before_optional_imports_and_data_access(self):
        with patch('builtins.__import__', side_effect=AssertionError('unexpected heavy import')):
            with self.assertRaisesRegex(ValueError, 'required'):
                builder.build_macm_surface_z_matrix('missing_dataset.pkl.gz')
        invalid = [np.eye(3), np.full((4, 4), np.nan), np.zeros((4, 4))]
        singular = np.eye(4); singular[1, 1] = 0
        projective = np.eye(4); projective[3, 0] = .01
        for value in [*invalid, singular, projective]:
            with self.subTest(transform=value), self.assertRaises(ValueError):
                builder.build_macm_surface_z_matrix('missing_dataset.pkl.gz', surface_to_mask_world_mm=value)
        affine = np.eye(4); affine[:3, 3] = [3, 4, 5]
        np.testing.assert_array_equal(builder.validate_surface_transform(affine), affine)

    def test_empty_vertex_bins_do_not_divide_by_zero(self):
        with np.errstate(all='raise'):
            result = builder._vertex_average(np.array([2., 4., 8.]), np.array([0, 2, 2]), np.array([1, 0, 2]))
        np.testing.assert_array_equal(result, [2., 0., 6.])

    def test_existing_output_is_protected_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'legacy.npz'
            output.write_bytes(b'existing artifact')
            with patch.dict(sys.modules, {'nimare': None, 'mne': None, 'nibabel': None}):
                with self.assertRaises(FileExistsError):
                    builder.build_macm_surface_z_matrix('unused', output_npz=output,
                                                        surface_to_mask_world_mm=np.eye(4))
            self.assertEqual(output.read_bytes(), b'existing artifact')

    def test_small_mock_build_applies_transform_and_keeps_statistic_and_provenance(self):
        # Synthetic dependency fixtures test orchestration; no NiMARE dataset is rebuilt.
        modules = {name: ModuleType(name) for name in
                   ('nimare', 'nimare.dataset', 'nimare.meta', 'nimare.meta.kernel', 'mne', 'nibabel')}
        maps = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
        mask = SimpleNamespace(affine=np.eye(4), get_fdata=lambda: np.ones((2, 1, 1)))
        dataset = SimpleNamespace(ids=['a', 'b', 'c'], masker=SimpleNamespace(mask_img=mask))
        modules['nimare.dataset'].Dataset = SimpleNamespace(load=Mock(return_value=dataset))
        modules['nimare.meta.kernel'].MKDAKernel = Mock(return_value=SimpleNamespace(
            transform=Mock(return_value=maps)))
        modules['mne'].read_source_spaces = Mock(return_value=[
            dict(rr=np.array([[-.01, 0., 0.]]), vertno=np.array([0]), coord_frame=5),
            dict(rr=np.array([[-.009, 0., 0.]]), vertno=np.array([0]), coord_frame=5)])
        modules['mne'].datasets = SimpleNamespace(fetch_fsaverage=Mock(side_effect=AssertionError('download')))
        modules['nibabel'].affines = SimpleNamespace(
            apply_affine=lambda affine, points: np.asarray(points) @ affine[:3, :3].T + affine[:3, 3])
        transform = np.eye(4); transform[0, 3] = 10.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, source_path, output = root/'dataset.fixture', root/'source.fixture', root/'out.npz'
            dataset_path.write_bytes(b'dataset fixture'); source_path.write_bytes(b'source fixture')
            with patch.dict(sys.modules, modules):
                matrix, coords = builder.build_macm_surface_z_matrix(dataset_path, source_path,
                    min_studies=1, output_npz=output, surface_to_mask_world_mm=transform)
            expected = np.array([.5 * (maps[0] + maps[1]), .5 * (maps[1] + maps[2])])
            expected = (expected - maps.mean(axis=0)) / maps.std(axis=0, ddof=1)
            np.testing.assert_allclose(matrix, expected, rtol=1e-6)
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['coords'], coords)
                np.testing.assert_allclose(saved['coords_mask_world_mm'], [[0, 0, 0], [1, 0, 0]], atol=1e-5)
                np.testing.assert_array_equal(saved['surface_to_mask_world_mm'], transform)
                np.testing.assert_array_equal(saved['seed_vox_idx'], [0, 1])
                metadata = json.loads(str(saved['metadata_json']))
                self.assertIn('not permutation Z', metadata['interpretation'])
                self.assertIn('no cortical-ribbon or distance constraint', metadata['limitation'])
                self.assertIn('not independently verified', metadata['voxel_order'])
                self.assertEqual(metadata['legacy_builder_sha256'], builder.LEGACY_BUILDER_SHA256)
                self.assertEqual(metadata['processed_seed_count'], 2)
                self.assertEqual(len(metadata['dataset_sha256']), 64)

    def test_cli_no_args_only_shows_help_and_explicit_args_are_forwarded(self):
        with patch.object(builder, 'build_macm_surface_z_matrix') as build, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(builder.main([]), 0)
            build.assert_not_called()
        self.assertIn('--surface-to-mask-world-mm', output.getvalue())
        with tempfile.TemporaryDirectory() as temporary:
            affine = Path(temporary)/'affine.txt'
            np.savetxt(affine, np.eye(4))
            with patch.object(builder, 'build_macm_surface_z_matrix') as build, patch('logging.basicConfig'):
                self.assertEqual(builder.main(['--dataset', 'dataset.pkl.gz', '--source-space', 'source.fif',
                    '--output', 'new.npz', '--surface-to-mask-world-mm', str(affine),
                    '--radius', '8', '--min-studies', '7', '--jobs', '2']), 0)
            self.assertEqual(build.call_args.args, (Path('dataset.pkl.gz'), Path('source.fif'), 8., 2, 7, Path('new.npz')))
            np.testing.assert_array_equal(build.call_args.kwargs['surface_to_mask_world_mm'], np.eye(4))


if __name__ == '__main__':
    unittest.main()
