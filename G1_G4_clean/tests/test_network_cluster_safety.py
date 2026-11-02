"""Small fixtures protect legacy alignment, semantics and execution boundaries."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class NetworkClusterSafetyTests(unittest.TestCase):
    def test_scripts_import_without_optional_imaging_packages_or_logging_changes(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'nilearn', 'nibabel', 'matplotlib', 'wordcloud', 'sklearn', 'tqdm', 'pandas'}:
                raise AssertionError(f'Heavy import during module import: {name}')
            return original(name, *args, **kwargs)

        for name in ('neurovault_data_cluster', 'neurovault_400_clusters'):
            spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            with patch('builtins.__import__', side_effect=guarded), patch('logging.basicConfig') as logging_setup:
                spec.loader.exec_module(module)
            logging_setup.assert_not_called()
            self.assertTrue(callable(module.main))

    def test_failed_images_keep_terms_aligned_to_actual_image_count(self):
        from network_cluster_utils import mask_aligned_images
        weights = np.arange(12).reshape(3, 4)

        def transform(path):
            if path == 'bad':
                raise ValueError('invalid image')
            return np.array([[float(path), 2.]])

        x, terms, usable = mask_aligned_images(['1', 'bad', '3'], weights, transform)
        np.testing.assert_array_equal(usable, [True, False, True])
        np.testing.assert_array_equal(terms, weights[[0, 2]])
        np.testing.assert_array_equal(x[:, 0], [1, 3])
        with self.assertRaisesRegex(ValueError, 'aligned'):
            mask_aligned_images(['1'], weights, transform)
        with self.assertRaisesRegex(ValueError, 'No usable'):
            mask_aligned_images(['bad'], weights[:1], transform)

    def test_provenance_records_replay_seed_and_effective_input_hash(self):
        from network_cluster_utils import write_provenance
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            fixture = folder / 'meta.csv'
            fixture.write_text('id\n17\n', encoding='utf-8')
            target = folder / 'provenance.json'
            values = np.arange(15).reshape(3, 5)
            write_provenance(target, seed=17, arrays={'X': values}, input_files=[fixture])
            first = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(first['random_seed'], 17)
            self.assertEqual(first['input_files'][str(fixture)], hashlib.sha256(fixture.read_bytes()).hexdigest())
            write_provenance(target, seed=17, arrays={'X': values.copy()}, input_files=[fixture])
            self.assertEqual(first, json.loads(target.read_text(encoding='utf-8')))
            values[0, 0] = 99
            write_provenance(target, seed=17, arrays={'X': values}, input_files=[fixture])
            self.assertNotEqual(first['arrays']['X']['sha256'], json.loads(target.read_text(encoding='utf-8'))['arrays']['X']['sha256'])

    def test_final_bootstrap_semantic_projection_preserves_image_scores(self):
        # Research-method contract: PCA image scores, not spatial loadings,
        # multiply the same selected images' term vectors.
        for name in ('neurovault_data_cluster.py', 'neurovault_400_clusters.py'):
            tree = ast.parse((ROOT / name).read_text(encoding='utf-8'))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            projections = [node for node in calls if isinstance(node.func, ast.Attribute)
                           and node.func.attr == 'dot' and len(node.args) == 2
                           and isinstance(node.args[0], ast.Name) and node.args[0].id == 'image_scores']
            self.assertEqual(len(projections), 1)
            self.assertEqual(projections[0].args[1].id, 'term_weights_selected')
            source = ast.unparse(tree)
            self.assertNotIn('np.random.choice(', source)
            self.assertNotIn('gaussian_filter(', source)


if __name__ == '__main__':
    unittest.main()
