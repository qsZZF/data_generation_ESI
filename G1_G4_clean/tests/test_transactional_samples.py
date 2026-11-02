"""Samples publish all-or-nothing and resume only after identity/hash checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1.sample_io import save_sample_directory, verified_sample


class TransactionalSampleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.arrays = {'sensor': np.arange(12).reshape(3, 4), 'baseline_sensor': np.ones((3, 8))}
        self.request = {'seed': 9, 'resources': {'leadfield_sha256': 'fixture'}}

    def save(self, **kwargs):
        return save_sample_directory(self.root, 'sample', self.arrays, {'sample_id': 'sample'},
                                     request_identity=self.request, **kwargs)

    def test_committed_layout_no_overwrite_and_verified_resume(self):
        expected = self.save(artifacts={'overview.png': b'fixture-plot'})
        self.assertEqual(verified_sample(self.root, 'sample', self.request), expected)
        self.assertEqual(self.save(resume=True), expected)
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertIsNone(verified_sample(self.root, 'absent', self.request))
        with np.load(self.root / 'sample/sample_data.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['sensor'], self.arrays['sensor'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            verified_sample(self.root, 'sample', {'seed': 10})

    def test_each_committed_file_is_checked(self):
        for filename in ('sample_data.npz', 'sample_manifest.json', 'overview.png'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                save_sample_directory(root, 'sample', self.arrays, {}, request_identity=self.request,
                                      artifacts={'overview.png': b'fixture'})
                path = root / 'sample' / filename
                path.write_bytes(path.read_bytes() + b'corruption')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    verified_sample(root, 'sample', self.request)

    def test_failed_write_and_failed_publish_leave_no_committed_sample(self):
        for target in ('numpy.savez_compressed', 'Simulation_G1.sample_io._publish_directory'):
            with self.subTest(target=target), patch(target, side_effect=OSError('injected failure')):
                with self.assertRaises(OSError):
                    self.save()
            self.assertFalse((self.root / 'sample').exists())
            self.assertFalse(list(self.root.glob('*.staging')))

    def test_unsafe_ids_unserializable_manifest_and_legacy_directory_are_rejected(self):
        for value in ('../escape', 'a/b', 'a\\b', '.', 'CON', 'name.'):
            with self.assertRaises(ValueError):
                save_sample_directory(self.root, value, self.arrays, {}, request_identity=self.request)
        with self.assertRaises((TypeError, ValueError)):
            save_sample_directory(self.root, 'sample', self.arrays, {'array': np.ones(2)},
                                  request_identity=self.request)
        legacy = self.root / 'sample'
        legacy.mkdir()
        (legacy / 'keep.txt').write_text('existing user data')
        with self.assertRaises(ValueError):
            self.save(resume=True)
        self.assertEqual((legacy / 'keep.txt').read_text(), 'existing user data')

    def test_competing_writers_publish_one_complete_sample(self):
        def write():
            try:
                return self.save()
            except FileExistsError:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: write(), range(2)))
        self.assertEqual(sum(result == 'conflict' for result in results), 1)
        self.assertIsInstance(verified_sample(self.root, 'sample', self.request), dict)
        self.assertFalse(list(self.root.glob('*.staging')))

    def test_standard_cli_writer_preserves_input_and_resumes_explicitly(self):
        from simulate_source_sensor import save_standard_sample
        rng = np.random.default_rng(77)
        source = rng.normal(size=(4, 128)).astype(np.float32)
        clean = source[:3]
        noise = clean * .1
        baseline = rng.normal(size=(3, 256)).astype(np.float32)
        covariance = np.cov(baseline)
        metadata = dict(sample_id='sample', srate=128, random_seed=77,
            baseline_sensor=baseline, noise_covariance=covariance,
            oracle_noise_covariance=covariance, sensor_noise_covariance=covariance)
        arguments = (self.root, 'sample', clean + noise, clean, noise, source, None, metadata, False)
        expected = save_standard_sample(*arguments)
        self.assertIn('baseline_sensor', metadata)
        self.assertEqual(save_standard_sample(*arguments, resume=True), expected)
        with self.assertRaises(FileExistsError):
            save_standard_sample(*arguments)

    def test_g1_cli_uses_named_arrays_without_noise_lists(self):
        import simulate_source_sensor as pipeline
        from unittest.mock import Mock
        source = np.arange(4 * 128, dtype=np.float32).reshape(4, 128)
        sample = dict(sensor=source[:3], clean_sensor=source[:3], sensor_noise=np.zeros((3, 128)),
            source_total=source, source_foreground=source, source_background=np.zeros_like(source),
            baseline_sensor=np.zeros((3, 256)), noise_covariance=np.zeros((3, 3)),
            oracle_noise_covariance=np.zeros((3, 3)), sensor_noise_covariance=np.zeros((3, 3)),
            manifest=dict(sample_id='sample', srate=128, random_seed=2))
        simulator = Mock()
        simulator.generate_sample.return_value = sample
        args = pipeline.build_parser().parse_args(['--engine', 'g1', '--source-type', 'erp',
            '--sample-id', 'sample', '--srate-hz', '128', '--duration-s', '1'])
        with patch.object(pipeline, 'load_brainstorm_cortex', return_value={'Vertices': np.zeros((3, 4))}), \
             patch.object(pipeline, 'load_brainstorm_leadfield', return_value=np.ones((3, 4))), \
             patch.object(pipeline.G1SourceSimulator, 'from_brainstorm_cortex', return_value=simulator):
            pipeline.run_g1(args, self.root)
        simulator.generate_sensor.assert_not_called()
        with np.load(self.root / 'sample/sample_data.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['source_total'], source)
            self.assertIn('source_foreground', saved.files)
        manifest = json.loads((self.root / 'sample/sample_manifest.json').read_text())
        self.assertNotIn('baseline_sensor', manifest)


if __name__ == '__main__':
    unittest.main()
