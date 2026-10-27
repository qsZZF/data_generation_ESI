import json
from pathlib import Path
import tempfile
import unittest

from simulate_source_sensor import build_parser, build_g1_config


class SpectralCLITests(unittest.TestCase):
    def test_spectral_json_and_explicit_mode_reach_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'spectral.json'
            settings=dict(basis_path='basis.npz',scenario='ersp',ersp_depth=-.3)
            path.write_text(json.dumps(settings),encoding='utf-8')
            args=build_parser().parse_args(['--engine','g1','--source-type','erp',
                '--g1-source-model','lbo','--g1-spectral-config',str(path),'--srate-hz','256'])
            config=build_g1_config(args)
            config.validate_recipe()
            self.assertEqual(config.spectral,settings)
            self.assertEqual(config.source_model,'lbo')
            self.assertEqual(config.srate,256)
            self.assertEqual(config.baseline_duration_s,2)

    def test_legacy_default_remains_patch(self):
        args=build_parser().parse_args(['--engine','g1','--source-type','erp'])
        config=build_g1_config(args)
        self.assertEqual(config.source_model,'patch')
        self.assertEqual(config.spectral,{})
