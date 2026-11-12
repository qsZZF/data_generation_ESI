import json
from pathlib import Path
import tempfile
import unittest

from simulate_source_sensor import build_parser, build_g1_config


class SpectralCLITests(unittest.TestCase):
    def test_preset_defaults_and_explicit_legacy_values(self):
        args=build_parser().parse_args(['--engine','g1','--g1-preset','rest_calibrated'])
        config=build_g1_config(args)
        self.assertEqual(config.srate,256)
        self.assertEqual(config.time_window,(0,2))
        self.assertEqual(config.sensor_snr_db,20)
        self.assertEqual(config.covariance_estimator,'oas')
        args=build_parser().parse_args(['--engine','g1','--g1-preset','rest_calibrated',
            '--srate-hz=500','--duration-s','1','--sensor-snr-db','10','--covariance-estimator','empirical'])
        config=build_g1_config(args)
        self.assertEqual(config.srate,500)
        self.assertEqual(config.time_window,(0,1))
        self.assertEqual(config.sensor_snr_db,10)
        self.assertEqual(config.covariance_estimator,'empirical')

    def test_preset_spectral_override_and_conflicting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'spectral.json'
            path.write_text(json.dumps({'alpha_power_ratio':.1}),encoding='utf-8')
            args=build_parser().parse_args(['--engine','g1','--g1-preset','rest_calibrated',
                '--g1-spectral-config',str(path),'--resource-root',directory])
            config=build_g1_config(args)
            self.assertEqual(config.spectral['alpha_power_ratio'],.1)
            self.assertEqual(config.spectral['alpha_hz'],10.5)
            self.assertTrue(Path(config.spectral['network_path']).is_relative_to(Path(directory)))
        args=build_parser().parse_args(['--engine','g1','--g1-preset','erp_network','--source-type','erp'])
        with self.assertRaisesRegex(ValueError,'omit --source-type'):
            build_g1_config(args)

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

    def test_registered_noise_and_residual_flags_reach_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'residual.json'
            path.write_text(json.dumps(dict(fraction=.1,n_components=4)),encoding='utf-8')
            args=build_parser().parse_args(['--engine','g1','--source-type','erp',
                '--g1-source-model','lbo','--g1-multiscale-config',str(path),
                '--sensor-noise-type','real','--real-noise-registry','noise_registry.json',
                '--real-noise-record-id','rest','--real-noise-split','train',
                '--allow-combined-real-source-background'])
            config=build_g1_config(args)
            config.validate_recipe()
            self.assertEqual(config.multiscale,dict(fraction=.1,n_components=4))
            self.assertEqual(config.real_noise_record_id,'rest')
            self.assertEqual(config.real_noise_split,'train')
            self.assertTrue(config.allow_combined_real_source_background)
