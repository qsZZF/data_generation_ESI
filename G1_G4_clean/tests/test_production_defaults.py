"""Unspecified noise is IID; neural backgrounds require an explicit resource."""
import unittest
from Simulation_G1.g1_simulation_core import G1SimulationConfig,G1ScenarioSampler,G1DifficultyScenarioSampler
from Simulation_G2.g2_simulation_core import G2SimulationConfig
from simulate_source_sensor import build_parser


class ProductionDefaultsTests(unittest.TestCase):
    def test_configs_and_cli_use_unambiguous_iid_measurement_noise(self):
        self.assertEqual(G1SimulationConfig().sensor_noise_type,'iid_white')
        self.assertEqual(G2SimulationConfig().sensor_noise_type,'iid_white')
        self.assertEqual(build_parser().parse_args(['--engine','g1']).sensor_noise_type,'iid_white')

    def test_unresourced_random_scenarios_do_not_invent_spatial_background(self):
        samplers=[G1ScenarioSampler(random_seed=7)]+[
            G1DifficultyScenarioSampler(level,random_seed=7) for level in ('easy','medium','hard','extreme')]
        for sampler in samplers:
            for index in range(12):
                config=sampler.sample(str(index),'template','mesh','gain')
                self.assertEqual(config.sensor_noise_type,'iid_white')
                self.assertIsNone(config.source_background_type)
                self.assertIsNone(config.source_snr_db)
                config.validate_recipe()


if __name__=='__main__':
    unittest.main()
