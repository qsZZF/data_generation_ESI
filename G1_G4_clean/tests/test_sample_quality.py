"""Objective sample contracts, without rejecting difficult or tiny signals."""
import json
import unittest
import numpy as np
from Simulation_G1.noise_covariance import noise_covariance


def sample_fixture(*, snr=10., estimator='empirical', physical=False):
    rng=np.random.default_rng(12)
    fg=rng.normal(size=(5,128)); bg=.2*rng.normal(size=fg.shape)
    gain=rng.normal(size=(3,5))
    foreground=gain@fg; background=gain@bg
    for values in (foreground,background):
        values-=values.mean(axis=1,keepdims=True)
        if not physical: values-=values.mean(axis=0,keepdims=True)
    clean=foreground+background
    noise=rng.normal(size=clean.shape); baseline=rng.normal(size=(3,256))
    for values in (noise,baseline):
        values-=values.mean(axis=1,keepdims=True)
        if not physical: values-=values.mean(axis=0,keepdims=True)
    scale=np.sqrt(np.mean(clean**2)/np.mean(noise**2)/10**(snr/10))
    noise*=scale; baseline*=scale
    if physical:
        units=np.array([1e-6,1e-13,1e-11])[:,None]
        for values in (foreground,background,clean,noise,baseline): values*=units
        fg*=1e-15; bg*=1e-15
    arrays=dict(source_foreground=fg,source_background=bg,source_total=fg+bg,
                foreground_sensor=foreground,background_sensor=background,
                clean_sensor=clean,sensor_noise=noise,sensor=clean+noise,baseline_sensor=baseline)
    arrays={key:value.astype(np.float32) for key,value in arrays.items()}
    arrays['noise_covariance'],info=noise_covariance(arrays['baseline_sensor'],average_reference=not physical,estimator=estimator)
    arrays['manifest']=dict(srate=128,baseline_duration_s=2.,sensor_snr_db=None if physical else snr,
        sensor_snr_mode='global_rms',reference_mode='as_supplied' if physical else 'average',
        amplitude_mode='physical' if physical else 'relative',
        sensor_units=['V','T','T/m'] if physical else 'V',noise_covariance_info=info,
        realized_sensor_snr_reference_db=None if physical else snr)
    return arrays


class SampleQualityTests(unittest.TestCase):
    def test_zero_variance_columns_and_full_covariance_status_are_checked(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture(physical=True)
        sample['baseline_sensor'][0]=0
        matrix,info=noise_covariance(sample['baseline_sensor'],average_reference=False)
        sample['noise_covariance']=matrix
        sample['manifest']['noise_covariance_info']=info
        self.assertEqual(assess_sample(sample,full_covariance=True)['status'],'passed')
        matrix[1,0]=matrix[1,1]
        for full in (False,True):
            quality=assess_sample(sample,full_covariance=full)
            self.assertEqual(quality['status'],'failed')
            self.assertEqual(quality['checks']['noise_covariance_zero_rows']['status'],'failed')
        sample=sample_fixture()
        sample['sensor_noise_covariance']=np.zeros((2,2))
        quality=assess_sample(sample,full_covariance=True)
        self.assertEqual(quality['status'],'failed')
        self.assertEqual(quality['checks']['full_covariance']['status'],'failed')

    def test_valid_basic_and_full_covariance(self):
        from Simulation_G1.quality import assess_sample,require_sample_quality
        sample=sample_fixture()
        quality=require_sample_quality(sample)
        self.assertEqual(quality['status'],'passed')
        self.assertIs(sample['manifest']['quality'],quality)
        self.assertEqual(quality['checks']['full_covariance']['status'],'skipped')
        audited=assess_sample(sample,full_covariance=True)
        self.assertEqual(audited['status'],'passed',audited)
        json.dumps(audited,allow_nan=False)

    def test_closure_nonfinite_shapes_and_covariance_corruption_fail(self):
        from Simulation_G1.quality import assess_sample
        for key in ('source_total','clean_sensor','sensor'):
            sample=sample_fixture(); sample[key][0,0]+=.1
            self.assertEqual(assess_sample(sample)['status'],'failed',key)
        sample=sample_fixture(); sample['source_foreground'][0,0]=np.nan
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample=sample_fixture(); sample['sensor']=sample['sensor'][:,:-1]
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample=sample_fixture(); sample['noise_covariance'][0,1]*=2
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample=sample_fixture(); sample['noise_covariance']*=2
        self.assertEqual(assess_sample(sample)['status'],'failed')

    def test_mask_and_overlapping_supports_fail(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture()
        self.assertEqual(assess_sample(sample,cortex_mask=[0,1,1,1,1])['status'],'failed')
        self.assertEqual(assess_sample(sample,supports=[[0,1],[1,2]])['status'],'failed')
        self.assertEqual(assess_sample(sample,supports=[[0,1],[2,3]])['status'],'passed')

    def test_mixed_physical_units_and_tiny_source_errors(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture(physical=True)
        self.assertEqual(assess_sample(sample,full_covariance=True)['status'],'passed')
        sample['source_total'][0,0]+=1e-15
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample=sample_fixture(physical=True); sample['noise_covariance'][1,1]*=2
        self.assertEqual(assess_sample(sample)['status'],'failed')

    def test_peak_limit_increase_and_negative_snr_are_valid(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture()
        sample['sensor_noise']*=.5
        sample['sensor']=sample['clean_sensor']+sample['sensor_noise']
        sample['manifest'].update(noise_peak_limit_applied=True,noise_peak_limit_scale=.5,
            realized_sensor_snr_reference_db=10+20*np.log10(2))
        self.assertEqual(assess_sample(sample)['status'],'passed')
        sample['manifest']['realized_sensor_snr_reference_db']=10
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample=sample_fixture(snr=-10)
        quality=assess_sample(sample)
        self.assertEqual(quality['status'],'warning')
        self.assertFalse(quality['errors'])

    def test_oas_and_baseline_duration(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture(estimator='oas')
        self.assertEqual(assess_sample(sample,full_covariance=True)['status'],'passed')
        sample['manifest']['baseline_duration_s']=3
        self.assertEqual(assess_sample(sample)['status'],'failed')

    def test_zero_target_is_warning_not_failure_and_require_raises_only_fail(self):
        from Simulation_G1.quality import assess_sample,require_sample_quality
        sample=sample_fixture()
        for name in ('source_foreground','source_background','source_total','foreground_sensor','background_sensor','clean_sensor'):
            sample[name].fill(0)
        sample['sensor']=sample['sensor_noise'].copy()
        quality=require_sample_quality(sample)
        self.assertEqual(quality['status'],'warning')
        sample['sensor'][0,0]+=.1
        with self.assertRaises(ValueError): require_sample_quality(sample)

    def test_optional_truth_and_near_null_warning(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture()
        sample['manifest']['observability']={'white_near_null_energy_fraction':.3}
        self.assertEqual(assess_sample(sample)['status'],'warning')
        sample['foreground_residual']=np.full((5,128),np.nan)
        self.assertEqual(assess_sample(sample)['status'],'failed')

    def test_full_audit_respects_projector_and_catches_correlation_only_corruption(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture(physical=True)
        covariance=sample['noise_covariance']
        delta=.1*np.sqrt(covariance[0,0])*np.sqrt(covariance[1,1])
        covariance[0,1]+=delta; covariance[1,0]+=delta
        self.assertEqual(assess_sample(sample)['status'],'passed')
        self.assertEqual(assess_sample(sample,full_covariance=True)['status'],'failed')
        sample=sample_fixture(physical=True)
        projector=np.diag([1.,1.,0.])
        for name in ('foreground_sensor','background_sensor','clean_sensor','sensor_noise','sensor','baseline_sensor'):
            sample[name]=projector@sample[name]
        sample['noise_covariance'],info=noise_covariance(sample['baseline_sensor'],projector=projector,estimator='oas')
        sample['manifest']['noise_covariance_info']=info
        sample['sensor_projector']=projector
        self.assertEqual(assess_sample(sample,full_covariance=True)['status'],'passed')

    def test_source_snr_recomputed_and_area_metric_explicitly_skipped(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture()
        actual=float(10*np.log10(np.mean(sample['source_foreground'].astype(float)**2)/
                                np.mean(sample['source_background'].astype(float)**2)))
        sample['manifest'].update(source_snr_db=actual,realized_source_snr_db=actual,source_snr_metric='vertex_mean')
        self.assertEqual(assess_sample(sample)['status'],'passed')
        sample['manifest']['source_snr_db']=actual+3
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample['manifest']['source_snr_metric']='area_weighted'
        quality=assess_sample(sample)
        self.assertEqual(quality['status'],'passed')
        self.assertEqual(quality['checks']['selected_source_snr']['status'],'skipped')

    def test_online_contract_explicitly_skips_source_audit_but_checks_sensor(self):
        from Simulation_G1.quality import assess_sample
        sample=sample_fixture()
        sample['source_total'][0,0]=np.nan
        online=assess_sample(sample,full_source=False,cortex_mask=np.ones(5,dtype=bool))
        self.assertEqual(online['status'],'passed')
        self.assertEqual(online['profile'],'online_contract')
        for name in ('source_total_finite','source_closure','cortex_support','selected_source_snr'):
            self.assertEqual(online['checks'][name]['status'],'skipped')
        self.assertEqual(assess_sample(sample)['status'],'failed')
        sample['sensor'][0,0]+=.1
        self.assertEqual(assess_sample(sample,full_source=False)['status'],'failed')
        sample=sample_fixture()
        self.assertEqual(assess_sample(sample,full_source=False,cortex_mask=[0,1,1,1,1],supports=[[0]])['status'],'failed')


if __name__=='__main__':
    unittest.main()
