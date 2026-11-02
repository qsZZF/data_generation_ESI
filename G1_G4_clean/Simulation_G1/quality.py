"""Small objective QC for named simulation arrays; difficulty is not failure.

Basic checks avoid covariance eigendecomposition/recomputation. Relative errors
are assessed per row/channel, so tiny physical units cannot hide behind EEG
amplitudes or an arbitrary absolute tolerance. Full covariance audit is opt-in.
"""
from collections.abc import Mapping, MutableMapping
import numpy as np


def _row_error(total, first, second=None):
    """Peak relative closure error per row, bounded working memory."""
    if (second is None or not np.any(second)) and np.array_equal(total,first):
        return 0.
    worst=0.
    dtype=np.result_type(total.dtype,first.dtype,np.float32 if second is None else second.dtype,np.float32)
    for start in range(0,len(total),1024):
        output=np.asarray(total[start:start+1024],dtype=dtype)
        a=np.asarray(first[start:start+1024],dtype=dtype)
        b=0. if second is None else np.asarray(second[start:start+1024],dtype=dtype)
        scale=np.maximum(np.max(np.abs(output),axis=1),np.max(np.abs(a),axis=1))
        if second is not None:
            scale=np.maximum(scale,np.max(np.abs(b),axis=1))
        limits=np.finfo(dtype)
        ordinary=(scale==0)|((scale>=64*limits.tiny)&(scale<=limits.max/4))
        if np.all(ordinary):
            error=output-a
            if second is not None: error-=b
            peak=np.max(np.abs(error),axis=1)
            ratio=np.divide(peak,scale,out=np.zeros_like(peak),where=scale>0)
            worst=max(worst,float(ratio.max(initial=0)))
        else:
            # Rare subnormal/extreme rows are normalized BEFORE subtraction:
            # this keeps tiny physical units visible and prevents overflow.
            safe=np.where(scale>0,scale,1.)[:,None]
            error=output/safe-a/safe
            if second is not None: error-=b/safe
            worst=max(worst,float(np.max(np.abs(error),initial=0)))
    return worst


def _roundoff(*arrays):
    eps=[np.finfo(a.dtype).eps for a in arrays if a.dtype.kind=='f']
    # A float64 array may be an upcast of the float32 export or its projection.
    # Preserve the export's relative precision budget without an absolute floor.
    return max(16*np.finfo(np.float32).eps,16*max(eps,default=np.finfo(float).eps))


def assess_sample(sample, *, cortex_mask=None, supports=None, full_covariance=False, full_source=True):
    """Return status/errors/warnings/metrics/checks without modifying the sample.

    Zero targets, zero-noise physical samples, low SNR and reported near-null
    energy are legitimate conditions. Only violated numerical/data contracts
    fail. full_source=False is the explicitly limited online contract: source
    shapes/support indices remain checked, but source values, closure, cortical
    leakage and source-SNR recomputation are skipped. Sensor arrays always get
    numerical checks. Full covariance audit is separately opt-in.
    """
    errors,warnings,metrics,checks=[],[],{},{}
    def check(name,passed,reason):
        checks[name]={'status':'passed' if passed else 'failed'}
        if not passed:
            checks[name]['reason']=reason
            errors.append(reason)
    def skip(name,reason):
        checks[name]={'status':'skipped','reason':reason}
    def result():
        return dict(status='failed' if errors else 'warning' if warnings else 'passed',
                    profile='online_contract' if not full_source else 'full_audit' if full_covariance else 'source_and_sensor_contract',
                    errors=errors,warnings=warnings,metrics=metrics,checks=checks)
    if not isinstance(sample,Mapping):
        check('sample_mapping',False,'sample must contain named arrays and a manifest')
        return result()
    meta=sample.get('manifest',{})
    if not isinstance(meta,Mapping):
        check('manifest',False,'manifest must be a mapping')
        return result()
    config=meta.get('config',meta)
    if not isinstance(config,Mapping): config={}
    baseline_meta=meta.get('baseline',{})
    if not isinstance(baseline_meta,Mapping): baseline_meta={}
    fields=('source_foreground','source_background','source_total','foreground_sensor',
            'background_sensor','clean_sensor','sensor_noise','sensor','baseline_sensor','noise_covariance')
    arrays={}
    source_fields={'source_foreground','source_background','source_total'}
    for name in fields:
        value=sample.get(name,sample.get('source') if name=='source_total' else None)
        valid=isinstance(value,np.ndarray) and value.ndim==2 and min(value.shape)>0 and value.dtype.kind in 'fiu'
        check(name+'_shape',valid,f'{name} must be a nonempty real two-dimensional NumPy array')
        if not valid: continue
        if not full_source and name in source_fields:
            skip(name+'_finite','Online contract: source values require a full-source audit')
            arrays[name]=value
            continue
        finite=bool(np.isfinite(value).all())
        check(name+'_finite',finite,f'{name} contains nonfinite values')
        if finite: arrays[name]=value
    checked_ids={id(value) for value in arrays.values()}
    for name,value in sample.items():
        if isinstance(value,np.ndarray) and id(value) not in checked_ids and value.dtype.kind in 'fciu':
            if not full_source and (name.startswith(('source','region_')) or name.endswith(('_coarse','_residual'))):
                skip(name+'_finite','Online contract: source component values require a full-source audit')
                continue
            check(name+'_finite',not np.iscomplexobj(value) and bool(np.isfinite(value).all()),
                  f'{name} contains complex or nonfinite values')
            checked_ids.add(id(value))
    if len(arrays)!=len(fields): return result()
    n,t=arrays['source_total'].shape
    c,event_t=arrays['sensor'].shape
    check('event_length',t==event_t,'Source and sensor event lengths differ')
    for name in ('source_foreground','source_background'):
        check(name+'_dimensions',arrays[name].shape==(n,t),f'{name} dimensions differ from source_total')
    for name in ('foreground_sensor','background_sensor','clean_sensor','sensor_noise'):
        check(name+'_dimensions',arrays[name].shape==(c,event_t),f'{name} dimensions differ from sensor')
    check('baseline_channels',arrays['baseline_sensor'].shape[0]==c and arrays['baseline_sensor'].shape[1]>=2,
          'Baseline channels differ or baseline has fewer than two samples')
    check('covariance_dimensions',arrays['noise_covariance'].shape==(c,c),'Noise covariance shape must match sensor channels')
    if errors: return result()
    for name,out,a,b in (('source_closure','source_total','source_foreground','source_background'),
                         ('clean_sensor_closure','clean_sensor','foreground_sensor','background_sensor'),
                         ('sensor_closure','sensor','clean_sensor','sensor_noise')):
        if name=='source_closure' and not full_source:
            skip(name,'Online contract: full-source closure not recomputed')
            continue
        error=_row_error(arrays[out],arrays[a],arrays[b])
        metrics[name+'_max_row_relative_error']=error
        check(name,error<=_roundoff(arrays[out],arrays[a],arrays[b]),f'{name} violates component additivity')
    if full_source and not np.any(arrays['source_foreground']): warnings.append('Foreground target is zero; this is not an automatic rejection')
    if not np.any(arrays['foreground_sensor']):
        warnings.append('Foreground sensor projection is zero; zero/null-space targets are legitimate')
    mask=None; mask_valid=False
    if cortex_mask is not None:
        mask=np.asarray(cortex_mask)
        valid=mask.shape==(n,) and np.isin(mask,[0,1]).all()
        mask_valid=bool(valid)
        check('cortex_mask_shape',bool(valid),'Cortex mask must be a binary vector matching source rows')
        if valid and full_source:
            outside=~mask.astype(bool)
            count=sum(np.count_nonzero(arrays[name][outside]) for name in ('source_foreground','source_background'))
            metrics['outside_cortex_nonzero_values']=int(count)
            check('cortex_support',count==0,'Source activity occurs outside the declared cortex mask')
        elif valid:
            skip('cortex_support','Online contract: source leakage outside cortex is not scanned')
    else: skip('cortex_support','No cortical mask supplied')
    if supports is None and meta.get('patches'):
        supports=[patch.get('vertices',[]) for patch in meta['patches']]
    if supports is not None:
        seen=set(); valid=True; overlap=0; outside_count=0
        for support in supports:
            indices=np.asarray(support)
            if indices.ndim!=1 or not len(indices) or indices.dtype.kind not in 'iu' or np.any((indices<0)|(indices>=n)):
                valid=False; break
            current=set(indices.tolist())
            overlap+=len(seen.intersection(current))
            valid &= len(current)==len(indices)
            if mask_valid:
                outside_count+=int(np.count_nonzero(~mask[indices].astype(bool)))
            seen.update(current)
        metrics['overlapping_support_vertices']=overlap
        check('support_indices',bool(valid),'Foreground supports contain invalid or duplicate vertex indices')
        check('disjoint_supports',overlap==0,'Foreground supports overlap')
        if mask is not None:
            check('support_cortex_indices',outside_count==0,'Declared foreground supports include vertices outside the cortex mask')
    else: skip('disjoint_supports','No foreground patch supports supplied')

    rate=config.get('srate',meta.get('sampling_rate_hz',baseline_meta.get('srate_hz')))
    duration=config.get('baseline_duration_s',baseline_meta.get('duration_s',baseline_meta.get('baseline_duration_s',2.)))
    if rate is not None:
        try:
            from .noise_covariance import baseline_sample_count
        except ImportError:
            from noise_covariance import baseline_sample_count
        try:
            expected=baseline_sample_count(duration,rate)
            metrics['baseline_samples_expected']=expected
            check('baseline_duration',arrays['baseline_sensor'].shape[1]==expected,'Baseline duration differs from recorded sampling rate/duration')
        except (TypeError,ValueError) as error:
            check('baseline_duration',False,f'Invalid baseline rate/duration: {error}')
    else: skip('baseline_duration','Sampling rate unavailable; cannot verify duration')
    average=meta.get('reference_mode',meta.get('noise_covariance_info',{}).get('reference'))=='average'
    units=meta.get('sensor_units',config.get('sensor_units'))
    mixed=isinstance(units,(list,tuple)) and len(set(units))>1
    if average and mixed:
        check('reference_units',False,'Average reference cannot mix different declared sensor units')
    if average:
        for name in ('sensor','clean_sensor','sensor_noise','baseline_sensor'):
            value=np.asarray(arrays[name],dtype=float)
            denominator=np.sum(np.abs(value),axis=0)
            ratio=np.divide(np.abs(np.sum(value,axis=0)),denominator,out=np.zeros_like(denominator),where=denominator>0)
            error=float(ratio.max(initial=0))
            metrics[name+'_car_relative_error']=error
            check(name+'_car',error<=max(1e-7,_roundoff(arrays[name])),f'{name} does not preserve common-average reference')

    covariance_fields=['noise_covariance']+[name for name in ('oracle_noise_covariance','sensor_noise_covariance') if name in sample]
    projector=sample.get('sensor_projector',meta.get('sensor_projector',config.get('sensor_projector')))
    for name in covariance_fields:
        matrix=sample[name]
        valid=isinstance(matrix,np.ndarray) and matrix.shape==(c,c) and matrix.dtype.kind in 'fiu' and np.isfinite(matrix).all()
        check(name+'_matrix',bool(valid),f'{name} must be finite and match sensor dimensions')
        if not valid: continue
        matrix=np.asarray(matrix,dtype=float); diagonal=np.diag(matrix)
        check(name+'_diagonal',bool(np.all(diagonal>=0)),f'{name} has a negative variance')
        if np.any(diagonal<0): continue
        positive=diagonal>0
        check(name+'_zero_rows',not (np.any(matrix[~positive]) or np.any(matrix[:,~positive])),
              f'{name} has nonzero covariance at a zero-variance channel')
        if np.any(positive):
            std=np.sqrt(diagonal[positive])
            standardized=matrix[np.ix_(positive,positive)]/std[:,None]/std[None,:]
            symmetry=float(np.max(np.abs(standardized-standardized.T)))
            metrics[name+'_standardized_symmetry_error']=symmetry if np.isfinite(symmetry) else None
            check(name+'_symmetry',np.isfinite(symmetry) and symmetry<=1e-7,f'{name} is not symmetric in channel-standardized units')
            check(name+'_pairwise_bound',bool(np.max(np.abs(standardized))<=1+1e-6),f'{name} violates the covariance Cauchy-Schwarz bound')
        if average:
            denom=np.sum(np.abs(matrix),axis=1)
            error=float(np.max(np.divide(np.abs(matrix.sum(axis=1)),denom,out=np.zeros_like(denom),where=denom>0)))
            check(name+'_car',error<=1e-6,f'{name} does not preserve the removed average-reference direction')
        info=meta.get(name+'_info',{})
        expected_samples=arrays['baseline_sensor'].shape[1] if name=='noise_covariance' else event_t
        if 'n_samples' in info:
            check(name+'_sample_count',info['n_samples']==expected_samples,f'{name} recorded sample count differs from its estimation window')
        estimator=info.get('estimator',config.get('covariance_estimator','empirical'))
        estimator={'empirical_centered':'empirical','oas_standardized_subspace':'oas'}.get(estimator,estimator)
        if name=='noise_covariance' and estimator=='empirical' and not info.get('projection_applied',False):
            expected_variance=np.var(arrays['baseline_sensor'].astype(float),axis=1,ddof=1)
            denominator=np.maximum(diagonal,expected_variance)
            error=float(np.max(np.divide(np.abs(diagonal-expected_variance),denominator,
                out=np.zeros_like(denominator),where=denominator>0)))
            metrics['baseline_variance_relative_error']=error
            check('baseline_variance',error<=1e-6,'Noise covariance diagonal does not match baseline variance')
        if full_covariance:
            if info.get('projector_supplied',False) and projector is None:
                skip(name+'_recompute','Recorded projector was not supplied; audit cannot reproduce estimator')
                warnings.append(f'{name}: full audit missing the recorded projector')
                continue
            try:
                from .noise_covariance import noise_covariance
            except ImportError:
                from noise_covariance import noise_covariance
            data=(arrays['baseline_sensor'] if name=='noise_covariance' else arrays['sensor_noise'] if name=='sensor_noise_covariance'
                  else arrays['background_sensor'].astype(float)+arrays['sensor_noise'])
            try:
                expected,_=noise_covariance(data,average_reference=average,estimator=estimator,projector=projector)
                scales=np.sqrt(np.maximum(np.diag(expected),diagonal))
                scales=np.where(scales>0,scales,1.)
                error=float(np.max(np.abs((matrix-expected)/scales[:,None]/scales[None,:])))
                metrics[name+'_recompute_standardized_error']=error if np.isfinite(error) else None
                check(name+'_recompute',np.isfinite(error) and error<=2e-5,f'{name} differs from its recorded baseline/event estimator')
            except (TypeError,ValueError,np.linalg.LinAlgError) as error:
                check(name+'_recompute',False,f'{name} covariance audit failed: {error}')
    if full_covariance:
        failed=any(value['status']=='failed' and (name.startswith(tuple(covariance_fields))
                   or name=='baseline_variance') for name,value in checks.items())
        complete=all(checks.get(name+'_recompute',{}).get('status')=='passed' for name in covariance_fields)
        checks['full_covariance']={'status':'failed' if failed else 'passed' if complete else 'skipped'}
    else:
        skip('full_covariance','Disabled: no full estimator reproduction or eigendecomposition in basic QC')
    skip('covariance_psd','No separate eigendecomposition; full audit compares to the trusted covariance estimator' if full_covariance
         else 'Basic symmetry/diagonal/pairwise bounds do not prove PSD')

    source_requested=meta.get('requested_source_snr_db',config.get('source_snr_db'))
    source_metric=meta.get('source_snr_metric',config.get('source_snr_metric','vertex_mean'))
    if not full_source:
        skip('selected_source_snr','Online contract: source-domain power ratio not recomputed')
        recorded=meta.get('realized_source_snr_db')
        if source_requested is not None and recorded is not None:
            check('recorded_source_snr_request',bool(np.isfinite(source_requested) and np.isfinite(recorded)
                  and abs(source_requested-recorded)<=.002),'Recorded source SNR and requested source SNR disagree')
    elif source_requested is None:
        skip('selected_source_snr','No source-domain SNR requested')
    elif source_metric=='area_weighted':
        skip('selected_source_snr','Area-weighted source SNR requires vertex-area resources; not available in the sample contract')
    elif source_metric!='vertex_mean':
        check('selected_source_snr',False,f'Unknown source SNR metric {source_metric!r}')
    else:
        fg=arrays['source_foreground']; bg=arrays['source_background']
        a=float(np.einsum('ij,ij->',fg,fg,dtype=np.float64)/fg.size)
        b=float(np.einsum('ij,ij->',bg,bg,dtype=np.float64)/bg.size)
        if a<=0:
            warnings.append('Zero foreground energy: requested source SNR is undefined')
            skip('selected_source_snr','Zero foreground energy')
        elif b<=0:
            check('selected_source_snr',False,'Source SNR requested but source background is zero')
        else:
            source_snr=float(10/np.log(10)*(np.log(a)-np.log(b)))
            metrics['vertex_mean_source_snr_db']=source_snr
            check('selected_source_snr',bool(np.isfinite(source_requested) and abs(source_snr-source_requested)<=.002),
                  'Source SNR differs from the exported foreground/background component ratio')
            recorded=meta.get('realized_source_snr_db')
            if recorded is not None:
                check('recorded_source_snr',bool(np.isfinite(recorded) and abs(source_snr-recorded)<=.002),
                      'Recorded source SNR differs from exported components')
            if source_snr<0: warnings.append('Source background power exceeds foreground power; low source SNR remains valid')

    requested=meta.get('requested_sensor_snr_db',config.get('sensor_snr_db'))
    mode=meta.get('sensor_snr_mode',config.get('sensor_snr_mode','global_rms'))
    if mixed:
        skip('selected_sensor_snr','Mixed sensor units have no raw pooled-power SNR')
    else:
        clean=arrays['clean_sensor'].astype(float); noise=arrays['sensor_noise'].astype(float)
        clip=float(config.get('sensor_snr_clip_quantile',.95))
        if mode=='global_rms': signal_power=float(np.mean(clean**2))
        elif mode=='robust_global_rms':
            limit=np.quantile(np.abs(clean),np.clip(clip,.5,.999)); signal_power=float(np.mean(np.clip(clean,-limit,limit)**2))
        elif mode=='active_rms':
            gfp=np.sqrt(np.mean(clean**2,axis=0)); active=gfp>=np.quantile(gfp,np.clip(clip,.5,.999))
            signal_power=float(np.mean(clean[:,active]**2))
        else:
            signal_power=None; check('selected_sensor_snr',False,f'Unknown selected sensor SNR mode {mode!r}')
        noise_power=float(np.mean(noise**2))
        if signal_power is not None and signal_power>0 and noise_power>0:
            realized=float(10/np.log(10)*(np.log(signal_power)-np.log(noise_power)))
            metrics['selected_sensor_snr_db']=realized
            if realized<0: warnings.append('Selected sensor SNR is below 0 dB; difficult samples remain valid')
            recorded=meta.get('realized_sensor_snr_reference_db',meta.get('realized_sensor_snr_db') if mode=='global_rms' else None)
            if recorded is not None:
                check('recorded_sensor_snr',bool(np.isfinite(recorded) and abs(recorded-realized)<=.002),
                      'Recorded selected SNR differs from the exported sensor components')
            if requested is not None:
                scale=meta.get('noise_peak_limit_scale',1.)
                valid=np.isscalar(scale) and np.isfinite(scale) and 0<scale<=1
                check('peak_limit_scale',bool(valid),'Noise peak-limit gain must be in (0,1]')
                if valid:
                    expected=float(requested)-20*np.log10(scale)
                    metrics['expected_selected_sensor_snr_db']=float(expected)
                    check('selected_sensor_snr',bool(np.isfinite(expected) and abs(realized-expected)<=.002),
                          'Selected SNR does not match request plus recorded noise peak limit')
        elif signal_power is not None:
            metrics['selected_sensor_snr_db']=None
            if signal_power<=0:
                warnings.append('Zero projected clean signal: finite SNR is undefined, not an automatic failure')
                skip('selected_sensor_snr','Zero clean signal')
            elif requested is None:
                warnings.append('No measurement noise: SNR/whitening may be undefined')
                skip('selected_sensor_snr','Noise disabled')
            else: check('selected_sensor_snr',False,'Finite sensor SNR requested but measurement noise is zero')
    observability=meta.get('observability',{})
    if isinstance(observability,Mapping):
        near=observability.get('white_near_null_energy_fraction')
        if near is not None and np.isfinite(near) and near>0:
            metrics['reported_near_null_energy_fraction']=float(near)
            warnings.append('Reported near-null target energy is legitimate; QC does not reject inverse difficulty')
    return result()


def require_sample_quality(sample, **kwargs):
    """Attach the JSON report and raise only on objective contract violations."""
    quality=assess_sample(sample,**kwargs)
    if isinstance(sample,MutableMapping):
        manifest=sample.setdefault('manifest',{})
        if isinstance(manifest,MutableMapping): manifest['quality']=quality
    if quality['status']=='failed':
        raise ValueError('Sample quality failed: '+'; '.join(quality['errors']))
    return quality
