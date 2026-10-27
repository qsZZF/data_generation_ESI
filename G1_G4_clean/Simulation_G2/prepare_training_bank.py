"""Create 200/256 Hz training views and compare fixed-rank regional storage."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.signal import find_peaks, welch

from .nmm_bank import initialize_bank, load_bank_sample, prepare_bank_segment, read_bank, save_bank_sample, parent_group_ids


def _center(value):
    return value-value.mean(axis=1, keepdims=True)


def _relative(reference, estimate):
    denominator = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(estimate-reference))
    return difference/denominator if denominator > 0 else (0. if difference == 0 else None)


def _ids(values, n_regions):
    values = list(values)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            or v < 0 or v >= n_regions for v in values) or len(values) != len(set(values))):
        raise ValueError("Protected regions must be unique valid integer IDs")
    return sorted(map(int, values))


def _event_regions(source, rate, threshold):
    """Conservative positive PSP candidates, including elevated record boundaries."""
    counts, boundary = {}, []
    margin = max(1, int(np.ceil(.05*rate)))
    for region, trace in enumerate(source):
        peaks, _ = find_peaks(trace, height=np.nextafter(threshold, np.inf),
                              distance=int(np.floor(.2*rate))+1)
        if peaks.size:
            counts[str(region)] = int(peaks.size)
        if max(float(trace[:margin].max()), float(trace[-margin:].max())) > threshold:
            boundary.append(region)
    return sorted(set(map(int, counts)) | set(boundary)), counts, boundary


def select_protected(native, resampled, entry, rate, *, policy="auto", manual=(), threshold=8.):
    if policy not in {"auto", "configured", "events"} or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid protection policy or positive peak threshold")
    configured = _ids(entry["active_regions"], len(native))
    inherited = _ids(entry.get("storage", {}).get("protected_regions", []), len(native))
    manual = _ids(manual, len(native))
    eligible = entry["source_unit"] == "mV_PSP"
    if policy == "events" and not eligible:
        raise ValueError("Event threshold is in mV PSP; use configured protection for other source units")
    use_events = policy != "configured" and eligible
    native_ids, native_counts, native_edges = _event_regions(native, entry["native_srate_hz"], threshold) if use_events else ([], {}, [])
    output_ids, output_counts, output_edges = _event_regions(resampled, rate, threshold) if use_events else ([], {}, [])
    protected = sorted(set(configured) | set(inherited) | set(manual) | set(native_ids) | set(output_ids))
    return protected, dict(policy=policy, configured_regions=configured, manual_regions=manual,
        inherited_regions=inherited,
        protected_regions=protected, event_detection_applied=use_events,
        event_rule="positive local peak > threshold, separation >200ms; also protect >threshold in first/last50ms",
        event_threshold_mV=threshold if use_events else None,
        event_skip_reason=None if use_events else "configured-only policy or non-PSP units",
        native_candidate_counts=native_counts, output_candidate_counts=output_counts,
        native_boundary_regions=native_edges, output_boundary_regions=output_edges,
        parameter_preset_is_activation_truth=False)


def _power(value, metric=None):
    if metric is None:
        return float(np.mean(value**2))
    return float(np.mean(np.sum(value*(metric@value), axis=0)))


def _resampling_peaks(native, output, native_rate, output_rate, regions):
    rows=[]
    for region in regions:
        before=_center(native[[region]].astype(float))[0]
        after=_center(output[[region]].astype(float))[0]
        i,j=int(np.argmax(np.abs(before))),int(np.argmax(np.abs(after)))
        peak=float(abs(before[i]))
        rows.append(dict(region=region,native_centered_abs_peak=peak,
            output_centered_abs_peak=float(abs(after[j])),
            relative_peak_amplitude_change=float(abs(after[j])/peak-1) if peak>0 else None,
            dominant_peak_time_shift_ms=float(1000*(j/output_rate-i/native_rate))))
    return dict(rule='maximum absolute temporally demeaned sample; dominant event may change between repeated peaks',regions=rows)


def comparison_metrics(reference, decoded, rate, protected, *, forward=None, source_metric=None,
                       source_snrs=(10., 20., 30.)):
    protected = np.asarray(protected, dtype=int)
    background = np.setdiff1d(np.arange(len(reference)), protected)
    ref = _center(reference.astype(float)); out = _center(decoded.astype(float))
    foreground = np.zeros_like(ref); foreground[protected] = ref[protected]
    bg = ref.copy(); bg[protected] = 0
    estimate = out.copy(); estimate[protected] = 0
    report = dict(protected_max_abs_error=float(np.max(np.abs(decoded[protected]-reference[protected]))) if protected.size else 0.,
        source_background_relative_error=_relative(bg, estimate), source_total_relative_error=_relative(ref, out),
        source_background_power_ratio=(_power(estimate)/_power(bg)) if _power(bg)>0 else None,
        sensor_qc_available=forward is not None)
    if background.size:
        f, original_psd = welch(ref[background], fs=rate, nperseg=min(ref.shape[1], int(2*rate)), axis=-1)
        _, estimated_psd = welch(out[background], fs=rate, nperseg=min(ref.shape[1], int(2*rate)), axis=-1)
        band = (f>=1)&(f<=min(80.,rate/2))
        denominator = float(original_psd[:,band].sum())
        report['background_psd_relative_l1_1_80Hz'] = (float(np.abs(estimated_psd[:,band]-original_psd[:,band]).sum())/denominator
                                                     if denominator>0 else None)
    if forward is None:
        return report
    sensor_bg, sensor_estimate, sensor_fg = forward@bg, forward@estimate, forward@foreground
    singular = np.linalg.svd(sensor_bg, compute_uv=False)
    energy = singular**2
    cumulative = np.cumsum(energy)/energy.sum() if energy.sum()>0 else np.zeros_like(energy)
    covariance = sensor_bg@sensor_bg.T/bg.shape[1]
    report.update(sensor_background_relative_error=_relative(sensor_bg,sensor_estimate),
        sensor_total_relative_error=_relative(forward@ref,forward@out),
        sensor_background_covariance_relative_error=_relative(covariance,sensor_estimate@sensor_estimate.T/bg.shape[1]),
        sensor_background_rank_99pct=int(np.searchsorted(cumulative,.99)+1) if energy.sum()>0 else 0)
    f, psd = welch(sensor_bg,fs=rate,nperseg=min(ref.shape[1],int(2*rate)),axis=-1)
    _, estimated_psd = welch(sensor_estimate,fs=rate,nperseg=min(ref.shape[1],int(2*rate)),axis=-1)
    mask=(f>=1)&(f<=min(80.,rate/2)); total=float(psd[:,mask].sum())
    report['sensor_background_psd_relative_l1_1_80Hz'] = float(np.abs(psd[:,mask]-estimated_psd[:,mask]).sum())/total if total>0 else None
    pfg,pbg,pestimate = _power(foreground,source_metric),_power(bg,source_metric),_power(estimate,source_metric)
    report['source_snr_metric'] = 'whole-cortex area-weighted projected source power' if source_metric is not None else 'mean regional power'
    report['source_snr_context'] = []
    report['source_snr_context_status'] = 'unavailable_without_nonzero_foreground_background_and_sensor_foreground'
    if pfg>0 and pbg>0 and np.linalg.norm(sensor_fg)>0:
        report['source_snr_context_status'] = 'available'
        for snr in source_snrs:
            gain = np.sqrt(pfg/pbg/10**(snr/10))
            recalibrated = np.sqrt(pfg/pestimate/10**(snr/10)) if pestimate>0 else None
            report['source_snr_context'].append(dict(source_snr_db=float(snr),
                same_gain_sensor_error_relative_to_foreground=float(np.linalg.norm(gain*(sensor_estimate-sensor_bg))/np.linalg.norm(sensor_fg)),
                recalibrated_sensor_error_relative_to_foreground=(float(np.linalg.norm(recalibrated*sensor_estimate-gain*sensor_bg)/np.linalg.norm(sensor_fg))
                                                                 if recalibrated is not None else None)))
    return report


def convert_bank(input_dir, output_dir, *, sample_ids=None, rate=256., codecs=("dense","int16","lowrank"),
                 ranks=(8,16,32), protection="auto", protected_regions=(), spike_threshold=8.,
                 forward=None, source_metric=None, forward_metadata=None):
    """Write new schema2 banks; fixed-k codecs are evaluated, never silently rejected."""
    original=Path(input_dir).resolve(); destination=Path(output_dir).resolve()
    bank=read_bank(original)
    if rate not in (200.,256.):
        raise ValueError("Training view rate must be 200 or 256 Hz")
    codecs=list(codecs); ranks=list(ranks)
    if not codecs or len(set(codecs))!=len(codecs) or set(codecs)-{'dense','int16','lowrank'}:
        raise ValueError("Choose unique dense/int16/lowrank codecs")
    if 'lowrank' in codecs and (not ranks or len(set(ranks))!=len(ranks) or
            any(isinstance(k,bool) or not isinstance(k,int) or k<1 for k in ranks)):
        raise ValueError("Lowrank ranks must be unique positive integers")
    if protection not in {'auto','configured','events'} or not np.isfinite(spike_threshold) or spike_threshold<=0:
        raise ValueError("Invalid protection policy/threshold")
    protected_regions=_ids(protected_regions,bank['n_regions'])
    samples=sorted(bank['samples']) if sample_ids is None else list(sample_ids)
    if not samples or len(set(samples))!=len(samples) or any(s not in bank['samples'] for s in samples):
        raise ValueError("Select distinct existing bank samples")
    for sample in samples:
        entry=bank['samples'][sample]
        count=entry['n_samples']*rate/entry['native_srate_hz']
        if rate>entry['native_srate_hz'] or count<2 or not np.isclose(count,round(count),rtol=0,atol=1e-6):
            raise ValueError("Each record must support downsampling to a whole number of at least two samples")
    if forward is not None:
        forward=np.asarray(forward,dtype=float)
        if forward.ndim!=2 or min(forward.shape)<1 or forward.shape[1]!=bank['n_regions'] or not np.isfinite(forward).all():
            raise ValueError("Forward must be finite sensors-by-regions in bank order")
    if source_metric is not None:
        if source_metric.shape!=(bank['n_regions'],bank['n_regions']):
            raise ValueError("Source metric region shape mismatch")
    destination.mkdir(parents=True,exist_ok=False)
    variants=[(codec,k) for codec in codecs for k in (ranks if codec=='lowrank' else [None])]
    outputs={}
    for codec,k in variants:
        label=codec if k is None else f'lowrank_k{k:03d}'
        path=destination/label
        initialize_bank(path,atlas_id=bank['atlas_id'],n_regions=bank['n_regions'],
                        projection_sha256=bank['projection_sha256'],schema_version=2)
        outputs[label]=path
    code_files=[Path(__file__),Path(__file__).with_name('nmm_codec.py'),Path(__file__).with_name('nmm_bank.py')]
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    report=dict(input_bank=str(original),output_rate_hz=rate,scipy_version=scipy.__version__,numpy_version=np.__version__,code_sha256=code_hashes,
                automatic_codec_selection=False,forward=forward_metadata,records=[])
    for sample in samples:
        native,clock,parent=load_bank_sample(original,sample)
        duration=native.shape[1]/parent['native_srate_hz']
        reference=prepare_bank_segment(native,clock,srate=rate,duration_s=duration).astype(np.float32)
        ids,protection_meta=select_protected(native,reference,parent,rate,policy=protection,
                                            manual=protected_regions,threshold=spike_threshold)
        ratio=Fraction(float(rate/parent['native_srate_hz'])).limit_denominator(1000000)
        conversion=dict(method='scipy.signal.resample_poly',up=ratio.numerator,down=ratio.denominator,
            window=['kaiser',5.],padtype='line',scope='whole_recording_before_cropping',
            original_output_srate_hz=parent['native_srate_hz'],output_srate_hz=rate,
            original_source_sha256=parent['source_sha256'],scipy_version=scipy.__version__,numpy_version=np.__version__,
            code_sha256=code_hashes,
            peak_qc=_resampling_peaks(native,reference,parent['native_srate_hz'],rate,ids))
        manifest={**parent['manifest'],'output_srate_hz':rate,'n_regions_output':len(reference),
                  'region_source_shape':list(reference.shape),'time_shape':[reference.shape[1]],
                  'training_view':conversion,'foreground_protection':protection_meta}
        provenance=dict(parent_bank=str(original),parent_sample_id=sample,parent_source_sha256=parent['source_sha256'],
                        parent_simulation_group_id=parent['simulation_group_id'],original_provenance=parent['provenance'],
                        parent_data_group_ids=parent_group_ids(parent),
                        conversion=conversion)
        for codec,k in variants:
            label=codec if k is None else f'lowrank_k{k:03d}'
            storage=dict(codec=codec,protected_regions=ids)
            if k is not None: storage['rank']=k
            start=time.perf_counter()
            entry=save_bank_sample(outputs[label],sample,reference,np.arange(reference.shape[1])*1000/rate,
                                  manifest,provenance,storage=storage,simulation_group_id=parent['simulation_group_id'])
            encode_seconds=time.perf_counter()-start
            start=time.perf_counter(); decoded,_,_=load_bank_sample(outputs[label],sample)
            decode_seconds=time.perf_counter()-start
            metrics=comparison_metrics(reference,decoded,rate,ids,forward=forward,source_metric=source_metric)
            if metrics['protected_max_abs_error']!=0:
                raise AssertionError("Protected foreground differs from canonical downsampled float32 reference")
            size=(outputs[label]/entry['source_file']).stat().st_size
            report['records'].append(dict(sample_id=sample,codec=label,n_regions=len(reference),n_samples=reference.shape[1],
                protected_regions=ids,storage=entry['storage'],file_bytes=size,raw_float32_bytes=reference.nbytes,
                encode_seconds=encode_seconds,decode_seconds=decode_seconds,metrics=metrics))
    dense_bytes={r['sample_id']:r['file_bytes'] for r in report['records'] if r['codec']=='dense'}
    for row in report['records']:
        row['size_fraction_of_dense_npz']=(row['file_bytes']/dense_bytes[row['sample_id']] if row['sample_id'] in dense_bytes else None)
    (destination/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (destination/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['sample_id','codec','file_bytes','size_fraction_of_dense_npz','protected_count','encode_seconds','decode_seconds',
                'source_background_relative_error','sensor_background_relative_error','sensor_total_relative_error','sensor_background_rank_99pct']
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in report['records']:
            flat={**row,**row['metrics'],'protected_count':len(row['protected_regions'])}
            writer.writerow({key:flat.get(key) for key in fields})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-bank',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sample-ids',nargs='+')
    parser.add_argument('--rate',type=float,choices=[200.,256.],default=256.)
    parser.add_argument('--codecs',nargs='+',choices=['dense','int16','lowrank'],default=['dense','int16','lowrank'])
    parser.add_argument('--ranks',nargs='+',type=int,default=[8,16,32])
    parser.add_argument('--protection',choices=['auto','configured','events'],default='auto')
    parser.add_argument('--protect-regions',nargs='*',type=int,default=[])
    parser.add_argument('--spike-threshold-mv',type=float,default=8.)
    parser.add_argument('--projection',type=Path)
    parser.add_argument('--leadfield',type=Path)
    parser.add_argument('--cortex',type=Path,help='Optional mesh for area-weighted source-SNR context')
    parser.add_argument('--no-average-reference',action='store_true',help='Use for MEG or explicitly supplied reference')
    args=parser.parse_args()
    if bool(args.projection)!=bool(args.leadfield) or (args.cortex and not args.projection):
        parser.error('Supply projection and leadfield together; cortex additionally supplies area weights')
    forward,metric,identity=None,None,None
    if args.projection:
        from .g2_simulation_core import load_projection_matrix,load_brainstorm_leadfield,load_brainstorm_cortex
        from .nmm_bank import projection_hash
        from Simulation_G1.g1_simulation_core import normalize_cortex,mesh_vertex_areas
        projection=load_projection_matrix(args.projection)
        if projection_hash(projection)!=read_bank(args.input_bank)['projection_sha256']:
            raise ValueError('Projection differs from the input bank')
        leadfield=load_brainstorm_leadfield(args.leadfield,n_vertices=projection.shape[0])
        forward=leadfield@projection
        if not args.no_average_reference: forward-=forward.mean(axis=0,keepdims=True)
        identity=dict(projection=str(args.projection),projection_sha256=projection_hash(projection),
            leadfield=str(args.leadfield),leadfield_sha256=hashlib.sha256(args.leadfield.read_bytes()).hexdigest(),
            average_reference=not args.no_average_reference,n_sensors=forward.shape[0])
        if args.cortex:
            cortex=load_brainstorm_cortex(args.cortex)
            vertices,faces,_=normalize_cortex(cortex['Vertices'],cortex['Faces'],cortex['VertConn'])
            _,areas=mesh_vertex_areas(vertices,faces)
            if len(areas)!=projection.shape[0] or areas.sum()<=0: raise ValueError('Cortex/projection mismatch')
            metric=projection.T@sparse.diags(areas/areas.sum())@projection
            identity.update(cortex=str(args.cortex),cortex_sha256=hashlib.sha256(args.cortex.read_bytes()).hexdigest())
    result=convert_bank(args.input_bank,args.output_dir,sample_ids=args.sample_ids,rate=args.rate,codecs=args.codecs,
        ranks=args.ranks,protection=args.protection,protected_regions=args.protect_regions,spike_threshold=args.spike_threshold_mv,
        forward=forward,source_metric=metric,forward_metadata=identity)
    print(json.dumps(dict(output_dir=str(args.output_dir),views=len(result['records']),sensor_qc=forward is not None),indent=2))


if __name__=='__main__':
    main()
