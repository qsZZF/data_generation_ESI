"""Small, reproducible spectral-generator comparisons; real subjects stay held out.

This is a descriptive pilot, not an ESI accuracy or physiological validity claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

from Simulation_G1.geometry import load_basis, project_maps
from Simulation_G1.g1_simulation_core import G1SourceSimulator
from Simulation_G1.noise_covariance import sensor_noise_with_baseline
from Simulation_G1.spectral import generate_coefficients
from Simulation_G1.spectral_metrics import resting_features, evoked_features
from Simulation_G2.g2_simulation_core import load_brainstorm_cortex, load_brainstorm_leadfield
from Simulation_G4.atlas import load_atlas
from Simulation_G4.pipeline import channel_key

RATE = 256
FEATURES = ['aperiodic_exponent', 'alpha_peak_hz', 'alpha_peak_prominence_db',
            'alpha_relative_power', 'acf25', 'acf50', 'acf100', 'acf250']
FLOORS = np.array([.25, 1., 2., .05, .1, .1, .1, .1])


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def preprocess(eeg, rate):
    from fractions import Fraction
    eeg = np.asarray(eeg, dtype=float)
    if rate != RATE:
        ratio = Fraction(RATE / float(rate)).limit_denominator(10000)
        eeg = resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)
    eeg = sosfiltfilt(butter(4, [1, 40], btype='bandpass', fs=RATE, output='sos'), eeg, axis=-1)
    eeg -= eeg.mean(axis=-1, keepdims=True)
    eeg -= eeg.mean(axis=-2, keepdims=True)
    return eeg


def vector(features):
    return np.array([features[key] for key in FEATURES[:4]] + features['autocorrelation']['values'])


def descriptor(eeg):
    values = vector(resting_features(eeg, RATE))
    correlation = np.corrcoef(eeg)
    frequency, power = welch(eeg, fs=RATE, nperseg=4*RATE, axis=-1)
    power = power.mean(axis=0)
    select = (frequency >= 1) & (frequency <= 40)
    power /= np.trapezoid(power[select], frequency[select])
    return values, correlation, power


def load_resources(basis_path, network_path):
    cortex = load_brainstorm_cortex(ROOT/'Anatomy/Cortex_FS_20k.mat')
    leadfield = load_brainstorm_leadfield(ROOT/'Anatomy/Gain_20k_FS.mat',
        n_vertices=np.asarray(cortex['Vertices']).shape[-1])
    backend = G1SourceSimulator.from_brainstorm_cortex(leadfield, cortex)
    bundle = load_basis(basis_path, vertices_mm=backend.vertices*backend.coord_to_mm, faces=backend.faces)
    with np.load(network_path, allow_pickle=False) as data:
        if not np.allclose(data['vertices_brainstorm_mm'], backend.vertices*backend.coord_to_mm, atol=1e-4, rtol=0):
            raise ValueError('Network vertices are not aligned with this leadfield')
        if not np.array_equal(data['faces'], backend.faces):
            raise ValueError('Network faces do not match the simulation mesh')
        networks = project_maps(bundle, data['source_prior_weights'])
    names = [channel_key(name) for name in load_atlas()['channel_names']]
    bundle['_sensor_basis'] = leadfield @ bundle['basis']
    bundle['_asset_identity'] = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (basis_path,network_path,ROOT/'Anatomy/Gain_20k_FS.mat',ROOT/'Anatomy/Cortex_FS_20k.mat')}
    return backend, bundle, networks, names


def modal_observation(backend, bundle, networks, indices, config, seed, window=(0., 5.)):
    modes = config['n_modes']
    generated = generate_coefficients(bundle, config, srate=RATE, time_window=window,
        baseline_duration_s=2., rng=np.random.default_rng(seed),
        network_coefficients=networks if config['network_fraction'] else None)
    # Cached by the runner; the inverse projection is never inferred from EEG.
    projection = bundle['_sensor_basis'][indices, :modes].copy()
    projection -= projection.mean(axis=0, keepdims=True)
    clean = projection @ (generated['foreground'] + generated['background'])
    clean -= clean.mean(axis=1, keepdims=True)
    noise, _, _ = sensor_noise_with_baseline(backend, clean, srate=RATE,
        sensor_noise_type='iid_white', snr_db=20., seed=seed+100000,
        baseline_samples=2*RATE, average_reference=True)
    return preprocess(clean+noise, RATE)


def empirical_rest(names):
    directory = ROOT/'generated_simulation_dataset/G4/data_access/arithmetic_zyma2019/8359206e7e1b3d1f068033348e291d6677be1920'
    records, sources, indices = [], [], None
    # Prespecified pilot: eight training, four validation, four held-out subjects.
    chosen = set(range(8)) | set(range(28, 36))
    for path in sorted(directory.glob('recording_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            subjects = np.unique(data['subject']).tolist()
            if len(subjects) != 1 or int(subjects[0]) not in chosen or set(data['condition'].tolist()) != {'rest'}:
                continue
            if str(data['units']) != 'V':
                raise ValueError('Measured EEG must be in volts')
            channels = [channel_key(name) for name in data['ch_names']]
            keep = [i for i, name in enumerate(channels) if name in names]
            current_indices = [names.index(channels[i]) for i in keep]
            if indices is not None and indices != current_indices:
                raise ValueError('Rest recordings have different aligned montages')
            indices = current_indices
            eeg = preprocess(data['eeg'][:, keep], float(data['sfreq_hz']))
            splits = np.unique(data['split']).tolist()
            if len(splits) != 1:
                raise ValueError('A subject crosses stored data partitions')
            described = [descriptor(epoch) for epoch in eeg]
            records.append(dict(subject=str(subjects[0]), split=splits[0], windows=len(eeg),
                values=np.mean([row[0] for row in described], axis=0),
                correlation=np.mean([row[1] for row in described], axis=0),
                psd=np.mean([row[2] for row in described], axis=0)))
            sources.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                subject=str(subjects[0]),split=splits[0],windows=len(eeg),channels=channels))
    if {row['split'] for row in records} != {'train', 'val', 'test'}:
        raise ValueError('Prespecified rest subjects are missing')
    return records, sources, indices


def rest_comparison(args, backend, bundle, networks, names):
    records, sources, indices = empirical_rest(names)
    groups = {split:[r for r in records if r['split']==split] for split in ('train','val','test')}
    centers = {split:np.median([r['values'] for r in rows],axis=0) for split,rows in groups.items()}
    covariances = {split:np.mean([r['correlation'] for r in rows],axis=0) for split,rows in groups.items()}
    train = np.array([r['values'] for r in groups['train']])
    scales = np.maximum(np.median(abs(train-centers['train']),axis=0)*1.4826,FLOORS)
    protocol = dict(features=FEATURES,scale_floors=FLOORS.tolist(),actual_scales=scales.tolist(),
        selection='rank train; choose validation winner within best ten; test never selects',
        preprocessing='polyphase to256Hz;4th-order1-40Hzzero-phase;common-channelCAR;noamplitude-fit',
        synthetic_sensor_snr_db=20,activity_seconds=5,baseline_seconds=2,seeds=list(range(7101,7109)),
        score='mean standardized absolute feature errors + .25 * channel-correlation RMSE / .2',
        evidence='small descriptive pilot; no significance or population-wide realism claim',sources=sources)
    dump(args.output/'rest_protocol.json',protocol)
    rows=[]; arrays=[]; started=time.perf_counter()
    for index,(modes,fraction,exponent,power,frequency,width) in enumerate(itertools.product(
            (64,128,256),(0.,.5),(1.,1.5,2.),(.1,.5,2.),(9.,10.5,12.),(1.,2.))):
        config=dict(n_modes=modes,network_fraction=fraction,temporal_exponent=exponent,
            alpha_power_ratio=power,alpha_hz=frequency,alpha_width_hz=width,
            spatial_length_mm=20.,spatial_exponent=2.,scenario='rest')
        described=[descriptor(modal_observation(backend,bundle,networks,indices,config,seed)) for seed in range(7101,7109)]
        values=np.mean([r[0] for r in described],axis=0)
        correlation=np.mean([r[1] for r in described],axis=0)
        row=dict(candidate=index,**config,**dict(zip(FEATURES,map(float,values))))
        for split in ('train','val','test'):
            row[split+'_temporal_distance']=float(np.mean(abs(values-centers[split])/scales))
            row[split+'_spatial_rmse']=float(np.sqrt(np.mean((correlation-covariances[split])**2)))
            row[split+'_score']=row[split+'_temporal_distance']+.25*row[split+'_spatial_rmse']/.2
        rows.append(row);arrays.append((values,correlation,np.mean([r[2] for r in described],axis=0)))
        if (index+1)%24==0:print(json.dumps(dict(stage='rest_grid',completed=index+1,seconds=time.perf_counter()-started)),flush=True)
    with (args.output/'rest_candidates.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    finalists=sorted(rows,key=lambda r:r['train_score'])[:10]
    best=min(finalists,key=lambda r:r['val_score'])
    family={}
    for fraction in (0.,.5):
        pool=sorted([r for r in rows if r['network_fraction']==fraction],key=lambda r:r['train_score'])[:10]
        family['lbo' if fraction==0 else 'lbo_network']=min(pool,key=lambda r:r['val_score'])
    config={key:best[key] for key in config}
    # New random realizations after parameter selection, before reporting test fit.
    heldout=[descriptor(modal_observation(backend,bundle,networks,indices,config,seed)) for seed in range(8101,8109)]
    values=np.mean([r[0] for r in heldout],axis=0)
    final_test=float(np.mean(abs(values-centers['test'])/scales))
    summary=dict(best_config=config,best_candidate=best,per_family=family,final_seed_test_temporal_distance=final_test,
        independent_seed_features=dict(zip(FEATURES,map(float,values))),
        real_subjects={s:[r['subject'] for r in v] for s,v in groups.items()},
        real_group_features={s:dict(zip(FEATURES,map(float,v))) for s,v in centers.items()},
        candidates=len(rows),seconds=time.perf_counter()-started,protocol='rest_protocol.json')
    dump(args.output/'rest_results.json',summary)
    np.savez_compressed(args.output/'rest_plot_data.npz',frequency=np.fft.rfftfreq(4*RATE,1/RATE),
        measured_train=np.mean([r['psd'] for r in groups['train']],axis=0),
        measured_test=np.mean([r['psd'] for r in groups['test']],axis=0),
        simulated=np.mean([r[2] for r in heldout],axis=0),
        lbo=arrays[family['lbo']['candidate']][2],lbo_network=arrays[family['lbo_network']['candidate']][2])
    print(json.dumps(summary),flush=True)


def event_comparison(args,backend,bundle,networks,names):
    config=json.loads((args.output/'rest_results.json').read_text())['best_config']
    measured=[];indices=None; real_epochs={}; motor_indices=None
    def describe_event(epochs):
        features=evoked_features(epochs,RATE,-2.)
        motor=evoked_features(epochs[:,motor_indices],RATE,-2.)
        for key in ('alpha_ersp_db','beta_ersp_db','alpha_induced_ersp_db','beta_induced_ersp_db'):
            features[key]=motor[key]
        features['metadata']['ersp_channels']=[names[indices[i]] for i in motor_indices]
        return features
    for path in sorted((args.output/'prepared').glob('*R04.npz')):
        with np.load(path,allow_pickle=False) as data:
            channels=[channel_key(name) for name in data['ch_names']]
            keep=[i for i,name in enumerate(channels) if name in names]
            current=[names.index(channels[i]) for i in keep]
            if indices is not None and current!=indices:raise ValueError('Event montage changes across subjects')
            indices=current
            motor_indices=[i for i,index in enumerate(indices) if names[index] in ('C3','CZ','C4')]
            if len(motor_indices)!=3:raise ValueError('Motor ERSP requires C3, Cz and C4')
            epochs=preprocess(data['eeg'][:,keep],float(data['sfreq_hz']))
            # A common subject-level baseline scale permits unit-free ERP comparisons.
            time_axis=-2+np.arange(epochs.shape[-1])/RATE
            base=(time_axis>=-1.5)&(time_axis<-.5)
            scale=float(np.sqrt(np.mean(epochs[...,base]**2)))
            epochs/=scale
            features=describe_event(epochs)
            measured.append(dict(subject=str(data['subject']),split=str(data['split']),trials=len(epochs),features=features))
            real_epochs.setdefault(str(data['split']),[]).append(epochs)
    dump(args.output/'event_reference.json',dict(subjects=measured,
        scope='cue-aligned motor imagery, not a canonical P300/oddball ERP',
        preprocessing='sameCAR1-40Hz;subjectbaselineRMSnormalization',
        missing_reference=not bool(measured)))
    if indices is None:
        indices=list(range(len(names)))
        motor_indices=[i for i,name in enumerate(names) if name in ('C3','CZ','C4')]
    cases=[dict(scenario='rest',event_onset_s=0.,event_duration_s=2.5)]
    cases += [dict(scenario='erp',event_onset_s=0.,event_duration_s=2.5,
        erp_amplitude=a,erp_latency_s=latency,erp_width_s=.06,erp_latency_jitter_s=j)
        for a,j,latency in itertools.product((.2,.5,1.5),(0.,.04),(.15,.3,.45))]
    cases += [dict(scenario='ersp',event_onset_s=0.,event_duration_s=3.,ersp_depth=d)
              for d in (0.,-.1,-.3,-.6,-.9,.5)]
    cases += [dict(scenario='ersp',event_onset_s=0.,event_duration_s=3.,ersp_depth=d,
                   beta_power_ratio=.15,ersp_beta_depth=b)
              for d,b in itertools.product((-.3,-.6),(-.3,-.6))]
    cases += [dict(scenario='ersp',event_onset_s=0.,event_duration_s=3.,ersp_depth=d,
                   network_fraction=.5,network_indices=[0]) for d in (-.1,-.3,-.6)]
    results=[];plot_data={}
    trials=min([r['trials'] for r in measured],default=24)
    def band_timefrequency(epochs):
        from scipy.signal import stft
        frequency,time_axis,z=stft(epochs,fs=RATE,nperseg=128,noverlap=112,boundary=None,padded=False,axis=-1)
        power=np.mean(abs(z)**2,axis=(0,1));time_axis-=2
        base=(time_axis>=-1.5)&(time_axis<-.5)
        return frequency,time_axis,10*np.log10(np.maximum(power,1e-30)/np.maximum(power[:,base].mean(axis=1,keepdims=True),1e-30))
    for index,case in enumerate(cases):
        epochs=np.asarray([modal_observation(backend,bundle,networks,indices,{**config,**case},9100+seed,(-2.,3.))
                           for seed in range(trials)])
        times=-2+np.arange(epochs.shape[-1])/RATE
        mask=(times>=-1.5)&(times<-.5)
        epochs/=np.sqrt(np.mean(epochs[...,mask]**2))
        features=describe_event(epochs)
        results.append(dict(candidate=index,config=case,features=features,trials=len(epochs)))
        plot_data[f'erp_{index}']=(epochs-epochs[...,mask].mean(axis=-1,keepdims=True)).mean(axis=0)
        frequency,tf_time,plot_data[f'tf_{index}']=band_timefrequency(epochs[:,motor_indices])
    recommendations={}
    if measured:
        groups={s:[r for r in measured if r['split']==s] for s in ('train','test')}
        for label,keys,floors in [('erp',['erp_response_rms','evoked_power_fraction','erp_peak_latency_s'],[.1,.03,.08]),
                                  ('ersp',['alpha_ersp_db','beta_ersp_db','alpha_induced_ersp_db','beta_induced_ersp_db'],[1.,1.,1.,1.])]:
            centers={s:np.median([[r['features'][k] for k in keys] for r in v],axis=0) for s,v in groups.items()}
            for row in results:
                for split,center in centers.items():
                    row[f'{label}_{split}_distance']=float(np.mean(abs(np.array([row['features'][k] for k in keys])-center)/floors))
            candidates=[r for r in results if r['config']['scenario']==label]
            winner=min(candidates,key=lambda r:r[f'{label}_train_distance'])
            recommendations[label]=dict(candidate=winner['candidate'],selection='train_subjects_only',
                config=winner['config'],train_distance=winner[f'{label}_train_distance'],
                test_distance=winner[f'{label}_test_distance'],features=winner['features'],
                real_train=dict(zip(keys,map(float,centers['train']))),real_test=dict(zip(keys,map(float,centers['test']))))
        for split,rows in real_epochs.items():
            epochs=np.concatenate(rows)
            mask=(times>=-1.5)&(times<-.5)
            plot_data[f'real_erp_{split}']=(epochs-epochs[...,mask].mean(axis=-1,keepdims=True)).mean(axis=0)
            _,_,plot_data[f'real_tf_{split}']=band_timefrequency(epochs[:,motor_indices])
    dump(args.output/'event_results.json',dict(candidates=results,real_reference=measured,recommendations=recommendations,rest_config=config,
        interpretation='ERP amplitude/latency and inducedalpha modulation controls; measured MI comparison is descriptive',
        canonical_erp_validated=False))
    np.savez_compressed(args.output/'event_plot_data.npz',time_s=times,tf_frequency=frequency,tf_time=tf_time,**plot_data)
    print(json.dumps(dict(stage='event_comparison',candidates=len(results),real_subjects=len(measured))),flush=True)


def state_comparison(args,backend,bundle,networks,names):
    """A second dataset checks EO/EC separately, with its prespecified subject split."""
    records=[];indices=None
    for path in sorted((args.output/'prepared').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            condition=str(data['condition'])
            if condition not in ('eyes_open','eyes_closed'):continue
            channels=[channel_key(name) for name in data['ch_names']]
            keep=[i for i,name in enumerate(channels) if name in names]
            current=[names.index(channels[i]) for i in keep]
            if indices is not None and current!=indices:raise ValueError('State montage mismatch')
            indices=current
            described=[descriptor(epoch) for epoch in preprocess(data['eeg'][:,keep],float(data['sfreq_hz']))]
            records.append(dict(subject=str(data['subject']),split=str(data['split']),condition=condition,
                values=np.mean([row[0] for row in described],axis=0),psd=np.mean([row[2] for row in described],axis=0)))
    if not records:raise ValueError('Prepare the prespecified PhysioNet EO/EC recordings first')
    rows=[];descriptors=[]
    for index,(exponent,power,frequency) in enumerate(itertools.product((1.,1.5,2.),(0.,.01,.05,.2,.5,2.),(9.,10.5,12.))):
        config=dict(n_modes=256,network_fraction=0.,temporal_exponent=exponent,alpha_power_ratio=power,
                    alpha_hz=frequency,alpha_width_hz=2.,spatial_length_mm=20.,spatial_exponent=2.,scenario='rest')
        described=[descriptor(modal_observation(backend,bundle,networks,indices,config,seed,(0.,8.))) for seed in range(10101,10105)]
        rows.append(dict(candidate=index,config=config,features=dict(zip(FEATURES,map(float,np.mean([r[0] for r in described],axis=0))))))
        descriptors.append(described)
    results={};plot_data={}
    for condition in ('eyes_open','eyes_closed'):
        group={s:[r for r in records if r['split']==s and r['condition']==condition] for s in ('train','test')}
        if any(len(v)!=4 for v in group.values()):raise ValueError('Need four train and four test subjects per state')
        centers={s:np.median([r['values'] for r in v],axis=0) for s,v in group.items()}
        scales=np.maximum(np.median(abs(np.array([r['values'] for r in group['train']])-centers['train']),axis=0)*1.4826,FLOORS)
        scores=[float(np.mean(abs(np.array([r['features'][k] for k in FEATURES])-centers['train'])/scales)) for r in rows]
        winner=rows[int(np.argmin(scores))]
        heldout=[descriptor(modal_observation(backend,bundle,networks,indices,winner['config'],seed,(0.,8.))) for seed in range(11101,11109)]
        values=np.mean([r[0] for r in heldout],axis=0)
        results[condition]=dict(best_config=winner['config'],train_distance=min(scores),
            independent_seed_test_distance=float(np.mean(abs(values-centers['test'])/scales)),
            synthetic_features=dict(zip(FEATURES,map(float,values))),
            real_train=dict(zip(FEATURES,map(float,centers['train']))),real_test=dict(zip(FEATURES,map(float,centers['test']))),
            train_subjects=[r['subject'] for r in group['train']],test_subjects=[r['subject'] for r in group['test']])
        plot_data[condition+'_real']=np.mean([r['psd'] for r in group['test']],axis=0)
        plot_data[condition+'_sim']=np.mean([r[2] for r in heldout],axis=0)
    dump(args.output/'state_results.json',dict(conditions=results,candidates=rows,
        note='State-specific tuning on four subjects; separate four held out; fixed spatial configuration'))
    np.savez_compressed(args.output/'state_plot_data.npz',frequency=np.fft.rfftfreq(4*RATE,1/RATE),**plot_data)
    print(json.dumps(dict(stage='states',results=results)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated_simulation_dataset/G1_spectral_validation')
    parser.add_argument('--stage',choices=('rest','events','states','all'),default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    basis=args.output/'fsaverage5_lbo256.npz'
    network=ROOT/'projected_source_priors/fsaverage5_20k/neuro_motifs/neuro_motifs_fsaverage5_20k_source_priors_signed_maxabs.npz'
    resources=load_resources(basis,network)
    code_files=[ROOT/'Simulation_G1'/name for name in ('geometry.py','spectral.py','spectral_metrics.py','noise_covariance.py')]
    code_files.append(Path(__file__))
    dump(args.output/f'{args.stage}_run_identity.json',dict(assets=resources[1]['_asset_identity'],
        code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
        argv=sys.argv,rate_hz=RATE))
    if args.stage in ('rest','all'):rest_comparison(args,*resources)
    if args.stage in ('events','all'):event_comparison(args,*resources)
    if args.stage in ('states','all'):state_comparison(args,*resources)


if __name__=='__main__':main()
