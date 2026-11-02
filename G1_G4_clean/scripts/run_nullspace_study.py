"""Paired sensor/source-count study with a fixed head model and physical signal.

Source grids are nested fixed-normal point sets, not remeshed volume conductors.
MEG and template hdEEG cases illustrate inverse geometry, not device performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np

from resolution_benchmark.nullspace import (analyze_operator,projector_from_patterns,
                                            solve_svd,unit_safe_whitener)
from Simulation_G1.spectral import generate_coefficients


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fps(points,seed):
    """Deterministic nested spatial coverage, independent of leadfield/SVD scores."""
    points=np.asarray(points,dtype=float)
    rng=np.random.default_rng(seed);selected=[];distances=np.full(len(points),np.inf)
    current=int(rng.integers(len(points)))
    for _ in points:
        selected.append(current)
        distances=np.minimum(distances,np.sum((points-points[current])**2,axis=1))
        distances[selected]=-np.inf
        current=int(np.argmax(distances))
    return np.asarray(selected)


def source_order(coords,hemispheres):
    per_hemi=[]
    for group_index,hemi in enumerate(sorted(np.unique(hemispheres))):
        indices=np.flatnonzero(hemispheres==hemi)
        per_hemi.append(indices[fps(coords[indices],101+group_index)])
    return np.asarray([indices[i] for i in range(max(map(len,per_hemi)))
                       for indices in per_hemi if i<len(indices)])


def sensor_subsets(names,positions,kind,seed):
    if kind=='meg':
        groups={}
        for index,name in enumerate(names):groups.setdefault(str(name).replace(' ','')[:-1],[]).append(index)
        groups=list(groups.values())
        group_positions=np.asarray([positions[g].mean(axis=0) for g in groups])
        order=fps(group_positions,seed)
        # Preserve co-located magnetometer/gradiometer groups, including known bad-channel omissions.
        counts=sorted(set([len(groups),85,67,50,34,22,11]),reverse=True)
        return [np.asarray([i for k in order[:n] for i in groups[k]]) for n in counts if n<=len(groups)]
    order=fps(positions,seed)
    return [order[:n] for n in (len(order),200,150,100,64,32) if n<=len(order)]


def covariance_factor(covariance):
    std=np.sqrt(np.diag(covariance))
    corr=covariance/std[:,None]/std[None,:]
    eig,vec=np.linalg.eigh((corr+corr.T)/2)
    if eig[0]<-1e-8*eig[-1]:raise ValueError('Noise template is not PSD after projection and unit scaling')
    keep=eig>1e-10*eig[-1]
    return std[:,None]*vec[:,keep]*np.sqrt(eig[keep])[None,:]


def source_truth(n_sources,n_times,seed):
    generated=generate_coefficients({'eigenvalues':np.zeros(100)},
        dict(scenario='rest',n_modes=100,network_fraction=0.,temporal_exponent=1.5,
             alpha_hz=10.5,alpha_power_ratio=.5),srate=256,time_window=(0,n_times/256),
        baseline_duration_s=2.,rng=np.random.default_rng(seed))
    result=np.zeros((n_sources,n_times));result[:100]=generated['foreground']
    return result


def normalized_error(value,truth):
    return float(np.linalg.norm(value-truth)/np.linalg.norm(truth))


def nested_patch_bases(coords,hemispheres,counts):
    """Nested coherent groups on the fixed fine dipole grid; Euclidean norm, not area density."""
    groups=[np.flatnonzero(hemispheres==h) for h in np.unique(hemispheres)]
    bases={}
    while len(groups)<=max(counts):
        if len(groups) in counts:
            basis=np.zeros((len(coords),len(groups)))
            for index,vertices in enumerate(groups):basis[vertices,index]=1/np.sqrt(len(vertices))
            bases[len(groups)]=basis
        if len(groups)==max(counts):break
        scores=[float(np.sum((coords[g]-coords[g].mean(axis=0))**2)) if len(g)>1 else -1 for g in groups]
        index=int(np.argmax(scores));vertices=groups[index]
        if len(vertices)<2:raise ValueError('Cannot refine beyond the fine source grid')
        centered=coords[vertices]-coords[vertices].mean(axis=0)
        axis=np.linalg.svd(centered,full_matrices=False)[2][0]
        ordered=vertices[np.argsort(centered@axis,kind='stable')];mid=len(ordered)//2
        groups[index]=ordered[:mid];groups.append(ordered[mid:])
    return bases


def run_case(path,kind,output):
    with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    metadata=json.loads(str(data.pop('metadata_json')))
    channel_mask=np.isin(data['channel_types'],['mag','grad']) if kind=='meg' else data['channel_types']=='eeg'
    rows=np.flatnonzero(channel_mask)
    G=data['leadfield_fixed'][rows].astype(float)
    C=data['noise_cov'][np.ix_(rows,rows)].astype(float)
    patterns=data['sensor_projector_basis'][rows]
    full_projector=projector_from_patterns(patterns)
    C=full_projector@C@full_projector.T;C=(C+C.T)/2
    order=source_order(data['source_rr_m'],data['source_hemi'])
    G=G[:,order]
    positions=data['channel_positions_head_m'][rows];names=data['channel_names'][rows]
    rng=np.random.default_rng(20261031)
    factor=covariance_factor(C)
    baseline=factor@rng.standard_normal((factor.shape[1],512))
    noise=factor@rng.standard_normal((factor.shape[1],1024))
    baseline_cov=np.cov(baseline)
    truth=source_truth(G.shape[1],1024,20261032)
    full_whitener,white_info=unit_safe_whitener(full_projector@baseline_cov@full_projector.T)
    clean=full_projector@G@truth
    # One calibration for the full array; every source count sees the same field/data/noise.
    gain=np.sqrt(100*np.mean((full_whitener@noise)**2)/np.mean((full_whitener@clean)**2))
    truth*=gain
    clean=G@truth
    observed=clean+noise
    ridge_lambda=.01*np.linalg.svd(full_whitener@full_projector@G[:,:100],compute_uv=False)[0]
    counts=[n for n in (100,150,200,250,300,400) if n<=G.shape[1]]
    protocol=dict(kind=kind,source_file=str(path),asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_counts=counts,source_support=100,source_normal_orientation='fixed',
        source_order='nested hemisphere-balanced spatial farthest-point sampling',
        channel_selection='three spatial coverage orderings; MEG triplets kept together',
        baseline_s=2,srate_hz=256,activity_s=4,noise='Gaussian draws from supplied projected covariance template',
        calibration='20dB whitened SNR once on full array, never recalibrated after channel/source changes',
        numerical_rtol=1e-10,effective_rtol=1e-3,ridge_lambda=float(ridge_lambda),
        ridge_calibration='fixed at .01 times largest singular value of full-array 100-source reference; not retuned per cell',source_gain=float(gain),
        provided_metadata=metadata,
        interpretation='case-specific inverse conditioning; neither clinical recovery nor device performance')
    dump(output/f'{kind}_protocol.json',protocol)
    output_rows=[];spectra={};examples={};started=time.perf_counter()
    for repeat in range(3):
        selections=sensor_subsets(names,positions,kind,41+repeat)
        for selected in selections:
            P=projector_from_patterns(patterns[selected])
            Gsub=P@G[selected]
            Y=P@observed[selected];E=P@noise[selected]
            Cbase=P@baseline_cov[np.ix_(selected,selected)]@P.T
            Cref=P@C[np.ix_(selected,selected)]@P.T
            for cov_kind,cov in [('baseline_2s',Cbase),('reference_cov',Cref)]:
                for count in counts:
                    X=truth[:count]
                    analysis=analyze_operator(Gsub[:,:count],cov,source=X)
                    info=analysis['info'];singular=analysis['singular_values'];fractions=info['source_energy_fractions']
                    entry=dict(modality=kind,repeat=repeat,n_channels=len(selected),n_sources=count,covariance=cov_kind,
                        covariance_rank=info['whitening']['rank'],rank=info['rank'],nullity=info['nullity'],
                        effective_rank=info['effective_rank'],full_column_rank=info['full_column_rank'],
                        condition_number=info['condition_number'],nonzero_condition=info['nonzero_condition'],
                        null_energy=fractions['null'],near_null_energy=fractions['near_null'],stable_energy=fractions['stable'],
                        null_sensor_relative_residual=info['null_sensor_relative_residual'],
                        isotropic_prior_expected_null_fraction=info['nullity']/count,
                        fixed_support_prior_expected_null_fraction=float(1-np.sum(analysis['right_vectors'][:info['rank'],:100]**2)/100))
                    coefficients=analysis['right_vectors']@X
                    for tolerance in (1e-2,1e-3,1e-4):
                        rank=int(np.sum(singular>tolerance*singular[0]))
                        entry[f'rank_{tolerance:g}']=rank
                        entry[f'unresolved_energy_{tolerance:g}']=float(1-np.sum(coefficients[:rank]**2)/np.sum(X**2))
                    projected_signal=singular[:,None]*coefficients
                    projected_noise=analysis['left_vectors'].T@(analysis['whitener']@E)
                    entry['modes_signal_above_noise']=int(np.sum(np.mean(projected_signal**2,axis=1)>np.mean(projected_noise**2,axis=1)))
                    signal_power=np.mean((analysis['operator']@X)**2)
                    noise_power=np.mean((analysis['whitener']@E)**2)
                    entry['actual_whitened_snr_db']=float(10*np.log10(signal_power/noise_power))
                    clean_estimate=solve_svd(analysis,P@clean[selected],method='pinv')
                    entry['clean_pinv_nrmse']=normalized_error(clean_estimate,X)
                    entry['clean_min_norm_equals_observable']=bool(np.allclose(clean_estimate,analysis['observable_source'],rtol=1e-4,atol=np.linalg.norm(X)*1e-7))
                    for method in ('pinv','tsvd','ridge'):
                        estimate=solve_svd(analysis,Y,method=method,ridge_relative=ridge_lambda/singular[0])
                        entry[method+'_nrmse']=normalized_error(estimate,X)
                        entry[method+'_whitened_sensor_residual']=float(np.linalg.norm(analysis['whitener']@(Gsub[:,:count]@estimate-Y))/np.linalg.norm(analysis['whitener']@Y))
                        if method=='pinv':
                            # Use the common truth denominator for the orthogonal error decomposition.
                            range_squared=float(np.sum((estimate-analysis['observable_source'])**2)/np.sum(X**2))
                            entry['error_decomposition_relative_gap']=float(abs(entry[method+'_nrmse']**2-(fractions['null']+range_squared))/max(entry[method+'_nrmse']**2,1e-30))
                    output_rows.append(entry)
                    if repeat==0 and cov_kind=='baseline_2s':
                        spectra[f'm{len(selected)}_n{count}']=singular/singular[0]
                        if count==300 and len(selected)<=105 and not examples:
                            null=analysis['null_source'];perturbed=X+2*null
                            examples=dict(source=X,observable=analysis['observable_source'],null_source=null,
                                near_null_source=analysis['near_null_source'],source_perturbed=perturbed,
                                sensor_clean=Gsub[:,:count]@X,sensor_perturbed=Gsub[:,:count]@perturbed,
                                source_rr_m=data['source_rr_m'][order[:count]],channels=names[selected],
                                metadata_json=json.dumps(entry))
            print(json.dumps(dict(case=kind,repeat=repeat,channels=len(selected),seconds=time.perf_counter()-started)),flush=True)
    checks={}
    primary=[r for r in output_rows if r['covariance']=='baseline_2s']
    checks['null_energy_monotone_when_removing_channels']=all(
        np.all(np.diff([r['null_energy'] for r in sorted([r for r in primary if r['repeat']==rep and r['n_sources']==n],key=lambda x:-x['n_channels'])])>=-1e-8)
        for rep in range(3) for n in counts)
    checks['null_energy_monotone_when_adding_sources']=all(
        np.all(np.diff([r['null_energy'] for r in sorted([r for r in primary if r['repeat']==rep and r['n_channels']==m],key=lambda x:x['n_sources'])])>=-1e-8)
        for rep in range(3) for m in set(r['n_channels'] for r in primary if r['repeat']==rep))
    checks['max_error_decomposition_gap']=max(r['error_decomposition_relative_gap'] for r in primary)
    checks['max_null_sensor_relative_residual']=max(r['null_sensor_relative_residual'] for r in primary)
    with (output/f'{kind}_sweep.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output_rows[0]));writer.writeheader();writer.writerows(output_rows)
    np.savez_compressed(output/f'{kind}_spectra.npz',**spectra)
    np.savez_compressed(output/f'{kind}_null_example.npz',**examples)
    np.savez_compressed(output/f'{kind}_simulation.npz',baseline_sensor=baseline,noise_covariance=baseline_cov,
        reference_covariance=C,source=truth,source_order=order,
        sensor=full_projector@observed,clean_sensor=full_projector@clean,
        sensor_noise=full_projector@noise,sensor_projector=full_projector,
        channel_names=names,metadata_json=json.dumps(protocol))
    # Reconstruct out-of-grid activity on a coarse model as a separate misspecification control.
    offgrid=np.zeros_like(truth);offgrid[-100:]=source_truth(100,1024,20261033)*gain
    Ym=G@offgrid;P=full_projector;mismatch=[]
    covariance=P@baseline_cov@P.T
    for n in sorted(set(counts+[G.shape[1]])):
        analysis=analyze_operator(P@G[:,:n],covariance)
        estimate=solve_svd(analysis,P@Ym,method='pinv')
        embedded=np.zeros_like(offgrid);embedded[:n]=estimate
        mismatch.append(dict(n_sources=n,full_array_channels=len(names),
            offgrid_truth_fraction=float(np.sum(offgrid[n:]**2)/np.sum(offgrid**2)),
            source_nrmse_on_common_point_grid=normalized_error(embedded,offgrid),
            whitened_sensor_relative_residual=float(np.linalg.norm(analysis['whitener']@(P@G[:,:n]@estimate-P@Ym))/np.linalg.norm(analysis['whitener']@P@Ym))))
    # Same anatomical locations with three free orientations can have very different identifiability.
    orientations=[]
    free=data['leadfield_free'][rows].reshape(len(rows),-1,3)[:,order]
    normals=data['source_normals'][order]
    for n in (100,200,300):
        if n>G.shape[1]:continue
        A=P@free[:,:n].reshape(len(rows),3*n)
        cartesian=(truth[:n,None,:]*normals[:n,:,None]).reshape(3*n,-1)
        analysis=analyze_operator(A,covariance,source=cartesian)
        info=analysis['info']
        orientations.append(dict(vertices=n,source_dof=3*n,rank=info['rank'],nullity=info['nullity'],
            effective_rank=info['effective_rank'],condition_number=info['condition_number'],
            null_energy=info['source_energy_fractions']['null']))
    # Conditioning controls have their own matched truth; QR favors sensor-visible locations.
    from scipy.linalg import qr
    _,_,pivots=qr(full_whitener@P@G,pivoting=True,mode='economic')
    conditioning=[]
    for strategy in ('spatial_coverage','forward_qr'):
        for n in (20,50,100):
            columns=np.arange(n) if strategy=='spatial_coverage' else pivots[:n]
            A=P@G[:,columns];X=truth[:n].copy()
            for snr in (20.,30.):
                scaling=np.sqrt(10**(snr/10)*np.mean((full_whitener@P@noise)**2)/np.mean((full_whitener@A@X)**2))
                target=X*scaling;analysis=analyze_operator(A,covariance,source=target)
                estimate=solve_svd(analysis,A@target+P@noise)
                conditioning.append(dict(strategy=strategy,n_sources=n,snr_db=snr,
                    condition_number=analysis['info']['condition_number'],pinv_nrmse=normalized_error(estimate,target),
                    selection_note='QR is a forward-dependent sensitivity-biased control, not uniform cortical coverage'))
    # Refinement can also be studied without replacing the physical cortical field.
    bases=nested_patch_bases(data['source_rr_m'][order],data['source_hemi'][order],counts)
    patch_truth=bases[100]@truth[:100]
    scaling=np.sqrt(100*np.mean((full_whitener@P@noise)**2)/np.mean((full_whitener@P@G@patch_truth)**2))
    patch_truth*=scaling;patch_Y=P@(G@patch_truth+noise);patch_rows=[]
    for n,basis in bases.items():
        X=basis.T@patch_truth
        np.testing.assert_allclose(basis@X,patch_truth,rtol=1e-12,atol=np.linalg.norm(patch_truth)*1e-14)
        analysis=analyze_operator(P@G@basis,covariance,source=X)
        estimate=solve_svd(analysis,patch_Y,method='ridge',ridge_relative=ridge_lambda/analysis['singular_values'][0])
        info=analysis['info']
        patch_rows.append(dict(n_sources=n,rank=info['rank'],nullity=info['nullity'],condition_number=info['condition_number'],
            null_energy=info['source_energy_fractions']['null'],near_null_energy=info['source_energy_fractions']['near_null'],
            ridge_nrmse=normalized_error(estimate,X)))
    summary=dict(kind=kind,n_channels=len(names),fine_sources=G.shape[1],noise_subspace_rank=white_info['rank'],
        checks=checks,seconds=time.perf_counter()-started,rows=len(output_rows),
        full_array=[r for r in primary if r['repeat']==0 and r['n_channels']==len(names)],
        channel_path_n100=[r for r in primary if r['repeat']==0 and r['n_sources']==100],
        mismatch_control=mismatch,free_orientation_control=orientations,conditioning_controls=conditioning,
        nested_patch_control=patch_rows,patch_basis_definition='equal-norm coherent groups on fixed fine dipole grid; not area-weighted density',
        protocol=f'{kind}_protocol.json')
    dump(output/f'{kind}_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('meg','hdeeg','all'),default='all')
    parser.add_argument('--directory',type=Path,default=ROOT/'generated_simulation_dataset/nullspace_study')
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    paths=dict(meg=args.directory/'mne_sample/mne_sample_forward.npz',hdeeg=args.directory/'hdeeg_template/forward.npz')
    results={}
    for kind,path in paths.items():
        if args.case in (kind,'all'):results[kind]=run_case(path,kind,args.directory)
    dump(args.directory/'run_identity.json',dict(cases=list(results),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__),ROOT/'resolution_benchmark/nullspace.py',ROOT/'Simulation_G1/spectral.py')}))


if __name__=='__main__':main()
