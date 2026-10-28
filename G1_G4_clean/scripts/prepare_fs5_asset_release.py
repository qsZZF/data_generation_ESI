"""Package existing fsaverage5 derivatives without pickle or silent remapping.

Network labels are recovered from the matching QC CSV. Numeric arrays are copied
exactly. The historical coactivation archive is copied unchanged and labelled as
an unregistered legacy association matrix, not an EEG covariance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def export_network_archive(source,qc_csv,destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():
        raise ValueError('Release output must not overwrite the original asset')
    with Path(qc_csv).open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    with np.load(source,allow_pickle=False) as archive:
        numeric={key:archive[key] for key in archive.files if key not in {'network_names','source_paths'}}
    weights=numeric['source_prior_weights']
    if weights.ndim!=2 or weights.shape[1]!=len(rows) or not rows:
        raise ValueError('QC CSV must have one ordered row per network column')
    companion=source.with_suffix('.mat')
    if companion.exists():
        from scipy.io import loadmat
        labels=loadmat(companion,variable_names=['networkNames','sourcePaths'],simplify_cells=True)
        for key,column in (('networkNames','name'),('sourcePaths','source_file')):
            if key not in labels or np.asarray(labels[key]).astype(str).reshape(-1).tolist()!=[row[column] for row in rows]:
                raise ValueError('QC label order differs from the MATLAB companion')
    if any(not np.isfinite(value).all() for value in numeric.values() if value.dtype.kind in 'fc'):
        raise ValueError('Nonfinite network array')
    numeric.update(network_names=np.asarray([row['name'] for row in rows],dtype=str),
                   source_paths=np.asarray([row['source_file'] for row in rows],dtype=str))
    if any(value.dtype.hasobject for value in numeric.values()):
        raise ValueError('Release archives must not require pickle')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.with_suffix('.part').open('wb') as handle:
        np.savez_compressed(handle,**numeric)
    destination.with_suffix('.part').replace(destination)
    with np.load(destination,allow_pickle=False) as output:
        for key,value in numeric.items():
            if not np.array_equal(output[key],value):
                raise ValueError(f'Export changed {key}')
    return dict(source_sha256=digest(source),sha256=digest(destination),bytes=destination.stat().st_size,
                shape=list(weights.shape),dtype=str(weights.dtype),numeric_arrays_preserved=True,
                label_conversion='QC CSV row order to Unicode; original object arrays were not unpickled',
                labels_verified_against_matlab=companion.exists(),
                alignment='Brainstorm vertex order; NIfTI sampling uses legacy approximate surface coordinate transform')


def inspect_coactive(path):
    """Scan every matrix value with bounded memory; do not load a dense 1.68 GB array."""
    with zipfile.ZipFile(path) as archive,archive.open('M.npy') as handle:
        version=np.lib.format.read_magic(handle)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(handle)
        if len(shape)!=2 or min(shape)<1 or shape[0]!=shape[1] or dtype.kind!='f' or fortran:
            raise ValueError('Expected a row-major square floating coactivation matrix')
        remaining=int(np.prod(shape))*dtype.itemsize
        minimum,maximum=np.inf,-np.inf
        while remaining:
            block=handle.read(min(8*1024**2,remaining))
            if not block or len(block)%dtype.itemsize:
                raise ValueError('Truncated coactivation array')
            values=np.frombuffer(block,dtype=dtype)
            if not np.isfinite(values).all():raise ValueError('Nonfinite coactivation values')
            minimum=min(minimum,float(values.min()));maximum=max(maximum,float(values.max()))
            remaining-=len(block)
        if handle.read(1):raise ValueError('Unexpected trailing bytes in M.npy')
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files if key!='M'}
        if any(value.dtype.hasobject for value in arrays.values()):raise ValueError('Unsafe coactivation metadata')
        if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in 'fc'):
            raise ValueError('Nonfinite coactivation metadata')
        if 'coords' not in arrays or arrays['coords'].shape!=(shape[0],3):
            raise ValueError('Coactivation coordinates must identify every matrix row')
    return dict(shape=list(shape),dtype=str(dtype),all_matrix_values_finite=True,
                minimum=minimum,maximum=maximum,metadata_shapes={key:list(value.shape) for key,value in arrays.items()})


def verified_copy(source,destination,expected_hash=None):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Source and destination must differ')
    expected_hash=expected_hash or digest(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or digest(destination)!=expected_hash:
        temporary=destination.with_suffix('.part')
        shutil.copyfile(source,temporary)
        if digest(temporary)!=expected_hash:raise ValueError('Copied file checksum mismatch')
        temporary.replace(destination)
    return expected_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-root',type=Path,required=True)
    parser.add_argument('--coactive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files={}
    for label in ('SIM_data','neuro_motifs'):
        source_dir=args.network_root/label;qc=source_dir/f'{label}_projection_qc.csv'
        for variant in ('signed_maxabs','abs_sum1'):
            stem=f'{label}_fsaverage5_20k_source_priors_{variant}'
            rel=f'{label}/{stem}.npz'
            files[rel]=export_network_archive(source_dir/(stem+'.npz'),qc,args.output/rel)
            mat=source_dir/(stem+'.mat');rel=f'{label}/{mat.name}'
            checksum=verified_copy(mat,args.output/rel)
            files[rel]=dict(sha256=checksum,bytes=mat.stat().st_size,copy='unchanged MATLAB companion')
        rel=f'{label}/{qc.name}';checksum=verified_copy(qc,args.output/rel)
        files[rel]=dict(sha256=checksum,bytes=qc.stat().st_size)
    coactive=inspect_coactive(args.coactive)
    checksum=digest(args.coactive)
    rel='coactive_legacy/macm_surface_z.npz'
    verified_copy(args.coactive,args.output/rel,checksum)
    coactive.update(sha256=checksum,bytes=args.coactive.stat().st_size,copy='original bytes unchanged',
        status='legacy; coordinate registration and Brainstorm row correspondence unverified',
        interpretation='directed standardized conditional association; not permutation Z, correlation or PSD covariance',
        coordinate_provenance='frame not stored; inspected legacy builder compared source surface RAS to voxel world coordinates without explicit transform',
        projection_limitation='legacy nearest-voxel seeds and all-mask-voxel-to-surface averaging; no cortical ribbon or distance cutoff',
        generation_provenance='not recorded at artifact creation; inferred method is not verified execution provenance',
        legacy_code_inspected_sha256='05385f936d6e1688710b0e11cfc72360a6c88ad11a37bcd9bf348b0f934c1286')
    files[rel]=coactive
    (args.output/'coactive_legacy/provenance.json').write_text(json.dumps(coactive,indent=2)+'\n',encoding='utf-8')
    manifest=dict(schema='fs5-derived-assets-v1',files=files,
        storage={'network':'ordinary Git; numeric values preserved','coactive':'Git LFS; git lfs pull required'},
        limitations=['network Brainstorm row alignment does not certify MNI registration',
                     'coactive cannot be used directly as EEG noise covariance or substituted for a matching source projection'],
        upstream='User-generated NeuroVault/Neurosynth-derived statistical maps; retain original-source attribution and applicable terms')
    (args.output/'ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),coactive_sha256=checksum,coactive_bytes=coactive['bytes'],matrix=coactive['shape']),indent=2))


if __name__=='__main__':main()
