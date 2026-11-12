"""Metadata-only spectral recipes, replayed by the shared RecipeDataset.

Splits hold out realization groups, not cortical regions or network maps.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .presets import make_g1_config
from .training_recipes import (SCHEMA, STREAMS, RecipeDataset, _counts, _digest,
                               _json, _seed, _write_rows, asset_identity, generator_identity)


def build_network_recipes(output_dir, simulator, *, presets=('rest_calibrated', 'erp_network'),
                          n_groups=30, variants_per_group=3, split_fractions=(.8,.1,.1),
                          seed=42, sensor_snrs=(0.,10.,20.), resource_root=None, config_overrides=None):
    """Freeze spatial seeds per group and vary waveform/background/sensor seeds.

    All presets and variants belonging to a group share one split. Spectral
    overrides may point at alternative mesh-aligned resources; their hashes are
    pinned before any recipes are written. This does not change patch recipes.
    """
    for name,value in (('n_groups',n_groups),('variants_per_group',variants_per_group),('seed',seed)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value < (0 if name=='seed' else 1):
            raise ValueError(f'{name} must be an integer in range')
    n_groups,variants_per_group,seed=map(int,(n_groups,variants_per_group,seed))
    fractions=np.asarray(split_fractions,dtype=float)
    snrs=np.asarray(sensor_snrs,dtype=float)
    if fractions.shape!=(3,) or not np.isfinite(fractions).all() or np.any(fractions<0) or not np.isclose(fractions.sum(),1,rtol=0,atol=1e-12):
        raise ValueError('split_fractions must be three nonnegative values summing to one')
    if snrs.ndim!=1 or not snrs.size or not np.isfinite(snrs).all():
        raise ValueError('sensor_snrs must be a finite nonempty sequence')
    if isinstance(presets,str) or not presets or len(set(presets))!=len(presets):
        raise ValueError('presets must be a nonempty sequence of unique names')
    overrides=dict(config_overrides or {})
    forbidden={'random_seed','component_seeds','sample_id','sensor_snr_db'} & overrides.keys()
    if forbidden:
        raise ValueError(f'Recipe-owned config overrides: {sorted(forbidden)}')
    configs, resources = {}, {}
    from .spectral import _load_assets, _settings
    for name in presets:
        config=make_g1_config(name,resource_root=resource_root,**overrides)
        if config.amplitude_mode!='relative' or config.real_noise_registry:
            raise ValueError('Network recipe builder currently supports relative synthetic-noise recipes')
        config.validate_recipe()
        settings=_settings(config.spectral)
        loaded=_load_assets(simulator,settings,config.source_model)
        for kind in ('basis','network'):
            key=kind+'_path'
            if key in config.spectral:
                path=Path(config.spectral[key]).resolve()
                digest=_digest(path)
                if digest!=loaded[kind+'_sha256']:
                    raise ValueError('Spectral resource changed during recipe construction')
                config.spectral[key]=str(path)
                config.spectral[kind+'_sha256']=digest
                resources[str(path)]=dict(path=str(path),sha256=digest)
        configs[name]=asdict(config)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    identity=generator_identity()
    snapshot=output/'generator_snapshot/Simulation_G1'
    snapshot.mkdir(parents=True)
    for name in identity['files']:
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    assets=asset_identity(simulator)
    counts=_counts(n_groups,fractions)
    splits=np.repeat(['train','val','test'],counts)
    np.random.default_rng(_seed(seed,901)).shuffle(splits)
    supports,rows=[],[]
    indices={label:[] for label in ('train','val','test')}
    for group in range(n_groups):
        spatial_seed=_seed(seed,group,902)
        group_id=hashlib.sha256(_json(dict(mesh=assets['mesh_sha256'],spatial_seed=spatial_seed,
            resources=resources,presets=list(presets))).encode()).hexdigest()
        split=str(splits[group])
        supports.append(dict(index=group,support_id=group_id,split=split,
            kind='spectral_realization_group',spatial_seed=spatial_seed))
        for preset_index,name in enumerate(presets):
            for variant in range(variants_per_group):
                root=_seed(seed,group,preset_index,variant,903)
                streams={stream:_seed(root,i) for i,stream in enumerate(STREAMS)}
                streams['spatial']=spatial_seed
                config={**configs[name], 'sample_id':f'g1_{name}_{group:06d}_{variant:03d}',
                    'random_seed':root,'component_seeds':streams,
                    'sensor_snr_db':float(snrs[variant%len(snrs)])}
                row=dict(support_index=group,support_id=group_id,split=split,preset=name,
                    difficulty='configured',variant_index=variant,temporal_group=variant,
                    spatial_profile='spectral_realization_group',config=config)
                indices[split].append(len(rows))
                rows.append(row)
    support_table=_write_rows(output/'supports.jsonl',supports)
    shard=_write_rows(output/'recipes-00000.jsonl',rows)
    split_indexes={}
    for label,values in indices.items():
        path=output/f'{label}.indices.npy'
        np.save(path,np.asarray(values,dtype=np.int64),allow_pickle=False)
        split_indexes[label]=dict(file=path.name,count=len(values),sha256=_digest(path))
    manifest=dict(schema=SCHEMA,generator=identity,assets=assets,base_config={},dataset_seed=int(seed),
        n_supports=n_groups,n_groups=n_groups,n_samples=len(rows),variants_per_group=variants_per_group,
        presets=list(presets),support_table=support_table,shards=[dict(start=0,**shard)],split_indexes=split_indexes,
        split_support_counts=dict(zip(('train','val','test'),map(int,counts))),
        spectral_resources=list(resources.values()),metadata_only=True,
        generator_snapshot='generator_snapshot/Simulation_G1',
        split_policy='parent realization groups held out; all presets/variants grouped; NOT anatomical-support or network-map holdout')
    (output/'dataset.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resource-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--n-groups',type=int,default=30)
    parser.add_argument('--variants-per-group',type=int,default=3)
    parser.add_argument('--presets',nargs='+',default=['rest_calibrated','erp_network'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sensor-snrs',nargs='+',type=float,default=[0,10,20])
    args=parser.parse_args()
    from Simulation_G2.g2_simulation_core import load_brainstorm_cortex,load_brainstorm_leadfield
    from .g1_simulation_core import G1SourceSimulator
    cortex=load_brainstorm_cortex(args.resource_root/'Anatomy/Cortex_FS_20k.mat')
    gain=load_brainstorm_leadfield(args.resource_root/'Anatomy/Gain_20k_FS.mat',
                                  n_vertices=np.asarray(cortex['Vertices']).shape[-1])
    simulator=G1SourceSimulator.from_brainstorm_cortex(gain,cortex)
    manifest=build_network_recipes(args.output,simulator,presets=args.presets,n_groups=args.n_groups,
        variants_per_group=args.variants_per_group,seed=args.seed,sensor_snrs=args.sensor_snrs,resource_root=args.resource_root)
    print(json.dumps(dict(n_samples=manifest['n_samples'],output=str(args.output))))


if __name__=='__main__':
    main()
