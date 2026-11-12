"""Small real-mesh integration check for presets, recipes and clean-code copies."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from Simulation_G1.g1_simulation_core import G1SourceSimulator
from Simulation_G1.presets import PRESET_NAMES,make_g1_config
from Simulation_G1.network_recipes import build_network_recipes
from Simulation_G1.training_recipes import RecipeDataset,generator_identity
from Simulation_G1.quality import assess_sample
from Simulation_G2.g2_simulation_core import load_brainstorm_cortex,load_brainstorm_leadfield


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resource-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cortex=load_brainstorm_cortex(args.resource_root/'Anatomy/Cortex_FS_20k.mat')
    leadfield=load_brainstorm_leadfield(args.resource_root/'Anatomy/Gain_20k_FS.mat',
        n_vertices=np.asarray(cortex['Vertices']).shape[-1])
    simulator=G1SourceSimulator.from_brainstorm_cortex(leadfield,cortex)
    records=[]
    for index,name in enumerate(PRESET_NAMES):
        config=make_g1_config(name,resource_root=args.resource_root,random_seed=4300+index)
        tic=time.perf_counter()
        sample=simulator.generate_sample(config,compute_diagnostics=False)
        quality=assess_sample(sample,cortex_mask=simulator.cortex_mask,full_covariance=True)
        if quality['status']=='failed':
            raise AssertionError(quality['errors'])
        assert sample['sensor'].shape==(64,512)
        assert sample['baseline_sensor'].shape==(64,512)
        assert sample['noise_covariance'].shape==(64,64)
        hashes={key:digest(sample[key]) for key in ('sensor','source_foreground','baseline_sensor','noise_covariance')}
        record=dict(preset=name,seconds=time.perf_counter()-tic,quality=quality,
                    array_hashes=hashes,config=config.spectral)
        if name=='erp_network':
            info=sample['manifest']['network_erp']
            assert np.isclose(info['roi_power_ratio_realized'],2,rtol=1e-5)
            assert np.linalg.matrix_rank(sample['foreground_sensor'],tol=1e-5)>1
            record['network_erp']=info
        np.savez_compressed(args.output/f'{name}_sensor.npz',sensor=sample['sensor'],
                            baseline_sensor=sample['baseline_sensor'],noise_covariance=sample['noise_covariance'])
        del sample
        gc.collect()
        repeated=simulator.generate_sample(config,compute_diagnostics=False)
        assert hashes=={key:digest(repeated[key]) for key in hashes}
        del repeated
        gc.collect()
        records.append(record)
        print(json.dumps(dict(preset=name,quality=quality['status'],replay='exact',seconds=record['seconds'])),flush=True)
    manifest=build_network_recipes(args.output/'recipes',simulator,presets=PRESET_NAMES,
        n_groups=3,variants_per_group=2,split_fractions=(1/3,1/3,1/3),seed=123,
        sensor_snrs=(10,20),resource_root=args.resource_root)
    splits={}
    for split in ('train','val','test'):
        reader=RecipeDataset(args.output/'recipes',simulator,split=split)
        ids=set()
        for i in range(len(reader)):
            config,row=reader.recipe(i)
            ids.add(row['support_id'])
        splits[split]=ids
    assert all(not splits[a]&splits[b] for a,b in (('train','val'),('train','test'),('val','test')))
    reader=RecipeDataset(args.output/'recipes',simulator,split='train')
    recipe_records=[]
    for i in (0,2,4):
        sample=reader[i]
        quality=assess_sample(sample,cortex_mask=simulator.cortex_mask,full_covariance=True)
        assert quality['status']!='failed',quality['errors']
        recipe_records.append(dict(index=i,preset=reader.recipe(i)[1]['preset'],quality=quality['status'],
                                   sensor_hash=digest(sample['sensor'])))
        del sample
        gc.collect()
    import Simulation_G1.presets as origin
    assert Path(origin.__file__).resolve().is_relative_to(ROOT)
    result=dict(status='passed',code_root=str(ROOT),generator=generator_identity(),
                presets=records,recipes=recipe_records,recipe_count=manifest['n_samples'],
                split_group_counts={k:len(v) for k,v in splits.items()},
                split_policy=manifest['split_policy'],baseline_duration_s=2,
                verification='real fsaverage5; full sample QC/covariance; same-seed exact replay; group-split audit')
    (args.output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
