"""Deterministic NMM periods, component truth, bounded caches and batch prefetch."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

import numpy as np

from Simulation_G1.spatial import SpatialDynamicsConfig
from Simulation_G1.noise_covariance import baseline_sample_count, COVARIANCE_KEYS, NOISE_ARRAY_KEYS
from .bank_reader import BankWindowReader
from .nmm_bank import projection_hash, parent_group_ids
from .nmm_mixing import mix_sources, make_sensors, disjoint_baseline_start, project_baseline


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def _positive_int(value, name, minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


class NMMRecipeDataset:
    """Indexed observation recipes over frozen regional banks.

    Pass a resident G2SourceSimulator with the matching immutable projection and
    a chosen leadfield. The default subset is training only. Readers are created
    lazily per process; the cache budget is per dataset instance/worker.
    """

    def __init__(self, path, backend, *, split='train', cache_bytes=256*1024**2):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path/'dataset.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema')!='nmm-periods-v1':
            raise ValueError('Unsupported NMM period index schema')
        self.backend=backend
        self.cache_bytes=_positive_int(cache_bytes,'cache_bytes',0)
        banks=self.manifest['banks']
        if not banks or any(row['n_regions']!=backend.n_regions or
                row['projection_sha256']!=projection_hash(backend.projection_matrix) for row in banks):
            raise ValueError('Period bank/projection differs from the supplied backend')
        if backend.leadfield.ndim!=2 or backend.leadfield.shape[1]!=backend.projection_matrix.shape[0]:
            raise ValueError('Leadfield and source vertex dimensions differ')
        if not np.isfinite(backend.leadfield).all():
            raise ValueError('Leadfield must be finite')
        self.leadfield_sha256=hashlib.sha256(np.ascontiguousarray(backend.leadfield,dtype='<f8').tobytes()).hexdigest()
        table=self.manifest['table']
        self._table_path=self._verified_path(table['file'],table['sha256'])
        offsets_path=self._verified_path(table['index'],table['index_sha256'])
        self._offsets=np.load(offsets_path,allow_pickle=False)
        if (self._offsets.ndim!=1 or self._offsets.dtype.kind not in 'iu' or len(self._offsets)!=table['count']
                or np.any(self._offsets<0) or np.any(np.diff(self._offsets.astype(np.int64))<=0)):
            raise ValueError('Invalid period offsets')
        self._table_signature=self._signature(self._table_path)
        self.split=split
        self._selection=None
        if split is not None:
            if split not in {'train','val','test'}:
                raise ValueError('split must be train, val, test or None for audit')
            selection=self.manifest['splits'][split]
            index_path=self._verified_path(selection['file'],selection['sha256'])
            self._selection=np.load(index_path,allow_pickle=False)
            if (self._selection.ndim!=1 or self._selection.dtype.kind not in 'iu'
                    or len(self._selection)!=selection['count'] or np.any(self._selection<0)
                    or np.any(self._selection>=len(self._offsets))
                    or len(np.unique(self._selection))!=len(self._selection)):
                raise ValueError('Invalid split index')
        self._reset_readers()

    @staticmethod
    def _signature(path):
        stat=path.stat()
        return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def _verified_path(self, name, digest):
        path=(self.path/name).resolve()
        if not path.is_relative_to(self.path) or _sha(path)!=digest:
            raise ValueError(f'Period index path/checksum mismatch: {name}')
        return path

    def _reset_readers(self):
        self._readers={}
        self._lock=RLock()
        self._pid=os.getpid()

    def __getstate__(self):
        state=self.__dict__.copy()
        state.pop('_lock',None)
        state['_readers']={}
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self._reset_readers()

    def __len__(self):
        return len(self._offsets) if self._selection is None else len(self._selection)

    def recipe(self,index):
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(self):
            raise IndexError('Period index outside selected split')
        if self._signature(self._table_path)!=self._table_signature:
            raise ValueError('Period recipe table changed during training')
        global_index=int(index if self._selection is None else self._selection[index])
        with self._table_path.open('rb') as handle:
            handle.seek(int(self._offsets[global_index])); row=json.loads(handle.readline())
        if self.split is not None and row['split']!=self.split:
            raise ValueError('Recipe split differs from selected split')
        parents=row['parent_data_group_ids']
        if not parents or any(self.manifest['parent_splits'].get(parent)!=row['split'] for parent in parents):
            raise ValueError('Recipe parents cross or omit the declared split')
        return row

    def _reader(self,index):
        if self._pid!=os.getpid():
            self._reset_readers()
        with self._lock:
            if index not in self._readers:
                bank=self.manifest['banks'][index]
                reader=BankWindowReader(bank['path'],self.cache_bytes//len(self.manifest['banks']))
                if reader.bank_sha256!=bank['index_sha256']:
                    raise ValueError('Bank index differs from the frozen period plan')
                self._readers[index]=reader
            return self._readers[index]

    def _window(self,reference,expected_samples=None):
        window,entry=self._reader(reference['bank_index']).read(reference['sample_id'],
            reference['start_sample'],reference['stop_sample'])
        if (entry['source_sha256']!=reference['source_sha256']
                or entry['simulation_group_id']!=reference['simulation_group_id']
                or entry['native_srate_hz']!=self.manifest['rate_hz']
                or window.shape[1]!=(self.manifest['period_samples'] if expected_samples is None else expected_samples)):
            raise ValueError('Period identity, rate or length differs from its bank')
        return window,entry

    def __getitem__(self,index):
        row=self.recipe(index); config=row['config']
        region,entry=self._window(row['foreground'])
        foreground=region.copy(); background=None
        selected=config['selected_regions']
        mode=config['source_mode']
        if mode in {'selected','partition'}:
            self.backend._validate_selected_regions(selected)
            foreground[np.setdiff1d(np.arange(len(region)),selected)]=0
            if mode=='partition':
                background=region.copy(); background[selected]=0
        elif mode!='network':
            raise ValueError('Unknown period source mode')
        background_entry=None
        if row['background'] is not None:
            if mode=='partition':
                raise ValueError('Partitioned and independent backgrounds cannot be combined implicitly')
            background,background_entry=self._window(row['background'])
        rate=self.manifest['rate_hz']
        baseline_duration=config.get('baseline_duration_s',1.)
        baseline_samples=baseline_sample_count(baseline_duration,rate)
        baseline_region,baseline_reference=None,None
        if background is not None:
            event_reference=row['foreground'] if mode=='partition' else row['background']
            event_entry=entry if mode=='partition' else background_entry
            baseline_reference=row.get('baseline_background')
            if baseline_reference is None:
                # Older recipes may replay only if their original parent has unused samples.
                start=disjoint_baseline_start(event_entry['n_samples'],event_reference['start_sample'],
                    event_reference['stop_sample'],baseline_samples)
                baseline_reference={**event_reference,'start_sample':start,'stop_sample':start+baseline_samples}
            identity=('bank_index','sample_id','simulation_group_id','source_sha256')
            if (any(baseline_reference[key]!=event_reference[key] for key in identity)
                    or not (baseline_reference['stop_sample']<=event_reference['start_sample']
                            or baseline_reference['start_sample']>=event_reference['stop_sample'])
                    or baseline_reference['stop_sample']-baseline_reference['start_sample']!=baseline_samples):
                raise ValueError('Baseline must be a disjoint window of the same background parent')
            baseline_region,baseline_entry=self._window(baseline_reference,baseline_samples)
            if parent_group_ids(baseline_entry)!=parent_group_ids(event_entry):
                raise ValueError('Baseline and event background ancestry differ')
            if mode=='partition':
                baseline_region=baseline_region.copy(); baseline_region[selected]=0
        actual_parents=set(parent_group_ids(entry))
        if background_entry is not None: actual_parents.update(parent_group_ids(background_entry))
        if (not actual_parents.issubset(row['parent_data_group_ids'])
                or any(self.manifest['parent_splits'].get(parent)!=row['split'] for parent in actual_parents)):
            raise ValueError('Stored recording ancestry differs from the recipe split')
        rate=self.manifest['rate_hz']; duration=self.manifest['period_samples']/rate
        spatial_metadata={'spatial_profile':config['spatial_profile'],'spatial_time_origin':'period_start'}
        foreground_vertex=None
        if config['spatial_profile']!='uniform':
            if mode=='network':
                raise ValueError('Network periods preserve spatial projection')
            centered=foreground.astype(np.float64)-foreground.mean(axis=1,dtype=np.float64,keepdims=True)
            foreground_vertex,details=self.backend.project_regions(centered,selected,srate=rate,
                time_window=(0.,duration),decay_mm=config['spatial_decay_mm'],
                spatial=SpatialDynamicsConfig(**config['spatial']))
            spatial_metadata.update(details)
        arrays,source_meta=mix_sources(foreground,background,self.backend.projection_matrix,self.backend.vertex_area,
            source_snr_db=config['source_snr_db'],normalize_source=config['normalize_source'],
            foreground_vertex=foreground_vertex)
        sensor_arrays,sensor_meta=make_sensors(self.backend,arrays,srate=rate,
            sensor_noise_type=config['sensor_noise_type'],sensor_snr_db=config['sensor_snr_db'],
            sensor_seed=config['sensor_seed'],average_reference=config['average_reference'],
            baseline_duration_s=baseline_duration,
            baseline_source=None if baseline_region is None else project_baseline(
                baseline_region,self.backend.projection_matrix,source_meta),
            baseline_metadata={'background_period':baseline_reference,'split':row['split']})
        arrays.update(sensor_arrays)
        arrays['time_ms']=np.arange(region.shape[1])*1000/rate
        metadata=dict(generator_type='NMM_period_dataset',recipe_id=row['recipe_id'],period_id=row['period_id'],
            split=row['split'],parent_data_group_ids=row['parent_data_group_ids'],
            foreground_period=row['foreground'],background_period=row['background'],config=config,
            sampling_rate_hz=rate,source_mode=mode,signal_type=entry['signal_type'],
            configured_regions=entry['active_regions'],
            protected_regions=entry.get('storage',{}).get('protected_regions',entry['active_regions']),
            parameter_preset_is_activation_truth=False,
            atlas_id=entry['atlas_id'],projection_sha256=entry['projection_sha256'],leadfield_sha256=self.leadfield_sha256,
            source_unit='normalized_model_state' if config['normalize_source'] else entry['source_unit'],
            foreground_storage=entry.get('storage',{'codec':'schema1_dense'}),
            background_storage=None if background_entry is None else background_entry.get('storage',{'codec':'schema1_dense'}),
            background_policy=('same_trajectory_partition_reweighted' if mode=='partition' else
                               'independent_rest' if background is not None else 'none_added'),
            target_definition='source_foreground is the localization target; source includes the background',
            **spatial_metadata,**source_meta,**sensor_meta)
        with np.errstate(over='raise',invalid='raise'):
            arrays={key:value if key in {'time_ms',*COVARIANCE_KEYS}
                    else value.astype(np.float32) for key,value in arrays.items()}
        return {**arrays,'manifest':metadata}

    def get_batch(self,indices,*,fields=('sensor','source_foreground','noise_covariance'),workers=0):
        """Stack selected fixed-shape arrays; keep heterogeneous metadata as a list."""
        _positive_int(workers,'workers',0)
        indices=list(indices); fields=tuple(fields)
        allowed={'sensor','clean_sensor','sensor_noise','foreground_sensor','background_sensor','source','source_foreground',
                 'source_background','region_source','region_foreground','region_background','time_ms',
                 *NOISE_ARRAY_KEYS}
        if not indices or not fields or len(set(fields))!=len(fields) or set(fields)-allowed:
            raise ValueError('Select a nonempty batch and distinct known array fields')
        def load(index):
            sample=self[index]
            return {key:sample[key] for key in fields},sample['manifest']
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                values=list(pool.map(load,indices))
        else:
            values=[load(index) for index in indices]
        return {**{key:np.stack([item[0][key] for item in values]) for key in fields},
                'manifests':[item[1] for item in values]}

    def iter_batches(self,batch_size,*,shuffle=False,seed=0,drop_last=False,workers=0,prefetch=0,
                     fields=('sensor','source_foreground','noise_covariance')):
        """Bounded FIFO prefetch; seeds and sample values never depend on scheduling."""
        batch_size=_positive_int(batch_size,'batch_size')
        prefetch=_positive_int(prefetch,'prefetch',0)
        order=np.arange(len(self))
        if shuffle: np.random.default_rng(seed).shuffle(order)
        batches=(order[start:start+batch_size] for start in range(0,len(order),batch_size)
                 if not drop_last or start+batch_size<=len(order))
        if not prefetch:
            for batch in batches: yield self.get_batch(batch,fields=fields,workers=workers)
            return
        with ThreadPoolExecutor(max_workers=1) as producer:
            pending=deque()
            for batch in batches:
                pending.append(producer.submit(self.get_batch,batch,fields=fields,workers=workers))
                if len(pending)>=prefetch: yield pending.popleft().result()
            while pending: yield pending.popleft().result()

    def cache_info(self):
        with self._lock:
            values={index:reader.cache_info() for index,reader in self._readers.items()}
        return dict(bytes=sum(item['bytes'] for item in values.values()),max_cache_bytes=self.cache_bytes,banks=values)

    def clear_cache(self):
        with self._lock:
            for reader in self._readers.values(): reader.clear()
