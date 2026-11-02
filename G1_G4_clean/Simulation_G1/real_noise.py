"""Immutable local measured-noise registries and deterministic disjoint windows.

Records declare already prepared channels, rate, reference and units. This module
does not resample, re-reference, flatten epochs, tile recordings or infer missing
acquisition/subject identity. Grouping blocks known acquisition, subject and exact
numeric-content aliases from crossing splits within the registry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections import OrderedDict

import numpy as np

try:
    from .amplitude import normalize_sensor_units
except ImportError:
    from amplitude import normalize_sensor_units


SCHEMA = 'measured-noise-registry-v1'
MAPPED_SCHEMA = 'measured-noise-registry-v2'
_SPLITS = {'train', 'val', 'test'}
_REFERENCES = {'average', 'as_supplied'}
_ROLES = {'empty_room', 'resting_background', 'prestimulus', 'residual'}


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(array, order=None):
    digest = hashlib.sha256(json.dumps(list(array.shape)).encode('ascii'))
    for index in range(len(array)) if order is None else order:
        for start in range(0, array.shape[1], 131072):
            digest.update(np.ascontiguousarray(array[index, start:start+131072], dtype='<f8').tobytes())
    return digest.hexdigest()


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _channels(values):
    if isinstance(values, str):
        raise ValueError('channel_names must be a sequence of unique channel names')
    try:
        names = [_name(value, 'channel name') for value in values]
    except TypeError as exc:
        raise ValueError('channel_names must be a sequence of unique channel names') from exc
    if not names or len(set(names)) != len(names):
        raise ValueError('channel_names must be nonempty and unique')
    return names


def _unit_info(declaration, n_channels):
    factors, si_units = normalize_sensor_units(np.ones((n_channels, 1)), declaration)
    units = [declaration] * n_channels if isinstance(declaration, str) else list(declaration)
    return units, factors[:, 0], si_units


def _rate(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('fs_hz must be a finite positive number')
    if np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
        raise ValueError('fs_hz must be a finite positive number')
    return float(value)


def _read_array(path, key, expected_file_hash=None):
    """Hash on every access, including same-size edits with preserved timestamps."""
    path = Path(path)
    if path.suffix.lower() not in ('.npz', '.npy'):
        raise ValueError('Noise records require a local NPZ or NPY file')
    with path.open('rb') as stream:
        file_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if expected_file_hash is not None and file_hash != expected_file_hash:
            raise ValueError(f'Noise file checksum changed: {path}')
        stream.seek(0)
        if path.suffix.lower() == '.npy':
            if key != 'noise':
                raise ValueError('NPY records use array_key=noise; metadata lives in the registry')
            array = np.load(path, allow_pickle=False, mmap_mode='r')
        else:
            with np.load(stream, allow_pickle=False) as archive:
                if key not in archive:
                    raise ValueError(f'Noise array key {key!r} is absent: {path}')
                array = archive[key]
        if (array.dtype.kind not in 'fiu' or array.ndim != 2 or array.shape[0] < 1
                or array.shape[1] < 2 or any(not np.isfinite(row).all() for row in array)):
            raise ValueError('Noise array must be finite real channels x time with at least two samples; epochs are not flattened')
        stream.seek(0)
        if hashlib.file_digest(stream, 'sha256').hexdigest() != file_hash:
            raise ValueError(f'Noise file changed while reading: {path}')
    return array, file_hash


def _record(spec):
    if not isinstance(spec, dict):
        raise ValueError('Each noise record must be a dictionary')
    required = ('path', 'record_id', 'recording_id', 'split', 'fs_hz',
                'channel_names', 'sensor_units', 'reference', 'noise_role')
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(f'Missing noise record fields: {missing}')
    record_id = _name(spec['record_id'], 'record_id')
    recording_id = _name(spec['recording_id'], 'recording_id')
    subject_id = spec.get('subject_id')
    if subject_id is not None:
        _name(subject_id, 'subject_id')
    if spec['split'] not in _SPLITS or spec['reference'] not in _REFERENCES or spec['noise_role'] not in _ROLES:
        raise ValueError('Invalid split, reference, or noise_role declaration')
    channel_names = _channels(spec['channel_names'])
    units, _, _ = _unit_info(spec['sensor_units'], len(channel_names))
    fs_hz = _rate(spec['fs_hz'])
    path = Path(spec['path']).resolve()
    array_key = _name(spec.get('array_key', 'noise'), 'array_key')
    array, file_hash = _read_array(path, array_key)
    if len(array) != len(channel_names):
        raise ValueError('Noise channels do not match channel_names')
    record = dict(path=str(path), array_key=array_key, record_id=record_id,
                  recording_id=recording_id, subject_id=subject_id, split=spec['split'],
                  fs_hz=fs_hz, channel_names=channel_names, sensor_units=units,
                  reference=spec['reference'], noise_role=spec['noise_role'],
                  shape=list(array.shape), file_sha256=file_hash, numeric_sha256=_array_hash(array))
    # Catch the same numeric recording after merely reordering its named channels.
    record['channel_sorted_numeric_sha256'] = _array_hash(array, np.argsort(channel_names))
    if 'provenance' in spec:
        record['provenance'] = json.loads(json.dumps(spec['provenance'], allow_nan=False))
    record['metadata_sha256'] = _json_hash(record)
    return record


def _assign_groups(records):
    roots = list(range(len(records)))

    def root(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    seen = {}
    tokens_by_record = []
    for index, record in enumerate(records):
        tokens = [('recording', record['recording_id']), ('content', record['numeric_sha256']),
                  ('channel_sorted_content', record['channel_sorted_numeric_sha256'])]
        if record['subject_id'] is not None:
            tokens.append(('subject', record['subject_id']))
        tokens_by_record.append(tokens)
        for token in tokens:
            if token in seen:
                roots[root(index)] = root(seen[token])
            else:
                seen[token] = index
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(root(index), []).append(index)
    for members in groups.values():
        if len({records[index]['split'] for index in members}) != 1:
            raise ValueError('Noise acquisition/subject/content connected group crosses splits')
        tokens = sorted({token for index in members for token in tokens_by_record[index]})
        group_id = 'real_noise_group_' + _json_hash(tokens)
        for index in members:
            records[index]['parent_group_id'] = group_id
            # Stable ancestor IDs also detect shared ancestry across separate registries.
            records[index]['parent_data_group_ids'] = sorted([group_id] + [
                'real_noise_' + kind + '_' + _json_hash(value) for kind, value in tokens])


def create_noise_registry(records, manifest_path):
    """Validate local records and atomically create a new manifest, never overwrite.

    No signal files are copied. Paths are resolved and file/numeric/metadata
    hashes recorded. Use stable recording_id and subject_id across all splits;
    content checks detect exact aliases, not unknown overlapping crops.
    """
    path = Path(manifest_path).resolve()
    if path.exists():
        raise FileExistsError(f'Noise registry already exists: {path}')
    entries = [_record(record) for record in records]
    if not entries:
        raise ValueError('Noise registry must contain at least one record')
    if len({record['record_id'] for record in entries}) != len(entries):
        raise ValueError('Duplicate noise record_id')
    _assign_groups(entries)
    for entry in entries:
        entry['record_sha256'] = _json_hash(entry)
    mapped = any(Path(entry['path']).suffix.lower() == '.npy' for entry in entries)
    manifest = dict(schema=MAPPED_SCHEMA if mapped else SCHEMA, records=entries,
                    grouping_policy='connected acquisition ID, optional subject ID, exact numeric content or named-channel permutation',
                    data_policy='local referenced NPZ/NPY; exact rate/reference; no resampling, tiling or epoch flattening')
    content = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic exclusive publish, unlike replace/rename on POSIX.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_noise_registry(manifest_path):
    """Validate metadata checksums and connected split groups without reading arrays."""
    path = Path(manifest_path).resolve()
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get('schema') not in (SCHEMA, MAPPED_SCHEMA):
        raise ValueError('Unsupported measured-noise registry schema')
    entries = manifest.get('records', [])
    if len({row['record_id'] for row in entries}) != len(entries):
        raise ValueError('Duplicate noise record_id in registry')
    for entry in entries:
        payload = {key: value for key, value in entry.items() if key != 'record_sha256'}
        if _json_hash(payload) != entry.get('record_sha256'):
            raise ValueError('Noise record metadata checksum changed')
    recomputed = json.loads(json.dumps(entries))
    _assign_groups(recomputed)
    for actual, expected in zip(entries, recomputed):
        if any(actual.get(key) != expected[key] for key in ('parent_group_id', 'parent_data_group_ids')):
            raise ValueError('Noise connected group metadata differs from declared ancestry')
    return manifest


def _validated_record_array(record):
    array, file_hash = _read_array(record['path'], record['array_key'], record['file_sha256'])
    if list(array.shape) != record['shape'] or _array_hash(array) != record['numeric_sha256']:
        raise ValueError('Noise numeric content or shape checksum changed')
    return array, file_hash


def _file_signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class NoiseWindowReader:
    """Per-process LRU for validated, immutable recordings.

    NPZ arrays count toward max_cache_bytes. Read-only NPY mappings use no heap
    array allocation and count toward a separate 16-entry descriptor bound;
    their resident pages are managed by the OS, not this byte budget. Fast reads
    trust unchanged file stat identity. strict=True rehashes even cache hits.
    Registry metadata is always revalidated. Returned windows are writable copies.
    """
    def __init__(self, max_cache_bytes=256*1024*1024, *, strict=False):
        if isinstance(max_cache_bytes, bool) or not isinstance(max_cache_bytes, int) or max_cache_bytes < 0:
            raise ValueError('max_cache_bytes must be a nonnegative integer')
        if not isinstance(strict, bool):
            raise ValueError('strict must be Boolean')
        self.max_cache_bytes = max_cache_bytes
        self.strict = strict
        self._pid = os.getpid()
        self._cache = OrderedDict()
        self.cache_bytes = 0

    def clear(self):
        self._cache.clear()
        self.cache_bytes = 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_cache=OrderedDict(), cache_bytes=0)
        return state

    def _array(self, record, strict):
        if self._pid != os.getpid():
            self.clear()
            self._pid = os.getpid()
        path = str(Path(record['path']).resolve())
        key = (path, record['array_key'], record['file_sha256'], record['numeric_sha256'], tuple(record['shape']))
        identity = _file_signature(path)
        entry = self._cache.get(key)
        if entry is not None and entry['identity'] == identity:
            if strict:
                with Path(path).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != record['file_sha256']:
                        raise ValueError(f'Noise file checksum changed: {path}')
            self._cache.move_to_end(key)
            return entry['array'], record['file_sha256']
        if entry is not None:
            self.cache_bytes -= self._cache.pop(key)['bytes']
        array, file_hash = _validated_record_array(record)
        if _file_signature(path) != identity:
            raise ValueError('Noise file changed while validating cache input')
        array.setflags(write=False)
        charge = 0 if isinstance(array, np.memmap) else array.nbytes
        if charge <= self.max_cache_bytes:
            while self._cache and (self.cache_bytes + charge > self.max_cache_bytes or len(self._cache) >= 16):
                self.cache_bytes -= self._cache.popitem(last=False)[1]['bytes']
            self._cache[key] = dict(array=array, identity=identity, bytes=charge)
            self.cache_bytes += charge
        return array, file_hash

    def read(self, manifest_path, record_id, *, strict=None, **kwargs):
        strict = self.strict if strict is None else strict
        if not isinstance(strict, bool):
            raise ValueError('strict must be Boolean')
        return _noise_window(manifest_path, record_id,
                             array_loader=lambda record: self._array(record, strict), **kwargs)


def load_noise_window(manifest_path, record_id, **kwargs):
    """Strict uncached load: verify full file/numeric checksums on every access.

    Keyword arguments: expected_srate, expected_channels, expected_reference,
    baseline_samples, event_samples, seed, optional expected_units and split.
    """
    return _noise_window(manifest_path, record_id, array_loader=_validated_record_array, **kwargs)


def _noise_window(manifest_path, record_id, *, array_loader, expected_srate, expected_channels,
                      expected_units=None, expected_reference, baseline_samples,
                      event_samples, seed, split=None):
    """Return aligned contiguous baseline+event and its source interval/provenance."""
    path = Path(manifest_path).resolve()
    raw_manifest = path.read_bytes()
    manifest = load_noise_registry(path)
    if path.read_bytes() != raw_manifest:
        raise ValueError('Noise registry changed while reading')
    entries = manifest['records']
    matches = [entry for entry in entries if entry['record_id'] == record_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown noise record_id: {record_id}')
    record = matches[0]
    if split is not None and (split not in _SPLITS or record['split'] != split):
        raise ValueError('Noise record split differs from requested split')
    if _rate(expected_srate) != record['fs_hz']:
        raise ValueError('Noise sample rate differs; prepare an explicitly resampled record first')
    if expected_reference not in _REFERENCES or expected_reference != record['reference']:
        raise ValueError('Noise reference differs; prepare an explicitly referenced record first')
    channels = _channels(expected_channels)
    if set(channels) != set(record['channel_names']):
        raise ValueError('Noise channel set differs; prepare the explicit common-channel subset first')
    for name, value, minimum in (('baseline_samples', baseline_samples, 2), ('event_samples', event_samples, 2), ('seed', seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    order = [record['channel_names'].index(name) for name in channels]
    original_units = [record['sensor_units'][index] for index in order]
    _, original_factors, original_dimensions = _unit_info(original_units, len(channels))
    units, destination_factors, dimensions = _unit_info(
        original_units if expected_units is None else expected_units, len(channels))
    if original_dimensions != dimensions:
        raise ValueError('Noise units have incompatible physical dimensions')
    conversion = original_factors / destination_factors
    array, file_hash = array_loader(record)
    count = int(baseline_samples + event_samples)
    if count > array.shape[1]:
        raise ValueError('Noise recording is too short for baseline plus event; no tiling')
    start = int(np.random.default_rng(int(seed)).integers(array.shape[1] - count + 1)) if count < array.shape[1] else 0
    selected = np.asarray(array[order, start:start + count], dtype=np.float64) * conversion[:, None]
    if not np.isfinite(selected).all():
        raise ValueError('Noise unit conversion exceeds finite range')
    metadata = dict(registry_path=str(path), registry_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        record_id=record['record_id'], recording_id=record['recording_id'], subject_id=record['subject_id'],
        split=record['split'], noise_role=record['noise_role'], reference=record['reference'],
        parent_group_id=record['parent_group_id'], parent_data_group_ids=record['parent_data_group_ids'],
        input_path=record['path'], array_key=record['array_key'], file_sha256=file_hash,
        numeric_sha256=record['numeric_sha256'], record_sha256=record['record_sha256'], input_shape=record['shape'],
        original_fs_hz=record['fs_hz'], sampling_rate_hz=record['fs_hz'], channel_names=channels,
        original_sensor_units=original_units, sensor_units=units, unit_conversion_factors=conversion.tolist(),
        seed=int(seed), start_sample=start, n_samples=count,
        baseline_start_sample=start, baseline_stop_sample=start + int(baseline_samples),
        event_start_sample=start + int(baseline_samples), event_stop_sample=start + count,
        baseline_interval_samples=[0, int(baseline_samples)], event_interval_samples=[int(baseline_samples), count],
        baseline_event_nonoverlapping=True, selected_sha256=_array_hash(selected),
        amplitude_calibration='unit_conversion_only_no_gain_normalization')
    return selected, metadata
