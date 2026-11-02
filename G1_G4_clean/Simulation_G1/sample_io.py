"""Exclusive transactional sample directories with explicit verified resume.

The caller supplies a reproducible request identity (seed/config/resources/code).
Only directories carrying a valid commit marker count as completed samples.
Existing legacy or incomplete directories are never repaired or overwritten.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

SCHEMA = 'sample-directory-v1'


def _safe_name(value, max_length=240):
    if (not isinstance(value, str) or len(value) > max_length
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)
            or value.endswith('.') or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', value)):
        raise ValueError('Names must be safe nonempty single path components')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _request_hash(identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError('request_identity must be a nonempty JSON object identifying config/seed/resources')
    return hashlib.sha256(_json(identity).encode('utf-8')).hexdigest()


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(staging, target):
    """Atomically publish without replacing even an empty competing directory."""
    if os.name == 'nt':
        os.rename(staging, target)  # Windows rename never replaces an existing directory.
        return
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, 'renameat2', None)
        if rename is None:
            raise NotImplementedError('Exclusive directory publication requires renameat2 on Linux')
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:  # RENAME_NOREPLACE
            code = ctypes.get_errno()
            if code in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(code, os.strerror(code), str(target))
            raise OSError(code, os.strerror(code), str(target))
        return
    raise NotImplementedError('Exclusive sample publication is supported on Windows and Linux only')


def verified_sample(output_dir, sample_id, request_identity):
    """Return a verified manifest, or None only when no target exists.

    Identity mismatch, corruption and legacy/incomplete directories raise errors.
    Verification hashes every committed payload; no timestamp-only cache is used.
    """
    name = _safe_name(sample_id, 100)
    expected_request = _request_hash(request_identity)
    target = Path(output_dir).resolve() / name
    if not target.exists() and not target.is_symlink():
        return None
    if target.is_symlink() or not target.is_dir():
        raise ValueError('Sample target must be a real committed directory')
    marker = target / 'commit.json'
    if not marker.is_file() or marker.is_symlink():
        raise ValueError('Existing sample has no valid commit marker; preserve and inspect it separately')
    try:
        commit = json.loads(marker.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Invalid sample commit marker') from error
    if commit.get('schema') != SCHEMA or commit.get('sample_id') != name:
        raise ValueError('Sample commit schema or identity differs')
    if (commit.get('request_sha256') != expected_request
            or _request_hash(commit.get('request_identity')) != expected_request):
        raise ValueError('Sample request identity differs; use a new sample ID')
    files = commit.get('files_sha256')
    required = {f'{name}_data.npz', f'{name}_manifest.json'}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Commit marker lacks required sample payloads')
    for filename, expected in files.items():
        path = target / _safe_name(filename)
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'Sample payload checksum mismatch: {filename}')
    manifest = json.loads((target / f'{name}_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('sample_id') != name:
        raise ValueError('Saved manifest identity differs')
    return manifest


def save_sample_directory(output_dir, sample_id, arrays, manifest, *, request_identity,
                          resume=False, artifacts=None):
    """Write arrays/JSON/optional derived bytes, then exclusively publish once.

    Resume validates the caller's request identity and existing file hashes.
    Call verified_sample before generation when the caller can identify its
    request up front. This function itself does not skip prior generation work.
    """
    name = _safe_name(sample_id, 100)
    request_hash = _request_hash(request_identity)
    request_identity = json.loads(_json(request_identity))
    root = Path(output_dir).resolve()
    target = root / name
    if target.exists() or target.is_symlink():
        if resume:
            return verified_sample(root, name, request_identity)
        raise FileExistsError(f'Sample already exists: {target}; explicit verified resume is required')
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError('arrays must be a nonempty dictionary')
    payload = {}
    for key, value in arrays.items():
        _safe_name(key)
        array = np.asarray(value)
        if array.dtype.kind not in 'biufcSU' or (array.dtype.kind in 'iufc' and not np.isfinite(array).all()):
            raise ValueError(f'Array {key} must contain finite numeric or string data without pickle')
        payload[key] = array
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a JSON object without array payloads')
    saved = json.loads(_json(manifest))  # Snapshot JSON and reject NumPy arrays/nonfinite values.
    if saved.get('sample_id', name) != name:
        raise ValueError('manifest sample_id differs from destination')
    saved.update(sample_id=name, standard_data=f'{name}_data.npz', manifest=f'{name}_manifest.json',
                 arrays=sorted(payload))
    saved['files'] = dict(saved.get('files', {}), npz=f'{name}_data.npz', manifest=f'{name}_manifest.json')
    extras = {} if artifacts is None else dict(artifacts)
    reserved = {filename.casefold() for filename in (f'{name}_data.npz', f'{name}_manifest.json', 'commit.json')}
    for filename, value in extras.items():
        _safe_name(filename)
        if filename.casefold() in reserved or not isinstance(value, bytes):
            raise ValueError('Artifact names must be distinct from sample files and values must be bytes')
        reserved.add(filename.casefold())
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}.', suffix='.staging', dir=root))
    try:
        with (staging / f'{name}_data.npz').open('xb') as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(staging / f'{name}_manifest.json', saved)
        for filename, content in extras.items():
            with (staging / filename).open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        files = {path.name: _sha256(path) for path in staging.iterdir()}
        _write_json(staging / 'commit.json', dict(schema=SCHEMA, sample_id=name,
            request_identity=json.loads(_json(request_identity)), request_sha256=request_hash, files_sha256=files))
        try:
            _publish_directory(staging, target)
        except FileExistsError:
            if not resume:
                raise
            return verified_sample(root, name, request_identity)
        return saved
    finally:
        # Only this call's temporary directory is eligible for cleanup.
        if staging.exists() and not staging.is_symlink() and staging.resolve().parent == root:
            shutil.rmtree(staging)
