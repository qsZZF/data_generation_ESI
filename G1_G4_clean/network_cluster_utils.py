"""Alignment and provenance shared by the two legacy NeuroVault builders."""
import hashlib
import json
from pathlib import Path

import numpy as np


def mask_aligned_images(paths, term_weights, transform, on_error=None):
    """Mask one 3-D image per row and apply the same success mask to its terms."""
    paths = list(paths)
    term_weights = np.asarray(term_weights)
    if term_weights.ndim != 2 or len(paths) != len(term_weights):
        raise ValueError('Image paths and term weights must have aligned rows')
    usable = np.ones(len(paths), dtype=bool)
    rows = []
    for index, path in enumerate(paths):
        try:
            row = np.asarray(transform(path))
            if row.ndim != 2 or row.shape[0] != 1 or not np.isfinite(row).all():
                raise ValueError('Each image must produce one finite voxel row')
            if rows and row.shape != rows[0].shape:
                raise ValueError('Masked voxel counts do not match')
            rows.append(row)
        except Exception as error:
            usable[index] = False
            if on_error is not None:
                on_error(index, path, error)
    if not rows:
        raise ValueError('No usable images remain after masking')
    return np.vstack(rows), term_weights[usable], usable


def write_provenance(path, *, seed, arrays, input_files=(), settings=None):
    """Hash effective analysis inputs without duplicating large arrays in memory."""
    metadata = {'random_seed': int(seed), 'arrays': {}, 'input_files': {},
                'settings': settings or {}}
    for name, values in arrays.items():
        values = np.asarray(values)
        if values.dtype.hasobject:
            raise ValueError('Provenance arrays must not contain Python objects')
        digest = hashlib.sha256()
        for row in values:
            digest.update(np.ascontiguousarray(row).tobytes())
        metadata['arrays'][name] = dict(sha256=digest.hexdigest(),
                                      shape=list(values.shape), dtype=values.dtype.str)
    for filename in input_files:
        digest = hashlib.sha256()
        with Path(filename).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        metadata['input_files'][str(filename)] = digest.hexdigest()
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
