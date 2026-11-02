"""Cortical eligibility without changing the original vertex/leadfield ordering."""
from __future__ import annotations

import hashlib

import numpy as np


def validate_cortex_mask(mask, n_vertices):
    """Accept Boolean or binary masks; never silently convert labels to a mask."""
    if mask is None:
        return np.ones(n_vertices, dtype=bool)
    values = np.asarray(mask)
    if (values.shape != (n_vertices,) or values.dtype.kind not in 'biuf'
            or not np.isin(values, [0, 1]).all() or not np.any(values)):
        raise ValueError('cortex mask must contain one Boolean/binary value per vertex and retain at least one vertex')
    return values.astype(bool, copy=True)


def cortex_mask_metadata(mask, provenance):
    return {'provenance': provenance, 'eligible_vertices': int(mask.sum()),
            'excluded_vertices': int((~mask).sum()),
            'sha256': hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()}


def cortex_mask_from_dict(cortex):
    """Loader contract: CortexMask is an optional vertex-aligned binary vector."""
    mask = cortex.get('CortexMask', cortex.get('cortex_mask'))
    provenance = cortex.get('CortexMaskProvenance', cortex.get('cortex_mask_provenance'))
    return mask, provenance or ('provided_mask' if mask is not None else 'unspecified_all_vertices')


def load_cortex_mask(path, n_vertices, cortex_key=None):
    """Read optional embedded Desikan scouts from a Brainstorm v7.3 MAT file.

    Missing atlases return None for custom meshes. An existing but malformed
    Desikan atlas raises rather than silently enabling excluded cortex. The
    retained union excludes unknown and corpus-callosum scouts; indices remain
    aligned with the full mesh. h5py is only imported when loading a file.
    """
    import h5py

    with h5py.File(path, 'r') as file:
        if cortex_key is None:
            keys = [key for key in file if isinstance(file[key], h5py.Group)
                    and 'Vertices' in file[key] and 'Faces' in file[key]]
            if len(keys) != 1:
                raise ValueError('Specify cortex_key when MAT file does not contain exactly one cortex group')
            cortex_key = keys[0]
        cortex = file[cortex_key]
        if 'Atlas' not in cortex:
            return None, 'unspecified_all_vertices'
        atlas = cortex['Atlas']
        def read_text(ref):
            return ''.join(chr(int(c)) for c in np.asarray(file[ref]).flat)
        names = [read_text(ref) for ref in np.asarray(atlas['Name']).flat]
        if 'Desikan-Killiany' not in names:
            return None, 'unspecified_all_vertices'
        scouts = file[np.asarray(atlas['Scouts']).flat[names.index('Desikan-Killiany')]]
        labels = np.asarray(scouts['Label']).ravel()
        supports = np.asarray(scouts['Vertices']).ravel()
        if len(labels) != len(supports):
            raise ValueError('Desikan scout labels and supports have different lengths')
        mask = np.zeros(n_vertices, dtype=bool)
        for name_ref, vertices_ref in zip(labels, supports):
            name = read_text(name_ref).strip().lower()
            if name.startswith(('corpuscallosum', 'unknown')):
                continue
            vertices = np.asarray(file[vertices_ref]).ravel()
            if (vertices.dtype.kind not in 'iuf' or not np.isfinite(vertices).all()
                    or np.any(vertices != np.floor(vertices))
                    or np.any((vertices < 1) | (vertices > n_vertices))):
                raise ValueError('Desikan scout contains invalid MATLAB vertex indices')
            mask[vertices.astype(int) - 1] = True
        return validate_cortex_mask(mask, n_vertices), 'embedded_Desikan-Killiany_cortical_scouts'
