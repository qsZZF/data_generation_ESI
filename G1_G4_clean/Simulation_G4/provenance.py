"""Dataset-qualified measured-data ancestry for model-conditioned simulations."""
import hashlib
import json
from urllib.parse import quote

import numpy as np


def _qualified(*parts):
    if any(not isinstance(value,str) or not value.strip() for value in parts):
        raise ValueError('Dataset, subject and recording identities must be nonempty strings')
    return ':'.join(quote(value,safe='') for value in parts)


def _parent(kind,identifier):
    # Match the measured-noise registry's stable identifier hashing convention.
    raw=json.dumps(identifier,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    return 'real_noise_'+kind+'_'+hashlib.sha256(raw.encode('utf-8')).hexdigest()


def reconstruction_ancestry(dataset_id,subject,condition,provenance,fitting_indices,evaluation_indices):
    """Only fitting records/windows are generative parents; subject spans conditions.

    Evaluation windows are separately listed, not used as fitted-parameter parents.
    To link a future measured-noise resource, use these qualified subject/recording
    IDs in its registry. Raw legacy IDs are never guessed or silently aliased.
    """
    if dataset_id is None:
        return dict(dataset=None,subject=subject,condition=condition,parent_data_group_ids=[],
                    ancestry_status='dataset_not_declared_not_exportable')
    subject_id=_qualified(dataset_id,subject)
    if not isinstance(condition,str) or not condition.strip():
        raise ValueError('Condition identity must be a nonempty string')
    recordings=np.asarray(provenance['recording']).astype(str)
    windows=np.asarray(provenance['window_id']).astype(str)
    if recordings.ndim!=1 or windows.shape!=recordings.shape:
        raise ValueError('Recording/window provenance must have matching one-dimensional shapes')
    selections=[]
    for selected in (fitting_indices,evaluation_indices):
        indices=np.asarray(selected)
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
                or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(recordings))):
            raise ValueError('Fitting/evaluation indices must be nonempty unique valid indices')
        selections.append(indices)
    if np.intersect1d(*selections).size:
        raise ValueError('Fitting and evaluation indices overlap')
    result=dict(dataset=dataset_id,subject=subject,condition=condition,qualified_subject_id=subject_id,
        ancestry_policy='dataset-qualified subject across conditions; fitting acquisitions/windows only as generative parents')
    for prefix,indices in zip(('fitting','evaluation'),selections):
        record_ids=sorted({_qualified(dataset_id,subject,str(recordings[i])) for i in indices})
        window_ids=[str(windows[i]) for i in indices]
        qualified_windows=[_qualified(dataset_id,subject,str(recordings[i]),str(windows[i])) for i in indices]
        if len(set(qualified_windows))!=len(qualified_windows):
            raise ValueError('Repeated window identity within a reconstruction split')
        parents=[_parent('subject',subject_id)]+[_parent('recording',value) for value in record_ids]
        parents+=['g4_fitwindow_'+hashlib.sha256(value.encode('utf-8')).hexdigest() for value in qualified_windows]
        result[prefix+'_recording_ids']=record_ids
        result[prefix+'_window_ids']=window_ids
        result['parent_data_group_ids' if prefix=='fitting' else 'evaluation_parent_data_group_ids']=sorted(parents)
        result[prefix+'_qualified_window_ids']=qualified_windows
    if set(result['fitting_qualified_window_ids'])&set(result['evaluation_qualified_window_ids']):
        raise ValueError('Fitting/evaluation window identities overlap')
    return result
