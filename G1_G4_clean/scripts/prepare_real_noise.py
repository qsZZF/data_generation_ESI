"""Prepare a local continuous EDF/FIF/NPZ recording as a measured-noise asset.

No downloads, interpolation, epoch flattening, anatomical channel aliases or
unit guesses are performed. Resting EEG is neural background, not instrument
noise. The returned JSON record can be passed to create_noise_registry.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SI_UNITS = {'eeg': 'V', 'mag': 'T', 'grad': 'T/m'}
NOISE_ROLES = ('empty_room', 'resting_background', 'prestimulus', 'residual')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_names(names):
    values = np.asarray(names)
    if values.ndim != 1 or not len(values):
        raise ValueError('channel names must be a nonempty one-dimensional list')
    canonical = [str(name).strip().upper().rstrip('.') for name in values]
    if any(not name for name in canonical) or len(set(canonical)) != len(canonical):
        raise ValueError('Empty or duplicate channel names after uppercase/trailing-dot normalization')
    return canonical


def _read_input(path, *, npz_key, npz_channels_key, npz_sfreq_key, npz_types_key, npz_units_key):
    if path.suffix.lower() == '.npz':
        if not npz_key or not npz_channels_key:
            raise ValueError('NPZ requires explicit npz_key and npz_channels_key')
        with np.load(path, allow_pickle=False) as archive:
            required = (npz_key, npz_channels_key, npz_sfreq_key, npz_types_key, npz_units_key)
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f'NPZ missing declared data/channel/SI metadata keys: {missing}')
            data = np.asarray(archive[npz_key])
            if data.ndim != 2 or data.dtype.kind not in 'fiu':
                raise ValueError('Noise input must be real numeric 2D channels by time; never flatten epochs')
            names = np.asarray(archive[npz_channels_key]).astype(str).tolist()
            types = np.asarray(archive[npz_types_key]).astype(str)
            units = np.asarray(archive[npz_units_key]).astype(str)
            rate_value = np.asarray(archive[npz_sfreq_key])
            if rate_value.shape != () or rate_value.dtype.kind not in 'fiu':
                raise ValueError('NPZ sampling rate must be a numeric scalar')
            rate = float(rate_value)
        if types.shape == ():
            types = np.repeat(types, len(names))
        if units.shape == ():
            units = np.repeat(units, len(names))
        if len(names) != data.shape[0] or types.shape != (len(names),) or units.shape != (len(names),):
            raise ValueError('NPZ channel names/types/units must match data rows')
        return dict(read=lambda start, stop: data[:, start:stop], close=lambda: None,
                    names=names, types=types.tolist(), units=units.tolist(), rate=rate,
                    native_indices=list(range(len(names))),
                    n_samples=data.shape[1], bads=[], first_sample=0,
                    native_filter_metadata=None, input_format='NPZ_explicit_continuous_SI')
    import mne
    if path.suffix.lower() == '.edf':
        raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
    elif path.name.lower().endswith(('.fif', '.fif.gz')):
        raw = mne.io.read_raw_fif(path, preload=False, verbose='ERROR')
    else:
        raise ValueError('Supported local formats are .edf, .fif, .fif.gz and 2D .npz')
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    names = [raw.ch_names[index] for index in picks]
    types = raw.get_channel_types(picks=picks)
    return dict(read=lambda start, stop: raw.get_data(picks=picks, start=start, stop=stop),
                close=raw.close, names=names, types=types, units=[SI_UNITS[kind] for kind in types],
                native_indices=picks.tolist(),
                rate=float(raw.info['sfreq']), n_samples=int(raw.n_times), bads=list(raw.info['bads']),
                first_sample=int(raw.first_samp), input_format='MNE_SI_reader',
                native_filter_metadata=dict(highpass_hz=float(raw.info['highpass']),
                                            lowpass_hz=float(raw.info['lowpass'])))


def prepare_real_noise(input_path, output_path, *, record_id, recording_id, noise_role, split,
                       subject_id=None, target_fs=256., start_s=0., duration_s=None,
                       reference='as_supplied', channel_names=None, npz_key=None, npz_channels_key=None,
                       npz_sfreq_key='sampling_rate_hz', npz_types_key='channel_types',
                       npz_units_key='channel_units', source_url=None, record_json=None):
    """Write continuous SI noise (compressed NPZ or mmap-ready NPY) and JSON metadata.

    Crop boundaries round to nearest native samples and stop is exclusive.
    Cropping precedes Kaiser-window polyphase anti-alias resampling, followed
    by optional EEG-only average reference. No temporal demeaning/bandpass is
    added. Existing input filter metadata is recorded, never assumed absent.
    """
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    sidecar = Path(record_json).resolve() if record_json else output.with_suffix('.record.json')
    if output.suffix.lower() not in ('.npz', '.npy'):
        raise ValueError('output_path must have .npz or .npy extension')
    if output == sidecar or source in (output, sidecar):
        raise ValueError('Input, output and record paths must differ')
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    if any(not isinstance(value, str) or not value.strip() for value in (record_id, recording_id)):
        raise ValueError('record_id and recording_id must be nonempty strings')
    if noise_role not in NOISE_ROLES or split not in ('train', 'val', 'test'):
        raise ValueError('Choose an explicit noise_role and train/val/test split')
    if reference not in ('average', 'as_supplied'):
        raise ValueError('reference must be average or as_supplied')
    if (isinstance(target_fs, bool) or not np.isfinite(target_fs) or target_fs <= 0
            or not np.isfinite(start_s) or start_s < 0
            or (duration_s is not None and (not np.isfinite(duration_s) or duration_s < 2))):
        raise ValueError('Require positive target rate, nonnegative crop start and duration >=2 seconds')
    source_hash = _sha256(source)
    recording = _read_input(source, npz_key=npz_key, npz_channels_key=npz_channels_key,
                            npz_sfreq_key=npz_sfreq_key, npz_types_key=npz_types_key, npz_units_key=npz_units_key)
    try:
        names = _canonical_names(recording['names'])
        for kind, unit in zip(recording['types'], recording['units']):
            if kind not in SI_UNITS or unit != SI_UNITS[kind]:
                raise ValueError('Input channels must declare EEG V, magnetometer T or gradiometer T/m SI units')
        requested = names if channel_names is None else _canonical_names(channel_names)
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(f'Requested channel names missing (no aliases/interpolation): {missing}')
        selection = [names.index(name) for name in requested]
        types = [recording['types'][index] for index in selection]
        units = [recording['units'][index] for index in selection]
        if reference == 'average' and (set(types) != {'eeg'} or len(types) < 2):
            raise ValueError('Average reference requires at least two EEG-only channels; never mix MEG units')
        native_rate = recording['rate']
        if not np.isfinite(native_rate) or native_rate <= 0:
            raise ValueError('Native sampling rate must be positive and finite')
        start = int(round(start_s * native_rate))
        stop = recording['n_samples'] if duration_s is None else start + int(round(duration_s * native_rate))
        if start >= stop or stop > recording['n_samples'] or (stop - start) / native_rate < 2:
            raise ValueError('Crop must fit within the recording and contain at least 2 seconds')
        values = np.asarray(recording['read'](start, stop), dtype=np.float64)[selection]
        if not np.isfinite(values).all():
            raise ValueError('Selected noise contains nonfinite values')
        ratio = Fraction(float(target_fs) / native_rate).limit_denominator(1000000)
        if not np.isclose(native_rate * ratio.numerator / ratio.denominator, target_fs, rtol=1e-12, atol=0):
            raise ValueError('Target rate is not representable by the bounded rational resampler')
        if ratio != 1:
            values = resample_poly(values, ratio.numerator, ratio.denominator, axis=-1,
                                   window=('kaiser', 5.), padtype='line')
        if reference == 'average':
            values -= values.mean(axis=0, keepdims=True)
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Prepared noise exceeds finite float32 range')
        provenance = dict(source_file=str(source), source_file_sha256=source_hash, source_url=source_url,
            input_format=recording['input_format'], native_sampling_rate_hz=native_rate,
            crop_native_samples=[start, stop], crop_stop_exclusive=True,
            raw_first_sample=recording['first_sample'],
            crop_acquisition_samples=[recording['first_sample'] + start, recording['first_sample'] + stop],
            original_channel_names=[recording['names'][index] for index in selection],
            selected_input_channel_indices=[recording['native_indices'][index] for index in selection],
            excluded_bad_channels=recording['bads'],
            channel_name_normalization='strip_whitespace_uppercase_remove_trailing_dots; no_anatomical_aliases',
            channel_types=types, native_filter_metadata=recording['native_filter_metadata'],
            processing_order=['native_crop', 'polyphase_resample' if ratio != 1 else 'unchanged_sampling_grid',
                              'EEG_average_reference' if reference == 'average' else 'reference_as_supplied', 'float32_export'],
            resampling=dict(method='scipy.signal.resample_poly' if ratio != 1 else 'none',
                up=ratio.numerator, down=ratio.denominator, antialias_window='kaiser_beta_5', padtype='line'),
            additional_bandpass_hz=None, temporal_demean=False, interpolation=False,
            n_samples=values.shape[1], duration_s=values.shape[1] / float(target_fs),
            role_note='Recorded neural background, not instrument noise' if noise_role == 'resting_background'
                      else 'User-declared acquisition/processing role; not inferred from waveforms')
        record = dict(record_id=record_id, recording_id=recording_id, path=str(output), array_key='noise',
            fs_hz=float(target_fs), channel_names=requested, sensor_units=units, reference=reference,
            noise_role=noise_role, split=split, provenance=provenance)
        if subject_id is not None:
            record['subject_id'] = subject_id
        serialized = json.dumps(record, indent=2, allow_nan=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            if output.suffix.lower() == '.npy':
                np.save(handle, values, allow_pickle=False)
            else:
                np.savez_compressed(handle, noise=values, channel_names=np.asarray(requested),
                    channel_types=np.asarray(types), channel_units=np.asarray(units), sampling_rate_hz=float(target_fs),
                    metadata_json=serialized)
        with sidecar.open('x', encoding='utf-8') as handle:
            handle.write(serialized + '\n')
        return record
    finally:
        recording['close']()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--subject-id')
    parser.add_argument('--noise-role', choices=NOISE_ROLES, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--target-fs', type=float, default=256.)
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--duration-s', type=float)
    parser.add_argument('--reference', choices=('average', 'as_supplied'), default='as_supplied')
    parser.add_argument('--channel-names-json', type=Path, help='JSON list of explicitly selected canonical channel names')
    parser.add_argument('--source-url', help='Provenance only; no network access is performed')
    parser.add_argument('--record-json', type=Path)
    parser.add_argument('--npz-key')
    parser.add_argument('--npz-channels-key')
    parser.add_argument('--npz-sfreq-key', default='sampling_rate_hz')
    parser.add_argument('--npz-types-key', default='channel_types')
    parser.add_argument('--npz-units-key', default='channel_units')
    options = vars(parser.parse_args())
    channel_path = options.pop('channel_names_json')
    options['channel_names'] = json.loads(channel_path.read_text(encoding='utf-8')) if channel_path else None
    record = prepare_real_noise(**options)
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
