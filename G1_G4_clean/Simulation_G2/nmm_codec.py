"""Regional bank codecs with exact float32 foreground and optional lossy background."""
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _protected_indices(values, n_regions):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("protected_regions must be a sequence of unique region indices")
    if any(not _integer(i) or i < 0 or i >= n_regions for i in values):
        raise ValueError("Invalid protected region index")
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
        raise ValueError("protected_regions must be one-dimensional and unique")
    return indices


def _background_indices(protected, n_regions):
    return np.setdiff1d(np.arange(n_regions), protected, assume_unique=True)


def encode_regions(source, storage):
    """Encode canonical float32 rows; fixed-rank loss is never silently thresholded."""
    source = np.asarray(source)
    if (source.ndim != 2 or min(source.shape) < 1 or source.shape[1] < 2
            or source.dtype.kind not in "fiu" or not np.isfinite(source).all()):
        raise ValueError("Source must be a finite real regional matrix with at least two samples")
    with np.errstate(over="raise", invalid="raise"):
        source = source.astype(np.float32)
    if not isinstance(storage, dict) or set(storage) - {"codec", "protected_regions", "rank"}:
        raise ValueError("Invalid regional storage options")
    codec = storage.get("codec")
    if codec not in {"dense", "int16", "lowrank"}:
        raise ValueError("Unsupported regional codec")
    protected = _protected_indices(storage.get("protected_regions", []), source.shape[0])
    background = source[_background_indices(protected, source.shape[0])]
    payload = {"protected_source": source[protected]}
    metadata = {"codec": codec, "protected_regions": protected.tolist()}
    if codec == "dense":
        payload["background_source"] = background
    elif codec == "int16":
        values = background.astype(np.float64)
        offset = (values.max(axis=1) + values.min(axis=1)) / 2
        scale = (values.max(axis=1) - values.min(axis=1)) / (2 * 32767)
        scale[scale == 0] = 1
        payload.update(background_offset=offset, background_scale=scale,
                       background_quantized=np.rint((values - offset[:, None]) / scale[:, None]).astype(np.int16))
    else:
        rank = storage.get("rank")
        if not _integer(rank) or rank < 0:
            raise ValueError("Lowrank storage requires a nonnegative integer rank")
        rank_stored = min(int(rank), background.shape[0], source.shape[1] - 1)
        mean = background.mean(axis=1, dtype=np.float64)
        centered = background.astype(np.float64) - mean[:, None]
        if rank_stored:
            spatial, singular, temporal = np.linalg.svd(centered, full_matrices=False)
            spatial = spatial[:, :rank_stored]
            temporal = singular[:rank_stored, None] * temporal[:rank_stored]
        else:
            spatial = np.empty((len(background), 0))
            temporal = np.empty((0, source.shape[1]))
        payload.update(background_mean=mean, background_spatial=spatial.astype(np.float32),
                       background_temporal=temporal.astype(np.float32))
        metadata.update(rank_requested=int(rank), rank_stored=rank_stored)
    return payload, metadata


def validate_regions_payload(payload, metadata, n_regions, n_samples):
    """Validate all encoded arrays without allocating a decoded regional matrix."""
    if (not _integer(n_regions) or n_regions < 1 or not _integer(n_samples) or n_samples < 2
            or not isinstance(metadata, dict)):
        raise ValueError("Invalid regional codec dimensions or metadata")
    codec = metadata.get("codec")
    if codec not in {"dense", "int16", "lowrank"}:
        raise ValueError("Unsupported regional codec")
    expected_metadata = {"codec", "protected_regions"}
    if codec == "lowrank":
        expected_metadata.update(("rank_requested", "rank_stored"))
    if set(metadata) != expected_metadata:
        raise ValueError("Invalid regional codec metadata fields")
    protected = _protected_indices(metadata["protected_regions"], n_regions)
    background_indices = _background_indices(protected, n_regions)
    n_background = len(background_indices)
    expected_keys = {"protected_source"}

    def array(name, shape, dtype):
        expected_keys.add(name)
        if name not in payload:
            raise ValueError(f"Missing codec array: {name}")
        values = np.asarray(payload[name])
        if values.shape != shape or values.dtype != np.dtype(dtype) or not np.isfinite(values).all():
            raise ValueError(f"Invalid codec array: {name}")
        return values

    array("protected_source", (len(protected), n_samples), np.float32)
    if codec == "dense":
        array("background_source", (n_background, n_samples), np.float32)
    elif codec == "int16":
        array("background_offset", (n_background,), np.float64)
        scale = array("background_scale", (n_background,), np.float64)
        if np.any(scale <= 0):
            raise ValueError("Quantization scales must be positive")
        array("background_quantized", (n_background, n_samples), np.int16)
    else:
        requested, rank = metadata.get("rank_requested"), metadata.get("rank_stored")
        if (not _integer(requested) or requested < 0 or not _integer(rank)
                or rank != min(requested, n_background, n_samples - 1)):
            raise ValueError("Invalid lowrank dimensions")
        array("background_mean", (n_background,), np.float64)
        array("background_spatial", (n_background, rank), np.float32)
        array("background_temporal", (rank, n_samples), np.float32)
    if set(payload) != expected_keys:
        raise ValueError("Unexpected codec arrays")
    return protected, background_indices


def _decode_regions_window(payload, metadata, n_regions, start, stop):
    """Decode a window of an already validated, immutable payload."""
    protected = np.asarray(metadata["protected_regions"], dtype=np.int64)
    background_indices = _background_indices(protected, n_regions)
    codec = metadata["codec"]
    if codec == "dense":
        background = payload["background_source"][:, start:stop]
    elif codec == "int16":
        background = (payload["background_offset"][:, None]
                      + payload["background_scale"][:, None]
                      * payload["background_quantized"][:, start:stop])
    else:
        background = (payload["background_spatial"].astype(np.float64)
                      @ payload["background_temporal"][:, start:stop].astype(np.float64)
                      + payload["background_mean"][:, None])
    source = np.empty((n_regions, stop - start), dtype=np.float32)
    source[protected] = payload["protected_source"][:, start:stop]
    with np.errstate(over="raise", invalid="raise"):
        source[background_indices] = background
    if not np.isfinite(source).all():
        raise ValueError("Decoded source contains nonfinite values")
    return source


def validate_window(start, stop, n_samples):
    """Half-open sample boundaries; an empty or out-of-record window is invalid."""
    if (not _integer(start) or not _integer(stop)
            or not 0 <= start < stop <= n_samples):
        raise ValueError("Window requires integer boundaries 0 <= start < stop <= n_samples")


def decode_regions(payload, metadata, n_regions, n_samples, *, start=0, stop=None):
    """Validate the complete payload, then decode only the requested columns."""
    validate_regions_payload(payload, metadata, n_regions, n_samples)
    stop = n_samples if stop is None else stop
    validate_window(start, stop, n_samples)
    return _decode_regions_window(payload, metadata, n_regions, start, stop)
