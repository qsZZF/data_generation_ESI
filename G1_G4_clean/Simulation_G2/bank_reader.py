"""Bounded window decoding for immutable local NMM banks.

NPZ is not chunked storage: a cache miss reads/decompresses the complete encoded
payload. Cached lowrank/int16 arrays decode only the requested time window.
Files must remain immutable during training; stat changes trigger revalidation,
while index changes require constructing a new reader.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
from pathlib import Path
from threading import RLock


from Simulation_G2.nmm_bank import _valid_id, load_bank_payload, read_bank
from Simulation_G2.nmm_codec import _decode_regions_window, _integer, validate_window


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


class BankWindowReader:
    """Read owned float32 regional windows at the recording's native rate."""

    def __init__(self, bank_dir, max_cache_bytes=256 * 1024**2):
        if not _integer(max_cache_bytes) or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.bank_dir = Path(bank_dir).resolve()
        self.max_cache_bytes = int(max_cache_bytes)
        self._index_path = self.bank_dir / "bank.json"
        before = self._index_path.read_bytes()
        self._bank = read_bank(self.bank_dir)
        if self._index_path.read_bytes() != before:
            raise ValueError("NMM bank index changed while opening")
        self.bank_sha256 = hashlib.sha256(before).hexdigest()
        self._index_signature = _signature(self._index_path)
        self._cache = OrderedDict()
        self._bytes = self._hits = self._misses = 0
        self._lock = RLock()

    @property
    def bank(self):
        """Return a detached copy, preventing callers from retargeting the reader."""
        return deepcopy(self._bank)

    def _check_index(self, *, hash_check=False):
        if (_signature(self._index_path) != self._index_signature
                or hash_check and hashlib.sha256(self._index_path.read_bytes()).hexdigest() != self.bank_sha256):
            raise ValueError("NMM bank index changed; construct a new reader")

    def read(self, sample_id, start_sample, stop_sample):
        """Return ``(regional_window, entry)`` for [start_sample, stop_sample)."""
        with self._lock:
            self._check_index()
            if not _valid_id(sample_id) or sample_id not in self._bank["samples"]:
                raise ValueError("Unknown or invalid NMM bank sample identifier")
            entry = self._bank["samples"][sample_id]
            validate_window(start_sample, stop_sample, entry["n_samples"])
            # Validate before stat: index filenames must never escape the bank.
            if entry.get("source_file") != f"{sample_id}.npz":
                raise ValueError("Unsafe source path in NMM bank")
            source_path = self.bank_dir / entry["source_file"]
            if not source_path.resolve().is_relative_to(self.bank_dir):
                raise ValueError("NMM bank recording escapes its directory")
            signature = _signature(source_path)
            cached = self._cache.pop(sample_id, None)
            if cached is not None and cached[0] != signature:
                self._bytes -= cached[2]
                cached = None
            if cached is None:
                self._misses += 1
                self._check_index(hash_check=True)
                payload, _, _ = load_bank_payload(self.bank_dir, sample_id, bank=self._bank)
                if _signature(source_path) != signature:
                    raise ValueError("NMM recording changed while reading")
                self._check_index()
                size = sum(array.nbytes for array in payload.values())
                for array in payload.values():
                    array.flags.writeable = False
                cached = (signature, payload, size)
                if size <= self.max_cache_bytes:
                    while self._cache and self._bytes + size > self.max_cache_bytes:
                        _, removed = self._cache.popitem(last=False)
                        self._bytes -= removed[2]
                    self._cache[sample_id] = cached
                    self._bytes += size
            else:
                self._hits += 1
                self._cache[sample_id] = cached
            payload = cached[1]
            if self._bank["schema_version"] == 2:
                source = _decode_regions_window(payload, entry["storage"], self._bank["n_regions"],
                                                start_sample, stop_sample)
            else:
                source = payload["region_source"][:, start_sample:stop_sample].copy()
            return source, deepcopy(entry)

    def cache_info(self):
        with self._lock:
            return {"bytes": self._bytes, "entries": len(self._cache),
                    "hits": self._hits, "misses": self._misses,
                    "max_cache_bytes": self.max_cache_bytes}

    def clear(self):
        """Release cached arrays and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._bytes = self._hits = self._misses = 0
