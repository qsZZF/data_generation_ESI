"""Export a pinned, real MNE sample MEG/EEG forward and noise covariance.

Only two small testing files are fetched. No whitening, covariance flooring,
regularization, sensor projection or SSS is applied to the exported operators.
The companion sensor_projector must be applied to both forward and covariance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
import mne
from mne._fiff.constants import FIFF
from mne._fiff.proj import make_projector
import numpy as np
import requests


COMMIT = "33ef2a6f6345a1adf8764f8f63725f0ed9e4bb92"
FILES = (
    ("sample_audvis_trunc-meg-eeg-oct-4-fwd.fif", 20_982_526,
     "3fd330f473cb43ede484df8df0521b7efa814d70b8bd38a0361ff5afae2b1539",
     "ad4327a8ab511662374a2874537c57819aa4325a"),
    ("sample_audvis_trunc-cov.fif", 547_354,
     "d1550d4c5830498ddc9f9eabc2c18bf433622035552bee11cd5657f429f39953",
     "b430aa596e06230fd93abf578572998bf9e29ae5"),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "generated_simulation_dataset/nullspace_study/mne_sample"


def verify_file(path, expected_size, expected_hash):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected file size: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Pinned SHA256 mismatch: {path}")


def fetch_files(directory):
    manifest_path = directory / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    prior_files = {item["name"]: item for item in previous.get("files", [])}
    entries = []
    for name, size, digest, blob in FILES:
        path = directory / name
        canonical = f"https://raw.githubusercontent.com/mne-tools/mne-testing-data/{COMMIT}/MEG/sample/{name}"
        api = f"https://api.github.com/repos/mne-tools/mne-testing-data/git/blobs/{blob}"
        transport = prior_files.get(name, {}).get("transport_url", canonical)
        if not path.exists():
            temporary = path.with_suffix(".part")
            for attempt, transport in enumerate((canonical, api, api)):
                try:
                    with requests.get(transport, stream=True, timeout=(15, 45),
                                      headers={"Accept": "application/vnd.github.raw+json"}) as response:
                        response.raise_for_status()
                        received = 0
                        with temporary.open("wb") as handle:
                            for chunk in response.iter_content(65_536):
                                received += len(chunk)
                                if received > size:
                                    raise ValueError(f"Download exceeds pinned file size: {name}")
                                handle.write(chunk)
                    verify_file(temporary, size, digest)
                    temporary.replace(path)
                    break
                except (requests.RequestException, OSError, ValueError):
                    if attempt == 2:
                        raise
        verify_file(path, size, digest)
        entries.append(dict(name=name, url=canonical, transport_url=transport,
                            bytes=size, sha256=digest, git_blob_sha1=blob))
    manifest = dict(repository="https://github.com/mne-tools/mne-testing-data",
                    commit=COMMIT, files=entries)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def covariance_diagnostics(covariance, relative_tolerance=1e-8):
    """Diagnose rank after per-channel variance scaling, never by mixed SI units."""
    std = np.sqrt(np.diag(covariance))
    if not np.isfinite(covariance).all() or np.any(std <= 0):
        raise ValueError("Covariance needs finite values and positive diagonal")
    standardized = covariance / std[:, None] / std[None, :]
    eig = np.linalg.eigvalsh(standardized)
    tolerance = float(eig[-1] * relative_tolerance)
    if eig[0] < -tolerance:
        raise ValueError("Covariance is not positive semidefinite within the stated tolerance")
    return dict(rank=int(np.sum(eig > tolerance)), n_channels=len(std),
                minimum_eigenvalue=float(eig[0]), maximum_eigenvalue=float(eig[-1]),
                relative_eigenvalue_tolerance=relative_tolerance,
                scaling="per-channel sqrt(diag(C)); rank is not computed on mixed SI units",
                tolerance_reason="Rank reporting accommodates stored FIF covariance rounding; exported entries are unchanged")


def export_forward(directory, manifest):
    original = mne.read_forward_solution(directory / FILES[0][0], verbose="ERROR")
    covariance = mne.read_cov(directory / FILES[1][0], verbose="ERROR")
    free = mne.convert_forward_solution(original, surf_ori=False, force_fixed=False,
                                        copy=True, verbose="ERROR")
    fixed = mne.convert_forward_solution(original, surf_ori=True, force_fixed=True,
                                         use_cps=False, copy=True, verbose="ERROR")
    bads = set(free["info"]["bads"]) | set(covariance["bads"])
    names = [name for name in free["sol"]["row_names"]
             if name in covariance["names"] and name not in bads]
    if not names or len(set(names)) != len(names):
        raise ValueError("No unique good common channels")
    forward_rows = [free["sol"]["row_names"].index(name) for name in names]
    cov_rows = [covariance["names"].index(name) for name in names]
    info_rows = [free["info"]["ch_names"].index(name) for name in names]
    gain_free = np.asarray(free["sol"]["data"][forward_rows], dtype=np.float64)
    mne_fixed = np.asarray(fixed["sol"]["data"][forward_rows], dtype=np.float64)
    raw_cov = covariance["data"]
    if covariance["diag"]:
        raw_cov = np.diag(raw_cov)
    noise_cov = np.asarray(raw_cov[np.ix_(cov_rows, cov_rows)], dtype=np.float64)
    if not np.allclose(noise_cov, noise_cov.T, rtol=1e-12, atol=0):
        raise ValueError("Input covariance is not symmetric")
    # Internal MNE helper implements the same channel-restricted SSP construction
    # used by MNE's covariance/inverse preparation; include active SSP vectors.
    projectors = list(covariance["projs"]) + list(free["info"].get("projs", []))
    projector, n_projectors, basis = make_projector(projectors, names, include_active=True)
    if not np.allclose(projector @ projector, projector, atol=1e-12):
        raise ValueError("SSP operator is not idempotent")
    projected_cov = projector @ noise_cov @ projector.T
    channel_types = np.asarray([mne.channel_type(free["info"], row) for row in info_rows])
    positions = []
    for row in info_rows:
        channel = free["info"]["chs"][row]
        position = channel["loc"][:3]
        if channel["coord_frame"] == FIFF.FIFFV_COORD_DEVICE:
            position = mne.transforms.apply_trans(free["info"]["dev_head_t"], position)
        elif channel["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError(f"Unsupported sensor coordinate frame: {channel['coord_frame']}")
        positions.append(position)
    if free["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise ValueError("Expected a forward already expressed in head coordinates")
    normals = np.asarray(fixed["source_nn"], dtype=np.float64)
    gain_fixed = np.einsum("cnk,nk->cn", gain_free.reshape(len(names), -1, 3), normals)
    row_scale = np.linalg.norm(gain_fixed, axis=1)
    fixed_error = np.linalg.norm(mne_fixed - gain_fixed, axis=1) / row_scale
    if np.max(fixed_error) > 1e-6:
        raise ValueError("Fixed forward disagrees with Cartesian forward dotted with normals")
    std = np.sqrt(np.diag(noise_cov))
    projection_delta = (projected_cov - noise_cov) / std[:, None] / std[None, :]
    standardized_cov = noise_cov / std[:, None] / std[None, :]
    metadata = dict(manifest, mne_version=mne.__version__, source_subject="sample",
                    source_space="oct-4, actual retained vertices; not fsaverage5",
                    coordinate_frame="head", spatial_units="m", source_current_units="A m",
                    channel_unit_by_type={"mag": "T", "grad": "T/m", "eeg": "V"},
                    channel_counts={str(kind): int(np.sum(channel_types == kind)) for kind in np.unique(channel_types)},
                    excluded_bads=sorted(bads), n_sources=int(free["nsource"]),
                    covariance_nfree=int(covariance["nfree"]),
                    covariance_method="empirical covariance supplied by MNE testing data; no new estimation",
                    covariance_export="unaltered good-channel submatrix; not regularized, floored or whitened",
                    covariance_projectors=[dict(description=p["desc"], active=bool(p["active"])) for p in covariance["projs"]],
                    ssp_independent_vectors=int(n_projectors),
                    ssp_application="Apply P to G and P C P.T before whitening; P is exported separately",
                    sss_applied_by_exporter=False,
                    sss_history="No SSS assumption; acquisition preprocessing history not established by these two FIF files",
                    fixed_orientation="surface normal, use_cps=False; source_normals are in head coordinates",
                    fixed_construction="float64 Cartesian forward contracted with saved float64 source_normals",
                    original_fif_numeric_precision="not independently established; promoting loaded arrays does not recover acquisition or storage precision",
                    mne_fixed_conversion_dtype=str(fixed["sol"]["data"].dtype),
                    mne_fixed_vs_float64_contraction_max_row_relative_error=float(np.max(fixed_error)),
                    fixed_conversion_max_row_relative_error=float(np.max(fixed_error)),
                    raw_covariance_rank=covariance_diagnostics(noise_cov),
                    projected_covariance_rank=covariance_diagnostics(projected_cov),
                    standardized_relative_projection_change=float(np.linalg.norm(projection_delta) / np.linalg.norm(standardized_cov)),
                    license_status="No explicit license file found at repository root or MEG/sample for this pinned commit",
                    usage_note="Official MNE sample documentation restricts use to software familiarization and excludes evaluating MEG/MRI system performance",
                    usage_source="https://mne.tools/stable/documentation/datasets.html#sample")
    for kind in np.unique(channel_types):
        indices = np.flatnonzero(channel_types == kind)
        metadata.setdefault("rank_by_channel_type", {})[str(kind)] = dict(
            raw=covariance_diagnostics(noise_cov[np.ix_(indices, indices)]),
            projected=covariance_diagnostics(projected_cov[np.ix_(indices, indices)]))
    arrays = dict(leadfield_fixed=gain_fixed, leadfield_free=gain_free,
                  source_rr_m=np.asarray(free["source_rr"], dtype=np.float64), source_normals=normals,
                  source_hemi=np.concatenate([np.full(len(src["vertno"]), hemi) for src, hemi in zip(free["src"], ("lh", "rh"))]),
                  source_vertex_ids=np.concatenate([src["vertno"] for src in free["src"]]),
                  channel_names=np.asarray(names), channel_types=channel_types,
                  channel_positions_head_m=np.asarray(positions), noise_cov=noise_cov,
                  channel_std=std, sensor_projector=projector, sensor_projector_basis=basis,
                  metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    for name, array in arrays.items():
        if array.dtype.hasobject or (np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
            raise ValueError(f"Invalid output array: {name}")
    output = directory / "mne_sample_forward.npz"
    temporary = output.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    (directory / "forward_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output), channels=metadata["channel_counts"],
                          sources=metadata["n_sources"], raw_rank=metadata["raw_covariance_rank"]["rank"],
                          projected_rank=metadata["projected_covariance_rank"]["rank"]), indent=2))
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_forward(args.output_dir, fetch_files(args.output_dir))


if __name__ == "__main__":
    main()
