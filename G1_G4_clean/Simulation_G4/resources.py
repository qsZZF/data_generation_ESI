"""Verified structural resources for the cortical AAL model."""
from pathlib import Path
import hashlib
import json

import numpy as np
import requests
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SC_RESOURCES = {
    "SCmatrices88healthy.mat": (
        "https://osf.io/download/6823g/",
        "6e68a62517490abdb63e451bfb48f3256abc31ed4d7591f7a85d0f566b9f2f64",
    ),
    "AAL_regions.csv": (
        "https://osf.io/download/6a8jx/",
        "c26dd24d785a88534186db280eff83f4a206743f1797a5282252b99e38fe3a0b",
    ),
}
SC_MIRROR = "https://raw.githubusercontent.com/carlosmig/Homo_DMF/main/SCmatrices88healthy.mat"


def fetch_connectivity(directory=ROOT / "Anatomy/G4_resources", *, include_region_table=False):
    """Download the author's CC-BY4.0 resources and verify OSF SHA256 checksums."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SC_RESOURCES.items():
        if name == "AAL_regions.csv" and not include_region_table:
            continue
        path = directory / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        urls = [url, SC_MIRROR] if name == "SCmatrices88healthy.mat" else [url]
        for index, address in enumerate(urls):
            try:
                with requests.get(address, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                break
            except requests.RequestException:
                if index == len(urls) - 1:
                    raise
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: {actual}")
        temporary.replace(path)
    manifest = {
        "source": "https://doi.org/10.17605/OSF.IO/YW5VF",
        "publication": "https://doi.org/10.1038/s41597-022-01596-9",
        "license": "CC-BY-4.0",
        "source_matrix_orientation": "source_target",
        "checksum_verified_mirror": SC_MIRROR,
        "region_order_reference": "Simulation_G4/reference/aal_regions.csv",
        "original_region_table_downloaded": (directory / "AAL_regions.csv").exists(),
        "files": {name: {"url": url, "sha256": sha} for name, (url, sha) in SC_RESOURCES.items()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return directory


def load_cortical_connectivity(directory, aal_roi_numbers):
    """Subset BOTH matrix axes using verified original one-based AAL90 ROI numbers."""
    data = loadmat(Path(directory) / "SCmatrices88healthy.mat")["SCmatrices"]
    if data.shape != (88, 90, 90):
        raise ValueError(f"Expected the published (88,90,90) matrices, got {data.shape}")
    indices = np.asarray(aal_roi_numbers, dtype=int) - 1
    if len(set(indices.tolist())) != len(indices) or np.any((indices < 0) | (indices >= 90)):
        raise ValueError("AAL ROI numbers must be unique values from 1 through 90")
    # Publication uses rows as streamline sources. Simulator uses target,source.
    mean = np.asarray(data, dtype=float).mean(axis=0).T
    selected = mean[np.ix_(indices, indices)]
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Connectivity must be finite and nonnegative")
    return selected


if __name__ == "__main__":
    directory = fetch_connectivity()
    matrix = loadmat(directory / "SCmatrices88healthy.mat")["SCmatrices"]
    print(json.dumps({"directory": str(directory), "matrix_shape": list(matrix.shape)}))
