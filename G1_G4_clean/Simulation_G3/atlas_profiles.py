"""Named atlas assets shared by the G3 producer and G2 projection interface."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
from scipy.sparse import load_npz
from Simulation_G2.nmm_bank import projection_hash

ROOT = Path(__file__).resolve().parents[1]
ATLAS_CHOICES = ("legacy998", "aal78", "aal78_membership")


def get_atlas_profile(name=None):
    if name in (None, "legacy998"):
        return dict(atlas_id="legacy998_export994", n_regions=994, n_regions_raw=998,
                    connectivity_path=ROOT / "Simulation_G3/anatomy/connectivity_998.zip",
                    projection_path=ROOT / "Anatomy/FS_994nmm_to_20k.mat",
                    reshape_to_994=True, region_names=None)
    if name not in ("aal78", "aal78_membership"):
        raise ValueError(f"Unknown atlas {name!r}; choose from {ATLAS_CHOICES}")
    membership = name == 'aal78_membership'
    atlas_dir = ROOT / ('Anatomy/G4_AAL_cortical_membership' if membership else 'Anatomy/G4_AAL_cortical')
    atlas_path = atlas_dir / 'atlas.npz'
    projection_path = atlas_dir / 'projection.npz'
    connectivity_path = ROOT / ('Anatomy/G3_AAL78_membership_TVB/connectivity_78.zip' if membership
                                else 'Anatomy/G3_AAL78_TVB/connectivity_78.zip')
    metadata = load_connectivity_metadata(connectivity_path)
    if metadata is None:
        raise ValueError('Build the verified AAL bundle first: python -m Simulation_G3.prepare_aal_tvb')
    expected_atlas = [digest for path, digest in metadata['source_files_sha256'].items()
                      if Path(path).name == 'atlas.npz']
    if expected_atlas != [hashlib.sha256(atlas_path.read_bytes()).hexdigest()]:
        raise ValueError('AAL atlas bundle differs from the connectivity build')
    if projection_hash(load_npz(projection_path)) != metadata['projection_sha256']:
        raise ValueError('AAL projection differs from the connectivity build')
    expected_mode = 'membership' if membership else 'area_unit_sum'
    manifest_path = atlas_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    with np.load(atlas_path, allow_pickle=False) as atlas:
        names = atlas["region_names"].tolist()
        array_mode = str(atlas['projection_mode'].item()) if 'projection_mode' in atlas else 'area_unit_sum'
    if (array_mode != expected_mode or manifest.get('projection_mode', 'area_unit_sum') != expected_mode
            or metadata.get('projection_mode', 'area_unit_sum') != expected_mode):
        raise ValueError('AAL projection mode differs from the named atlas or connectivity sidecar')
    if len(names) != 78 or len(set(names)) != 78:
        raise ValueError("Expected 78 distinct cortical AAL parcel names")
    return dict(atlas_id="aal78_cortical_fsaverage5_regfusion" + ('_membership' if membership else ''),
                projection_mode=expected_mode, n_regions=78, n_regions_raw=78,
                connectivity_path=connectivity_path, projection_path=projection_path,
                reshape_to_994=False, region_names=names)


def validate_atlas_connectivity(profile, path):
    """A matching matrix size alone cannot establish matching parcel order."""
    with ZipFile(path) as archive:
        names = np.loadtxt(archive.open('centres.txt'), dtype=str, ndmin=2)[:, 0].tolist()
    if len(names) != profile['n_regions_raw']:
        raise ValueError('Connectivity region count differs from the named atlas')
    if profile['region_names'] is not None and names != profile['region_names']:
        raise ValueError('Connectivity region order differs from the named atlas')


def load_connectivity_metadata(path):
    """Carry verified bundle assumptions into each simulation, if supplied."""
    path = Path(path)
    sidecar = path.parent / 'manifest.json'
    if not sidecar.exists():
        return None
    content = sidecar.read_bytes()
    metadata = json.loads(content)
    if metadata.get('connectivity_file') != path.name:
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('connectivity_sha256'):
        raise ValueError('Connectivity checksum differs from its bundle manifest')
    return {**metadata, 'manifest_sha256': hashlib.sha256(content).hexdigest()}
