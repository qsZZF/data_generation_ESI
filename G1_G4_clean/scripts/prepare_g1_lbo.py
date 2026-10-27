"""Cache LBO modes on the exact Brainstorm mesh used by the G1 leadfield."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation_G1.geometry import compute_basis, load_basis, save_basis
from Simulation_G1.g1_simulation_core import normalize_cortex, coordinate_unit_scale_to_mm
from Simulation_G2.g2_simulation_core import load_brainstorm_cortex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cortex',type=Path,default=ROOT/'Anatomy/Cortex_FS_20k.mat')
    parser.add_argument('--output',type=Path,default=ROOT/'Anatomy/G1_LBO/fsaverage5_lbo256.npz')
    parser.add_argument('--modes',type=int,default=256,help='Total retained nonconstant modes across both hemispheres')
    args=parser.parse_args()
    cortex=load_brainstorm_cortex(args.cortex)
    vertices,faces,_=normalize_cortex(cortex['Vertices'],cortex['Faces'],cortex['VertConn'])
    vertices=vertices*coordinate_unit_scale_to_mm(vertices)
    if args.output.exists():
        bundle=load_basis(args.output,vertices_mm=vertices,faces=faces)
        if bundle['basis'].shape[1]!=args.modes:
            raise ValueError('Existing cache has a different mode count; use a new output path')
    else:
        bundle=compute_basis(vertices,faces,n_modes=args.modes)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        save_basis(args.output,bundle)
    print(json.dumps(dict(basis_path=str(args.output),basis_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          n_modes=args.modes,metadata=bundle['metadata']),indent=2))


if __name__=='__main__':main()
