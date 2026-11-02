"""Build offline anatomical FEM support dictionaries for the nullspace study."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resolution_benchmark.support_bases import prepare_basis_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', type=Path, required=True)
    parser.add_argument('--source-fif', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes-per-hemisphere', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing generated basis archive')
    args = parser.parse_args(argv)
    _, metadata = prepare_basis_archive(args.forward, args.source_fif, args.output,
                                       args.modes_per_hemisphere, overwrite=args.overwrite)
    print(json.dumps(dict(output=str(args.output), **metadata), indent=2))


if __name__ == '__main__':
    main()
