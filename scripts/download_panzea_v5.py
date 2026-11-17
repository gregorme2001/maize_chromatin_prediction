#!/usr/bin/env python
"""Download the public 282-panel chromosome VCFs/indexes, verifying published hashes."""

import argparse
import json
from pathlib import Path
import shutil

from maize_regulatory.panzea_downloads import discover_sources, download_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--chromosomes', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw/panzea_v5'))
    parser.add_argument('--manifest', type=Path, default=Path('configs/panzea_v5_sources.json'))
    args = parser.parse_args()
    if args.discover or not args.manifest.exists():
        manifest = discover_sources()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    else:
        manifest = json.loads(args.manifest.read_text())
    entries = [entry for entry in manifest['files'] if entry['chromosome'] in args.chromosomes]
    required = sum(entry['bytes'] for entry in entries if not (args.raw_dir/entry['name']).exists())
    if required+2*1024**3 > shutil.disk_usage(Path.cwd()).free:
        raise RuntimeError('Insufficient free disk for requested downloads plus 2-GiB working reserve')
    print(f'Selected {len(entries)} files; missing compressed inputs {required/1e9:.2f} GB', flush=True)
    for entry in entries:
        download_entry(entry, args.raw_dir)


if __name__ == '__main__':
    main()
