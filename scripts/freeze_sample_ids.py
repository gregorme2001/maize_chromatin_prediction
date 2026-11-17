"""One-time GEO header extraction; does not read phenotype values or run models."""

import argparse
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

from maize_regulatory.downloads import GEO_ARCHIVE
from maize_regulatory.sample_identity import (
    RESOURCE_DIR,
    freeze_identities,
    genotype_header_from_archive,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing GEO archive instead of network")
    parser.add_argument("--phenotypes", type=Path,
                        default=Path("data/raw/maize282_blup_phenotypes.csv"))
    args = parser.parse_args()
    url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE273nnn/GSE273478/suppl/" + GEO_ARCHIVE)
    if args.archive:
        with args.archive.open("rb") as stream:
            header = genotype_header_from_archive(stream)
        source = str(args.archive)
    else:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={"User-Agent": "maize-regulatory/0.1"})
        print("Reading the GEO genotype header; no accessibility values are retained", flush=True)
        with urllib.request.urlopen(request, timeout=120, context=context) as stream:
            header = genotype_header_from_archive(stream)
        source = url
    names = pd.read_csv(args.phenotypes, usecols=["Family_Inbred_Name"])["Family_Inbred_Name"]
    freeze_identities(header, names, RESOURCE_DIR, source)
    print(f"Saved 172 published names, 169 analysis names, and 30 holdout IDs in {RESOURCE_DIR}")


if __name__ == "__main__":
    main()
