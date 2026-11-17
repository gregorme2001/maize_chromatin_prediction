"""Explicit source downloads; importing this module never accesses the network."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

GEO_ARCHIVE = "GSE273478_normalized_chromatin_accessibility_and_covariate_matrices.tar.gz"
SOURCES = {
    "cells": ("maize_282_scATAC_ACRs_x_barcodes.sparse.rds",
              "https://drive.google.com/uc?id=18op-hjidkSCjMxRTmc2VjP00Hhqfg9MR"),
    "metadata": ("Dryad_Data_S2.xlsx", "https://datadryad.org/downloads/file_stream/3746598"),
    "genome": ("Zm-B73-REFERENCE-NAM-5.0.fa.gz",
               ("https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/"
                "Zm-B73-REFERENCE-NAM-5.0.fa.gz")),
    "motifs": ("jaspar_plants_2026.txt",
               ("https://ncmm.jaspar.uiocloud.no/download/data/2026/CORE/"
                "JASPAR2026_CORE_plants_non-redundant_pfms_jaspar.txt")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(key: str, raw_dir: Path, *, overwrite: bool = False) -> None:
    filename, url = SOURCES[key]
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / filename
    if target.exists() and not overwrite:
        print(f"Keeping existing {target}")
        return
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {key}: {url}", flush=True)
    if key == "cells":
        import gdown

        result = gdown.download(url, str(partial), quiet=False)
        if result is None:
            raise RuntimeError("Google Drive download failed; supply the original RDS manually")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "maize-regulatory/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    with partial.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    if not prefix or prefix.startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"Download returned empty content or an HTML page: {url}")
    partial.replace(target)
    receipt = {"url": url, "bytes": target.stat().st_size, "sha256": sha256(target)}
    target.with_name(target.name + ".download.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--list", action="store_true", help="List sources without downloading")
    parser.add_argument("--only", choices=list(SOURCES), nargs="+")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing downloaded inputs")
    args = parser.parse_args()
    keys = args.only or list(SOURCES)
    if args.only is None and (args.raw_dir / "cell_assignments.csv").is_file():
        keys = [key for key in keys if key != "metadata"]
        print("Using supplied cell_assignments.csv; skipping the optional Dryad workbook")
    if args.list:
        for key in keys:
            filename, url = SOURCES[key]
            print(f"{key}: {args.raw_dir / filename}\n  {url}")
        return
    failures = []
    for key in keys:
        try:
            download(key, args.raw_dir, overwrite=args.overwrite)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            failures.append(key)
            print(f"FAILED {key}: {error}")
    if failures:
        raise SystemExit("Downloads incomplete: " + ", ".join(failures))
