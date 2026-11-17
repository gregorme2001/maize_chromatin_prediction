"""Rebuild legacy input artifacts; never fit phenotype models or report DTS values."""

import argparse
import gzip
import json
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .downloads import sha256
from .evaluation import make_holdout_indices
from .preparation import (
    align_cpm,
    ancestry_pcs,
    match_phenotypes,
    normalize_reconstructed_genotype,
    require_unique,
)
from .sample_identity import load_identities, validate_analysis_order
from .spatial import parse_acr_coordinates

GENOME = "Zm-B73-REFERENCE-NAM-5.0.fa"
PHENOTYPE_WORKBOOK = "Peiffer2014Genetics_blupPhenos20150325.xlsx"
CELL_MATRIX = "maize_282_scATAC_ACRs_x_barcodes.sparse.rds"
ACCESSIBILITY_OUTPUTS = [
    "maize_accessibility_cpm.parquet", "maize_flowering_dts.parquet", "maize_ancestry_pcs.parquet",
]


def missing_inputs(data_dir: Path, stage: str) -> list[str]:
    raw = data_dir / "raw"
    requirements = []
    if stage in ("all", "accessibility", "cells", "cpm"):
        requirements.append([raw / CELL_MATRIX])
    if stage in ("all", "accessibility", "cpm"):
        requirements.append([raw / "cell_assignments.csv", raw / "Dryad_Data_S2.xlsx"])
    if stage in ("all", "accessibility"):
        requirements.extend([
            [raw / "maize282_blup_phenotypes.csv", raw / PHENOTYPE_WORKBOOK],
            [raw / "mdp_genotype.hmp.txt"],
        ])
    if stage in ("all", "sequences"):
        requirements.append([raw / GENOME, raw / (GENOME + ".gz")])
        if stage == "sequences":
            requirements.append([data_dir / "intermediate" / "cells" / "acr_index.csv"])
    if stage in ("all", "motifs"):
        requirements.append([raw / "jaspar_plants_2026.txt"])
        if stage == "motifs":
            requirements.append([data_dir / "maize_acr_sequences.parquet"])
    return [" OR ".join(str(p) for p in alternatives)
            for alternatives in requirements if not any(p.is_file() for p in alternatives)]


def refuse_existing(paths, *, overwrite: bool = False):
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Use --overwrite to replace stage outputs: " + ", ".join(existing))


def receipt(path: Path, inputs: list[Path], details: dict) -> None:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "biopython", "numba", "pyfaidx",
                 "openpyxl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    record = {
        "inputs": {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs},
        "pipeline_sources": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "packages": packages,
        **details,
    }
    path.write_text(json.dumps(record, indent=2))


def write_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Keep an existing output intact if an overwrite fails during serialization."""
    temporary = target.with_name(target.name + ".part")
    frame.to_parquet(temporary)
    temporary.replace(target)


def phenotype_source(raw: Path, intermediate: Path) -> Path:
    csv = raw / "maize282_blup_phenotypes.csv"
    if csv.exists():
        return csv
    workbook = raw / PHENOTYPE_WORKBOOK
    required = {"Panel", "Family_Inbred_Name", "DTS"}
    with pd.ExcelFile(workbook) as excel:
        matches = [s for s in excel.sheet_names
                   if required.issubset(pd.read_excel(excel, sheet_name=s, nrows=0).columns)]
        if len(matches) != 1:
            raise ValueError("Cannot identify a unique phenotype sheet; supply legacy CSV instead")
        frame = pd.read_excel(excel, sheet_name=matches[0])
    frame = frame.loc[frame["Panel"].eq("ASSO")].copy()
    if len(frame) != 282:
        raise ValueError("Expected 282 ASSO phenotype rows; refusing to change sample selection")
    target = intermediate / "maize282_blup_phenotypes.csv"
    frame.to_csv(target, index=False)
    return target


def metadata_source(raw: Path, intermediate: Path) -> Path:
    csv = raw / "cell_assignments.csv"
    if csv.exists():
        frame = pd.read_csv(csv)
    else:
        frame = pd.read_excel(raw / "Dryad_Data_S2.xlsx", sheet_name="Table_S5")
        columns = ["cellID", "genotype", "celltype_full", "library"]
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"Table_S5 requires {columns}; supply your legacy cell_assignments.csv "
                             "if this workbook has a different layout")
        frame = frame[columns].copy()
    required = ["cellID", "genotype"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Cell assignments require cellID and genotype")
    if frame[required].isna().any().any() or not frame["cellID"].is_unique:
        raise ValueError("Cell metadata has missing values or duplicate cell IDs")
    frame["normalized_genotype"] = frame["genotype"].map(normalize_reconstructed_genotype)
    target = intermediate / "cell_assignments.csv"
    frame.to_csv(target, index=False)  # Preserve source cell order and original labels.
    return target


def prepare_cells(data_dir: Path, *, overwrite: bool = False, require_labels: bool = True) -> Path:
    """Save a sparse cell matrix plus exact row/column labels; only R Matrix is needed."""
    raw, intermediate = data_dir / "raw", data_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    directory = intermediate / "cells"
    script = Path(__file__).resolve().parents[2] / "scripts" / "preserve_cell_accessibility.R"
    source_metadata = raw / "cell_assignments.csv"
    if not source_metadata.exists():
        source_metadata = raw / "Dryad_Data_S2.xlsx"
    has_labels = source_metadata.exists()
    if require_labels and not has_labels:
        raise FileNotFoundError("Supply Dryad_Data_S2.xlsx or cell_assignments.csv in data/raw/")
    source_paths = [raw / CELL_MATRIX, script, Path(__file__)]
    if has_labels:
        source_paths.append(source_metadata)
    sources = {str(p): sha256(p) for p in source_paths}
    artifacts = [directory / name for name in ["cell_by_acr.sparse.rds", "cell_metadata.csv",
                                               "acr_index.csv", "matrix_summary.csv"]]
    cache = directory / "sources.json"
    if not overwrite and all(p.exists() for p in [*artifacts, cache]):
        saved = json.loads(cache.read_text())
        if saved["sources"] != sources or any(saved["artifacts"].get(p.name) != sha256(p)
                                              for p in artifacts):
            raise ValueError("Sparse intermediate or sources changed; rerun with --overwrite")
        print(f"Reusing sparse matrix and cell labels in {directory}", flush=True)
        return directory
    refuse_existing(artifacts, overwrite=overwrite)
    cell_csv = metadata_source(raw, intermediate) if has_labels else "-"
    print("Saving sparse cell × ACR accessibility with cell IDs; genotype labels "
          + ("available" if has_labels else "pending Dryad metadata"), flush=True)
    subprocess.run(["Rscript", str(script), str(raw / CELL_MATRIX), str(cell_csv), str(directory)],
                   check=True)
    cache.write_text(json.dumps({"sources": sources,
                                 "artifacts": {p.name: sha256(p) for p in artifacts}}, indent=2))
    return directory


def prepare_cpm(data_dir: Path, *, overwrite: bool = False,
                reuse_cells: bool = False) -> Path:
    """Aggregate the saved sparse matrix with the original edgeR CPM definition."""
    raw, intermediate = data_dir / "raw", data_dir / "intermediate"
    cell_dir = prepare_cells(data_dir, overwrite=overwrite and not reuse_cells)
    cell_csv = cell_dir / "cell_metadata.csv"
    cpm_tsv = intermediate / "maize_ACR_genotype_cpm.tsv"
    cpm_receipt = intermediate / "cpm_sources.json"
    r_script = Path(__file__).resolve().parents[2] / "scripts" / "reconstruct_cpm.R"
    counts_path = cell_dir / "genotype_by_acr_counts.rds"
    cpm_sources = {str(p): sha256(p) for p in [raw / CELL_MATRIX, cell_csv, r_script]}
    if overwrite or not cpm_tsv.exists() or not counts_path.exists():
        temporary = cpm_tsv.with_name(cpm_tsv.name + ".part")
        print("Reconstructing genotype CPM in R (no quantile/INT transformation)", flush=True)
        subprocess.run(["Rscript", str(r_script), str(cell_dir), str(temporary)], check=True)
        temporary.replace(cpm_tsv)
        cpm_receipt.write_text(json.dumps({"sources": cpm_sources, "cpm_sha256": sha256(cpm_tsv),
                                           "counts_sha256": sha256(counts_path)},
                                          indent=2))
    else:
        if not cpm_receipt.exists():
            raise ValueError("Cached CPM lacks cpm_sources.json; move the TSV to rebuild it")
        saved = json.loads(cpm_receipt.read_text())
        if (saved.get("sources") != cpm_sources
                or saved.get("cpm_sha256") != sha256(cpm_tsv)
                or saved.get("counts_sha256") != sha256(counts_path)):
            raise ValueError("Cached CPM or its sources changed; move the TSV to rebuild it")
        print(f"Reusing {cpm_tsv}; move this file to force reconstruction", flush=True)
    return cpm_tsv


def prepare_accessibility(data_dir: Path, *, overwrite: bool = False,
                          reuse_intermediates: bool = False) -> None:
    raw, intermediate = data_dir / "raw", data_dir / "intermediate"
    targets = [data_dir / name for name in ACCESSIBILITY_OUTPUTS]
    refuse_existing(targets, overwrite=overwrite)
    intermediate.mkdir(parents=True, exist_ok=True)
    names, _, _, _, _ = load_identities()
    pheno_csv = phenotype_source(raw, intermediate)
    y, excluded = match_phenotypes(names, pd.read_csv(pheno_csv))
    if len(y) != 169:
        raise ValueError("Legacy matching must retain 169 genotypes; refusing new exclusions")
    validate_analysis_order(y.index)
    snp_file = raw / "mdp_genotype.hmp.txt"
    snps = pd.read_csv(snp_file, sep="\t", dtype=str)
    if snps.shape != (3093, 292):
        raise ValueError("Expected legacy HapMap: 3,093 SNPs, 11 metadata + 281 genotype columns")
    all_pcs = ancestry_pcs(snps)
    if not y.index.isin(all_pcs.index).all():
        raise ValueError("SNP panel lacks a required analysis genotype")
    pcs = all_pcs.loc[y.index].copy()
    cell_dir = prepare_cells(data_dir, overwrite=overwrite and not reuse_intermediates)
    cell_csv = cell_dir / "cell_metadata.csv"
    cpm_tsv = prepare_cpm(data_dir, overwrite=overwrite and not reuse_intermediates,
                          reuse_cells=True)
    r_script = Path(__file__).resolve().parents[2] / "scripts" / "reconstruct_cpm.R"
    X = align_cpm(pd.read_csv(cpm_tsv, sep="\t", index_col=0), y.index)
    if X.shape != (169, 108843) or not pcs.index.equals(X.index):
        raise ValueError("Legacy output dimensions or genotype alignment failed")
    development, holdout = make_holdout_indices(len(X))
    # Save identities only; no holdout phenotype values or summaries.
    identity = {
        "ordered_genotypes": X.index.tolist(),
        "development_genotypes": X.index[development].tolist(),
        "holdout_genotypes": X.index[holdout].tolist(),
        "excluded_published_genotypes": excluded,
        "split": {"test_size": 30, "random_state": 42},
        "ancestry_preprocessing": "Legacy full 281-line panel; no DTS used",
        "R_version": subprocess.run(["Rscript", "--version"], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, check=True).stdout.strip(),
    }
    write_parquet(X, targets[0])
    write_parquet(y, targets[1])
    write_parquet(pcs, targets[2])
    inputs = [pheno_csv, snp_file, raw / CELL_MATRIX, cell_csv, cpm_tsv, r_script]
    inputs += [p for p in (raw / PHENOTYPE_WORKBOOK, raw / "Dryad_Data_S2.xlsx") if p.exists()]
    receipt(intermediate / "accessibility_provenance.json", inputs, identity)
    print("Wrote accessibility, DTS, and ancestry parquets; no DTS values displayed")


def extract_sequences(acr_ids: pd.Index, genome) -> pd.DataFrame:
    """Legacy zero-based, end-exclusive extraction, preserving ACR identity/order."""
    require_unique(acr_ids, "ACR")
    coords = parse_acr_coordinates(acr_ids)
    records = []
    for chrom, start, end in coords[["chrom", "start", "end"]].itertuples(index=False, name=None):
        key = f"chr{chrom}"
        if key not in genome or start < 0 or end > len(genome[key]):
            raise ValueError(f"Invalid reference interval {key}:{start}-{end}")
        sequence = str(genome[key][start:end]).upper()
        if len(sequence) != end - start or set(sequence) - set("ACGT"):
            raise ValueError("Unexpected sequence length or non-ACGT bases; no silent exclusions")
        records.append(sequence)
    return pd.DataFrame({"sequence": records}, index=acr_ids.copy())


def prepare_sequences(data_dir: Path, *, overwrite: bool = False) -> None:
    from pyfaidx import Fasta

    target = data_dir / "maize_acr_sequences.parquet"
    refuse_existing([target], overwrite=overwrite)
    intermediate = data_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    raw_fasta = data_dir / "raw" / GENOME
    compressed = data_dir / "raw" / (GENOME + ".gz")
    source = raw_fasta if raw_fasta.exists() else compressed
    # A writable intermediate copy keeps pyfaidx's index out of the raw sources.
    fasta = intermediate / GENOME
    temporary = fasta.with_name(fasta.name + ".part")
    if source == raw_fasta:
        with source.open("rb") as handle, temporary.open("wb") as out:
            shutil.copyfileobj(handle, out)
    else:
        with gzip.open(source, "rb") as handle, temporary.open("wb") as out:
            shutil.copyfileobj(handle, out)
    temporary.replace(fasta)
    acr_path = intermediate / "cells" / "acr_index.csv"
    ids = pd.Index(pd.read_csv(acr_path)["ACR"], name="ACR")
    with Fasta(str(fasta), as_raw=True, sequence_always_upper=True, rebuild=True) as genome:
        sequences = extract_sequences(ids, genome)
    if len(sequences) != 108843 or sequences["sequence"].str.len().sum() != 18964435:
        raise ValueError("Reference sequences do not match legacy ACR count/total bases")
    write_parquet(sequences, target)
    receipt(intermediate / "sequences_provenance.json", [source, acr_path],
            {"coordinates": "zero-based, end-exclusive", "n_acrs": len(sequences)})
    print(f"Wrote {target}")


def prepare_motifs(data_dir: Path, *, overwrite: bool = False) -> None:
    from Bio import motifs

    from .scoring import MotifScorer, motif_to_pwm

    target = data_dir / "maize_acr_motif_scores.parquet"
    refuse_existing([target], overwrite=overwrite)
    seq_path = data_dir / "maize_acr_sequences.parquet"
    source = data_dir / "raw" / "jaspar_plants_2026.txt"
    sequences = pd.read_parquet(seq_path)
    require_unique(sequences.index, "ACR")
    if len(sequences) != 108843:
        raise ValueError("Expected 108,843 reference ACR sequences")
    with source.open() as handle:
        plant_motifs = list(motifs.parse(handle, "jaspar"))
    ids = pd.Index([m.matrix_id for m in plant_motifs])
    require_unique(ids, "motif")
    if len(ids) != 927:
        raise ValueError("Expected 927 JASPAR 2026 CORE Plants motifs")
    pwms = {m.matrix_id: motif_to_pwm(m) for m in plant_motifs}
    scores = MotifScorer(pwms).score_sequences(sequences)
    if not scores.index.equals(sequences.index):
        raise ValueError("Motif/sequence ACR alignment failed")
    write_parquet(scores, target)
    reloaded = pd.read_parquet(target)
    if not reloaded.index.equals(scores.index) or not reloaded.columns.equals(scores.columns):
        raise ValueError("Motif parquet axes changed during roundtrip")
    np.testing.assert_array_equal(reloaded.to_numpy(), scores.to_numpy())
    intermediate = data_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    receipt(intermediate / "motifs_provenance.json", [seq_path, source],
            {"alpha": 0.99, "background": [0.25] * 4, "both_strands": True,
             "score": "log2 mean likelihood ratio over strand/window placements",
             "motif_order": scores.columns.tolist(), "dtype": "float32"})
    print(f"Wrote {target}")



def validate_outputs(data_dir: Path) -> None:
    from .data import load_data

    bundle = load_data(data_dir)
    shapes = {"accessibility": bundle.accessibility.shape,
              "phenotype": bundle.phenotype.shape,
              "ancestry_pcs": bundle.ancestry_pcs.shape,
              "motif_scores": bundle.motif_scores.shape,
              "acr_sequences": bundle.acr_sequences.shape}
    expected = {"accessibility": (169, 108843), "phenotype": (169, 1),
                "ancestry_pcs": (169, 10), "motif_scores": (108843, 927),
                "acr_sequences": (108843, 1)}
    if shapes != expected:
        raise ValueError(f"Unexpected output dimensions: {shapes}")
    inputs = [data_dir / name for name in [*ACCESSIBILITY_OUTPUTS,
              "maize_acr_sequences.parquet", "maize_acr_motif_scores.parquet"]]
    receipt(data_dir / "intermediate" / "validation.json", inputs,
            {"shapes": shapes, "holdout_evaluated": False})
    print("All five parquet files passed alignment, CPM, finite-value, shape, and sample-ID checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--stage", choices=["all", "cells", "cpm", "accessibility", "sequences", "motifs", "validate"],
                        default="all")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing stage outputs")
    parser.add_argument("--reuse-intermediates", action="store_true",
                        help="Validate and reuse cell/count intermediates (used by Make)")
    parser.add_argument("--check", action="store_true", help="Report missing files; do no processing")
    args = parser.parse_args()
    missing = missing_inputs(args.data_dir, args.stage)
    if missing:
        print("Missing inputs:\n" + "\n".join(f"  {p}" for p in missing))
        raise SystemExit(1)
    if args.check:
        load_identities()
        print("Required input files are present. See docs/data_pipeline.md for Python/R dependencies.")
        return
    if args.stage in ("all", "accessibility"):
        prepare_accessibility(args.data_dir, overwrite=args.overwrite,
                              reuse_intermediates=args.reuse_intermediates)
    if args.stage == "cells":
        prepare_cells(args.data_dir, overwrite=args.overwrite, require_labels=False)
    if args.stage == "cpm":
        prepare_cpm(args.data_dir, overwrite=args.overwrite,
                    reuse_cells=args.reuse_intermediates)
    if args.stage in ("all", "sequences"):
        prepare_sequences(args.data_dir, overwrite=args.overwrite)
    if args.stage in ("all", "motifs"):
        prepare_motifs(args.data_dir, overwrite=args.overwrite)
    if args.stage in ("all", "validate"):
        validate_outputs(args.data_dir)
