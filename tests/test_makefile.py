"""Exercise dependency propagation without downloading or processing real data."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_graph(tmp_path):
    make = ROOT / ".venv" / "bin" / "make"
    if not make.exists():
        found = shutil.which("gmake") or shutil.which("make")
        if not found:
            pytest.skip("GNU Make >=4.3 unavailable")
        make = Path(found)
    version = subprocess.run([str(make), "--version"], check=False, capture_output=True, text=True)
    if version.returncode or "GNU Make" not in version.stdout:
        pytest.skip("GNU Make unavailable")
    work = tmp_path / "project"
    work.mkdir()
    shutil.copyfile(ROOT / "Makefile", work / "Makefile")
    # A separate filesystem with past timestamps lets Make simulate rebuilt
    # products as newer than validation, without touching real inputs/code.
    old = time.time() - 1000
    code = ["Makefile", ".Renviron", "requirements-preparation.txt", "pyproject.toml",
            "scripts/prepare_data.py", "scripts/setup_r.R",
            "scripts/preserve_cell_accessibility.R", "scripts/reconstruct_cpm.R"]
    code += [f"src/maize_regulatory/{name}.py" for name in
             ["pipeline", "downloads", "preparation", "sample_identity", "evaluation",
              "models", "spatial", "scoring", "data"]]
    code += ["src/maize_regulatory/resources/analysis_genotypes.txt"]
    for filename in code:
        target = work / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        os.utime(target, (old, old))
    data = tmp_path / "data"
    generated = old + 10
    files = {
        "raw/maize282_blup_phenotypes.csv": old,
        "raw/mdp_genotype.hmp.txt": old,
        "raw/cell_assignments.csv": old,
        "raw/maize_282_scATAC_ACRs_x_barcodes.sparse.rds": old,
        "raw/Zm-B73-REFERENCE-NAM-5.0.fa.gz": old,
        "raw/jaspar_plants_2026.txt": old,
        "intermediate/cells/cell_by_acr.sparse.rds": generated,
        "intermediate/cells/cell_metadata.csv": generated,
        "intermediate/cells/acr_index.csv": generated,
        "intermediate/cells/matrix_summary.csv": generated,
        "intermediate/cells/sources.json": generated,
        "intermediate/cells/genotype_by_acr_counts.rds": generated,
        "intermediate/maize_ACR_genotype_cpm.tsv": generated,
        "intermediate/cpm_sources.json": generated,
        "intermediate/accessibility_provenance.json": generated,
        "intermediate/sequences_provenance.json": generated,
        "intermediate/motifs_provenance.json": generated,
        "maize_accessibility_cpm.parquet": generated,
        "maize_flowering_dts.parquet": generated,
        "maize_ancestry_pcs.parquet": generated,
        "maize_acr_sequences.parquet": generated,
        "maize_acr_motif_scores.parquet": generated,
        "intermediate/validation.json": generated + 10,
    }
    for filename, mtime in files.items():
        target = data / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        os.utime(target, (mtime, mtime))
    r_ready = tmp_path / "r.ready"
    r_ready.touch()
    os.utime(r_ready, (old, old))
    fake = tmp_path / "fake_prepare.py"
    fake.write_text('''import sys
from pathlib import Path
args = sys.argv
data = Path(args[args.index("--data-dir") + 1])
stage = args[args.index("--stage") + 1]
outputs = {
 "accessibility": ["maize_accessibility_cpm.parquet", "maize_flowering_dts.parquet",
                   "maize_ancestry_pcs.parquet", "intermediate/accessibility_provenance.json"],
 "motifs": ["maize_acr_motif_scores.parquet", "intermediate/motifs_provenance.json"],
 "validate": ["intermediate/validation.json"],
}
for filename in outputs[stage]:
    (data / filename).touch()
print("SIMULATED stage:", stage)
''')

    def dry_run(*, execute=False):
        flags = [] if execute else ["-n"]
        result = subprocess.run([str(make), *flags, "data", f"DATA_DIR={data}",
                                 f"R_READY={r_ready}", f"PYTHON={sys.executable} {fake}"],
                                cwd=work, check=True,
                                capture_output=True, text=True)
        return result.stdout

    return data, generated + 20, dry_run


def test_unchanged_make_graph_does_nothing(make_graph):
    _, _, dry_run = make_graph
    assert "--stage" not in dry_run()


def test_phenotype_change_leaves_motif_branch_untouched(make_graph):
    data, changed, dry_run = make_graph
    source = data / "raw/maize282_blup_phenotypes.csv"
    os.utime(source, (changed, changed))
    commands = dry_run(execute=True)
    assert commands.count("--stage accessibility") == 1
    assert "--stage validate" in commands
    assert "--stage motifs" not in commands
    assert "--stage sequences" not in commands
    assert "--stage cells" not in commands


def test_motif_input_change_rebuilds_only_motifs_and_validation(make_graph):
    data, changed, dry_run = make_graph
    source = data / "raw/jaspar_plants_2026.txt"
    os.utime(source, (changed, changed))
    commands = dry_run(execute=True)
    assert commands.count("--stage motifs") == 1
    assert "--stage validate" in commands
    assert "--stage sequences" not in commands
    assert "--stage accessibility" not in commands


def test_missing_group_product_runs_recipe_once(make_graph):
    data, _, dry_run = make_graph
    (data / "maize_accessibility_cpm.parquet").unlink()
    (data / "maize_ancestry_pcs.parquet").unlink()
    commands = dry_run()
    assert commands.count("--stage accessibility") == 1
    assert "--stage motifs" not in commands


def test_missing_provenance_is_rebuilt_with_its_data_product(make_graph):
    data, _, dry_run = make_graph
    (data / "intermediate/motifs_provenance.json").unlink()
    commands = dry_run()
    assert commands.count("--stage motifs") == 1
    assert "--stage validate" in commands


def test_missing_cell_product_is_recovered(make_graph):
    data, _, dry_run = make_graph
    (data / "intermediate/cells/cell_by_acr.sparse.rds").unlink()
    commands = dry_run()
    assert "--stage cells --overwrite" in commands
    assert "--stage cpm" in commands
