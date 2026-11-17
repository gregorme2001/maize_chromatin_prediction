import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def sparse_cells(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript unavailable")
    code = 'quit(status=if (requireNamespace("Matrix", quietly=TRUE)) 0 else 1)'
    check = subprocess.run([rscript, "-e", code], check=False, capture_output=True)
    if check.returncode:
        pytest.skip("R Matrix unavailable")
    source = tmp_path / "cells.rds"
    create = (
        'args <- commandArgs(TRUE); library(Matrix); '
        'x <- Matrix(matrix(c(1,0,1,0,1,1,1,0,0), nrow=3), sparse=TRUE); '
        'rownames(x) <- c("acr2", "acr1", "acr3"); '
        'colnames(x) <- c("c2", "c1", "c3"); saveRDS(x, args[1])'
    )
    subprocess.run([rscript, "-e", create, str(source)], check=True, capture_output=True)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"cellID": ["c2", "c1", "c3"], "genotype": ["G2", "G1", "G2"]}).to_csv(
        metadata, index=False
    )
    directory = tmp_path / "intermediate"
    subprocess.run([rscript, str(SCRIPTS / "preserve_cell_accessibility.R"), str(source),
                    str(metadata), str(directory)], check=True, capture_output=True)
    return rscript, source, metadata, directory


def test_sparse_cell_matrix_preserves_values_and_exact_mapping(sparse_cells):
    rscript, source, metadata, directory = sparse_cells
    check = (
        'args <- commandArgs(TRUE); library(Matrix); '
        'x <- readRDS(args[1]); z <- readRDS(args[2]); '
        'stopifnot(inherits(z, "sparseMatrix"), identical(z, t(x)))'
    )
    subprocess.run([rscript, "-e", check, str(source), str(directory / "cell_by_acr.sparse.rds")],
                   check=True, capture_output=True)
    saved = pd.read_csv(directory / "cell_metadata.csv")
    pd.testing.assert_frame_equal(saved[["cellID", "genotype"]], pd.read_csv(metadata))
    assert saved["matrix_row"].tolist() == [1, 2, 3]
    assert pd.read_csv(directory / "acr_index.csv")["ACR"].tolist() == ["acr2", "acr1", "acr3"]
    assert pd.read_csv(directory / "matrix_summary.csv").iloc[0]["genotype_labels_available"]
    pd.read_csv(metadata).iloc[::-1].to_csv(metadata, index=False)
    failed = subprocess.run([rscript, str(SCRIPTS / "preserve_cell_accessibility.R"), str(source),
                             str(metadata), str(directory)], check=False, capture_output=True)
    assert failed.returncode != 0


def test_preservation_without_labels_is_explicit(sparse_cells):
    rscript, source, _, directory = sparse_cells
    subprocess.run([rscript, str(SCRIPTS / "preserve_cell_accessibility.R"), str(source),
                    "-", str(directory)], check=True, capture_output=True)
    saved = pd.read_csv(directory / "cell_metadata.csv")
    assert "genotype" not in saved.columns
    assert saved["cellID"].tolist() == ["c2", "c1", "c3"]
    assert not pd.read_csv(directory / "matrix_summary.csv").iloc[0]["genotype_labels_available"]


def test_r_counts_to_cpm_preserves_acr_and_cell_identity(sparse_cells, tmp_path):
    rscript, _, _, directory = sparse_cells
    code = 'quit(status=if (requireNamespace("edgeR", quietly=TRUE)) 0 else 1)'
    check = subprocess.run([rscript, "-e", code], check=False, capture_output=True)
    if check.returncode:
        pytest.skip("R edgeR unavailable; install to test counts-to-CPM reconstruction")
    target = tmp_path / "cpm.tsv"
    subprocess.run([rscript, str(SCRIPTS / "reconstruct_cpm.R"), str(directory), str(target)],
                   check=True, capture_output=True)
    actual = pd.read_csv(target, sep="\t", index_col=0)
    assert actual.index.tolist() == ["acr2", "acr1", "acr3"]
    assert actual.columns.tolist() == ["G1", "G2"]
    np.testing.assert_allclose(actual, np.array([[0, 2/3], [1/2, 0], [1/2, 1/3]]) * 1e6)
    verify = (
        'args <- commandArgs(TRUE); library(Matrix); x <- readRDS(args[1]); '
        'stopifnot(identical(rownames(x), c("G1", "G2")), '
        'identical(colnames(x), c("acr2", "acr1", "acr3")), '
        'all(as.matrix(x) == matrix(c(0,2,1,0,1,1), nrow=2)))'
    )
    subprocess.run([rscript, "-e", verify, str(directory / "genotype_by_acr_counts.rds")],
                   check=True, capture_output=True)
