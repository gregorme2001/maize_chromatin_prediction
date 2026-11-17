import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from maize_regulatory import downloads
from maize_regulatory.pipeline import (
    extract_sequences,
    metadata_source,
    missing_inputs,
    phenotype_source,
)
from maize_regulatory.preparation import align_cpm, ancestry_pcs, match_phenotypes


def test_matching_preserves_published_order_and_legacy_ambiguity_exclusions():
    pheno = pd.DataFrame({
        "Family_Inbred_Name": ["G-2", "B73", "B73", "g1", "other"],
        "DTS": [60, 70, 71, 62, 80],
    })
    y, excluded = match_phenotypes(pd.Index(["G1", "B73", "G2", "absent"]), pheno)
    assert y.index.tolist() == ["G1", "G2"]
    assert y["DTS"].tolist() == [62, 60]
    assert excluded == ["B73", "ABSENT"]
    with pytest.raises(ValueError, match="duplicate"):
        match_phenotypes(pd.Index(["G1", "G-1"]), pheno)
    pheno.loc[3, "DTS"] = np.nan
    with pytest.raises(ValueError, match="DTS"):
        match_phenotypes(pd.Index(["G1", "G2"]), pheno)


def test_snp_parser_matches_known_dosages_including_missing_and_rejected_single_letters():
    # SNPs: variable, imputed, entirely missing, constant. Single-letter calls
    # are missing under the legacy rule, even when biologically interpretable.
    frame = pd.DataFrame({f"meta{i}": ["x"] * 4 for i in range(11)})
    frame = frame.rename(columns={"meta0": "alleles"})
    frame["alleles"] = ["A/C"] * 4
    frame["g1"] = ["AA", "AA", "N", "AA"]
    frame["g2"] = ["AC", "A", "N", "AA"]
    frame["g3"] = ["CC", "CC", "N", "AA"]
    frame["g4"] = ["AA", "AC", "N", "AA"]
    G = np.array([[0, 0], [1, 1], [2, 2], [0, 1]], dtype=float)
    expected = PCA(n_components=2, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(G)
    )
    actual = ancestry_pcs(frame, n_components=2)
    assert actual.index.tolist() == ["G1", "G2", "G3", "G4"]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_cpm_alignment_preserves_all_acrs_and_explicit_requested_genotypes():
    cpm = pd.DataFrame({
        "282set_G2Goodman-Buckler": [300000., 700000.],
        "282set_G1": [400000., 600000.],
        "282set_extra": [200000., 800000.],
    }, index=["chr2_0_10", "chr1_20_30"])
    order = pd.Index(["G1", "G2"], name="genotype")
    X = align_cpm(cpm, order)
    assert X.index.equals(order)
    assert X.columns.equals(cpm.index)
    np.testing.assert_array_equal(X, [[400000, 600000], [300000, 700000]])
    with pytest.raises(ValueError, match="Missing required"):
        align_cpm(cpm, pd.Index(["missing"]))


def test_reference_extraction_uses_half_open_coordinates_and_input_order():
    ids = pd.Index(["chr2_1_4", "chr1_0_3"], name="ACR")
    sequences = extract_sequences(ids, {"chr1": "ACGTACGT", "chr2": "TTAGC"})
    assert sequences.index.equals(ids)
    assert sequences["sequence"].tolist() == ["TAG", "ACG"]
    with pytest.raises(ValueError, match="Invalid reference"):
        extract_sequences(pd.Index(["chr1_1_20"]), {"chr1": "ACGT"})
    with pytest.raises(ValueError, match="non-ACGT"):
        extract_sequences(pd.Index(["chr1_0_3"]), {"chr1": "ANN"})


def test_workbook_conversions_keep_rows_and_duplicates(tmp_path):
    pytest.importorskip("openpyxl")
    raw, intermediate = tmp_path / "raw", tmp_path / "intermediate"
    raw.mkdir()
    intermediate.mkdir()
    metadata = pd.DataFrame({"cellID": ["c2", "c1"], "genotype": ["G2", "G1"],
                             "celltype_full": ["type", "type"], "library": ["L2", "L1"]})
    metadata.to_excel(raw / "Dryad_Data_S2.xlsx", sheet_name="Table_S5", index=False)
    saved_metadata = pd.read_csv(metadata_source(raw, intermediate))
    pd.testing.assert_frame_equal(saved_metadata[metadata.columns], metadata)
    assert saved_metadata["normalized_genotype"].tolist() == ["G2", "G1"]
    phenotypes = pd.DataFrame({"Panel": ["ASSO"] * 282 + ["other"],
                               "Family_Inbred_Name": ["B73", "B73"] + [f"G{i}" for i in range(281)],
                               "DTS": np.arange(283, dtype=float) + .5})
    phenotypes.to_excel(raw / "Peiffer2014Genetics_blupPhenos20150325.xlsx", index=False)
    saved = pd.read_csv(phenotype_source(raw, intermediate))
    pd.testing.assert_frame_equal(saved, phenotypes.iloc[:282])


def test_preflight_reports_manual_inputs_and_accepts_csv_alternatives(tmp_path):
    missing = missing_inputs(tmp_path, "all")
    assert any("mdp_genotype.hmp.txt" in p for p in missing)
    assert any("cell_assignments.csv OR" in p for p in missing)
    assert not (tmp_path / "raw").exists()  # Read-only preflight.


def test_downloader_keeps_existing_and_rejects_html(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        pass

    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return Response(b"<html>login</html>")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="HTML"):
        downloads.download("motifs", tmp_path)
    target = tmp_path / downloads.SOURCES["motifs"][0]
    assert not target.exists()
    monkeypatch.setattr(downloads.urllib.request, "urlopen", lambda *a, **k: Response(b">MA motif"))
    downloads.download("motifs", tmp_path)
    record = json.loads(Path(str(target) + ".download.json").read_text())
    assert record["sha256"] == downloads.sha256(target)
    downloads.download("motifs", tmp_path)
    assert target.read_bytes() == b">MA motif"


def test_download_overwrite_replaces_successfully_and_keeps_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / downloads.SOURCES["motifs"][0]
    target.write_bytes(b"old")
    monkeypatch.setattr(downloads.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"new"))
    downloads.download("motifs", tmp_path, overwrite=True)
    assert target.read_bytes() == b"new"
    monkeypatch.setattr(downloads.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"<html>error"))
    with pytest.raises(ValueError):
        downloads.download("motifs", tmp_path, overwrite=True)
    assert target.read_bytes() == b"new"


def test_default_download_uses_supplied_cell_assignments(tmp_path, monkeypatch):
    (tmp_path / "cell_assignments.csv").touch()
    called = []
    monkeypatch.setattr(downloads, "download", lambda key, raw_dir, **kwargs: called.append(key))
    monkeypatch.setattr(sys, "argv", ["download_inputs.py", "--raw-dir", str(tmp_path)])
    downloads.main()
    assert called == ["cells", "genome", "motifs"]
    called.clear()
    monkeypatch.setattr(sys, "argv", ["download_inputs.py", "--raw-dir", str(tmp_path),
                                      "--only", "metadata"])
    downloads.main()
    assert called == ["metadata"]
