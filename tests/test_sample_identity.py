import gzip
import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from maize_regulatory.pipeline import refuse_existing, write_parquet
from maize_regulatory.sample_identity import (
    GEO_MEMBER,
    freeze_identities,
    genotype_header_from_archive,
    load_identities,
    validate_analysis_order,
)


def test_permanent_holdout_matches_original_order_and_sklearn_split():
    published, analysis, holdout, development_idx, holdout_idx = load_identities()
    assert len(published) == 172
    assert len(analysis) == 169
    assert len(holdout) == 30
    expected_dev, expected_test = train_test_split(np.arange(169), test_size=30, random_state=42)
    np.testing.assert_array_equal(development_idx, expected_dev)
    np.testing.assert_array_equal(holdout_idx, expected_test)
    assert holdout.equals(analysis[expected_test])
    validate_analysis_order(analysis)
    with pytest.raises(ValueError, match="permanent"):
        validate_analysis_order(analysis[::-1])


def test_freezing_reads_only_identity_and_refuses_to_redefine_split(tmp_path):
    header = ("#Chr\tstart\tend\tTargetID\t"
              + "\t".join(f"G282set_G{i}" for i in range(172)) + "\n").encode()
    names = pd.Series([f"G{i}" for i in range(169)])
    freeze_identities(header, names, tmp_path, "synthetic test")
    _, analysis, _, _, _ = load_identities(tmp_path)
    assert analysis.tolist() == names.tolist()
    freeze_identities(header, names, tmp_path, "synthetic test")
    changed = header.replace(b"G282set_G0\tG282set_G1", b"G282set_G1\tG282set_G0")
    with pytest.raises(ValueError, match="Refusing to change"):
        freeze_identities(changed, names, tmp_path, "synthetic test")
    target = tmp_path / "holdout_genotypes.txt"
    target.write_text("wrong\n" + target.read_text())
    with pytest.raises(ValueError, match="checksum"):
        load_identities(tmp_path)


def test_header_extraction_does_not_need_accessibility_rows():
    header = b"#Chr\tstart\tend\tTargetID\tG282set_example\n"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        member = tarfile.TarInfo(GEO_MEMBER)
        contents = gzip.compress(header + b"numeric contents not retained\n")
        member.size = len(contents)
        bundle.addfile(member, io.BytesIO(contents))
    archive.seek(0)
    assert genotype_header_from_archive(archive) == header


def test_overwrite_flag_allows_derived_outputs(tmp_path):
    target = tmp_path / "result.parquet"
    target.touch()
    with pytest.raises(FileExistsError, match="--overwrite"):
        refuse_existing([target])
    refuse_existing([target], overwrite=True)


def test_failed_parquet_overwrite_keeps_previous_output(tmp_path, monkeypatch):
    target = tmp_path / "result.parquet"
    original = pd.DataFrame({"a": [1]})
    write_parquet(original, target)

    def fail(frame, path):
        path.write_bytes(b"incomplete serialization")
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", fail)
        with pytest.raises(OSError):
            write_parquet(pd.DataFrame({"a": [2]}), target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), original)
