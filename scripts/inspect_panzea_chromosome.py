"""Audit the downloaded Panzea-v5 chromosome 10 against our ACRs; no association tests."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from maize_regulatory.cohort import cohort_metadata, development_genotypes
from maize_regulatory.variant_pilot import audit_chromosome


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcf', type=Path,
                        default=Path('data/raw/panzea_v5/panzea_hapmap_282_chr10.vcf.gz'))
    parser.add_argument('--sequences', type=Path, default=Path('data/maize_acr_sequences.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/panzea_chr10_pilot'))
    parser.add_argument('--intermediate-dir', type=Path,
                        default=Path('data/intermediate/panzea_v5_pilot'))
    args = parser.parse_args()
    sequences = pd.read_parquet(args.sequences)
    print('Auditing chromosome 10 with eligible development genotype calls only; no DTS', flush=True)
    result = audit_chromosome(args.vcf, sequences, development_genotypes(),
                              progress=lambda s: print(s, flush=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [('variants', result.variants), ('alt_dosage', result.dosage)]:
        frame.to_parquet(args.intermediate_dir/f'chr10_{name}.parquet')
    for name, frame in [('sample_mapping', result.sample_map), ('acr_coverage', result.acrs),
                         ('development_genotype_qc', result.genotype_qc)]:
        frame.to_csv(args.output_dir/f'{name}.csv')
    result.variants.loc[~result.variants.ref_matches_b73].to_csv(
        args.output_dir/'reference_mismatches.csv')
    (args.output_dir/'vcf_metadata.txt').write_text('\n'.join(result.header)+'\n')
    summary = result.summary
    summary['cohort'] = cohort_metadata()
    summary['created_utc'] = datetime.now(UTC).isoformat()
    summary['source_file_page'] = (
        'https://iastate.box.com/s/9edkm5f51q88ltzrb0hxaz4p36fdsr7q/file/2130575142039')
    summary['input_sha256'] = {str(p): sha256(p) for p in [args.vcf, args.sequences]}
    summary['code_sha256'] = {str(p): sha256(p) for p in [
        Path(__file__), Path('src/maize_regulatory/variant_pilot.py')]}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
