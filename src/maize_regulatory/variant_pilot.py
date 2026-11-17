"""Sequence-only audit of one Panzea-v5 chromosome, without association testing."""

import gzip
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preparation import normalize_reconstructed_genotype
from .spatial import parse_acr_coordinates


@dataclass
class VariantPilot:
    header: list[str]
    sample_map: pd.DataFrame
    variants: pd.DataFrame
    dosage: pd.DataFrame
    acrs: pd.DataFrame
    genotype_qc: pd.DataFrame
    summary: dict


def read_header(path):
    """Read source metadata and sample identities, without interpreting any calls."""
    metadata = []
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('##'):
                metadata.append(line.rstrip('\n'))
            elif line.startswith('#CHROM\t'):
                fields = line.rstrip('\n').split('\t')
                if fields[:9] != ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                                  'INFO', 'FORMAT']:
                    raise ValueError('Unexpected VCF fixed columns')
                names = pd.Index(fields[9:], name='source_sample')
                if not names.is_unique:
                    raise ValueError('Duplicate raw VCF sample identities')
                return metadata, names
            else:
                raise ValueError('VCF data encountered before #CHROM header')
    raise ValueError('VCF sample header missing')


def sample_mapping(source_names, genotypes):
    """Explicit source-column map; absent genotypes remain present as missing calls."""
    if not genotypes.is_unique:
        raise ValueError('Duplicate requested genotype identities')
    normalized = pd.Index([normalize_reconstructed_genotype(n) for n in source_names])
    records = []
    for genotype in genotypes:
        positions = np.flatnonzero(normalized == genotype)
        if len(positions) > 1:
            raise ValueError(f'Ambiguous source sample mapping for {genotype}')
        source_position = int(positions[0]) if len(positions) else -1
        records.append({'genotype': genotype, 'source_sample_column': source_position,
                        'source_sample': str(source_names[source_position])
                        if source_position >= 0 else '',
                        'matched': source_position >= 0})
    result = pd.DataFrame(records).set_index('genotype')
    if not result.index.equals(genotypes):
        raise ValueError('Genotype mapping changed requested identity/order')
    return result


def _dosage(gt):
    """Diploid ALT allele count; partial/missing calls are missing, never reference.

    Phasing is not used for this audit. Unsupported/non-diploid calls raise rather
    than inventing a haplotype or converting an unexpected allele silently.
    """
    if gt == '.':
        return -1
    parts = re.split(r'[/|]', gt)
    if len(parts) != 2 or any(a not in {'0', '1', '.'} for a in parts):
        raise ValueError(f'Unsupported GT at a biallelic SNP: {gt!r}')
    if '.' in parts:
        return -1
    return int(parts[0]) + int(parts[1])


def _number(value):
    return np.nan if value is None or value == '.' else float(value)


def audit_chromosome(path, sequences, genotypes, *, chromosome=10, progress=None):
    """Stream chromosome VCF; intersect nonoverlapping ACRs and audit selected calls.

    REF is checked against stored B73-v5 ACR sequences. VCF POS is 1-based;
    ACR coordinates are 0-based, half-open. No motif/accessibility/DTS is used.
    All requested genotype columns are retained in original order. Only selected
    genotypes' GT fields are interpreted; other source samples are not summarized.
    """
    if not sequences.index.is_unique or 'sequence' not in sequences:
        raise ValueError('Need unique ACR identities and sequence column')
    header, names = read_header(path)
    if '##reference=B73_v5' not in header:
        raise ValueError('This pilot requires an explicit B73_v5 VCF reference declaration')
    mapping = sample_mapping(names, genotypes)
    coordinates = parse_acr_coordinates(sequences.index)
    coordinates = coordinates.loc[coordinates.chrom.eq(chromosome)].copy()
    if coordinates.empty:
        raise ValueError('No ACRs on the requested chromosome')
    selected_sequences = sequences.loc[coordinates.index, 'sequence']
    lengths = coordinates.end-coordinates.start
    if not np.array_equal(selected_sequences.str.len(), lengths):
        raise ValueError('ACR coordinate/sequence lengths disagree')
    order = np.argsort(coordinates.start.to_numpy(), kind='stable')
    starts = coordinates.start.to_numpy()[order]
    ends = coordinates.end.to_numpy()[order]
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError('This bounded pilot requires nonoverlapping ACRs')
    acr_ids = coordinates.index.to_numpy()[order]
    seqs = selected_sequences.to_numpy()[order]
    matched_at = np.flatnonzero(mapping.matched.to_numpy())
    source_columns = mapping.source_sample_column.to_numpy()[matched_at]
    records, dosages = [], []
    formats = Counter()
    gt_counts = Counter()
    counts = Counter()
    previous_position = -1
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            counts['vcf_records'] += 1
            fields = line.rstrip('\n').split('\t', 9)
            if len(fields) != 10:
                raise ValueError('VCF record missing genotype columns')
            chrom, pos, _, ref, alt, qual, filt, info, fmt, sample_text = fields
            if chrom.removeprefix('chr') != str(chromosome):
                raise ValueError('Unexpected chromosome in the single-chromosome input')
            position = int(pos)-1
            if position < previous_position:
                raise ValueError('VCF positions are not sorted')
            previous_position = position
            formats[fmt] += 1
            if len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
                counts['other_than_biallelic_snp'] += 1
                continue
            counts['biallelic_snp_records'] += 1
            at = int(np.searchsorted(starts, position, side='right'))-1
            if at < 0 or position >= ends[at]:
                continue
            counts['acr_overlapping_biallelic_snp_records'] += 1
            source_calls = sample_text.split('\t')
            if len(source_calls) != len(names):
                raise ValueError('VCF record sample count differs from sample header')
            format_fields = fmt.split(':')
            if 'GT' not in format_fields:
                raise ValueError('Overlapping SNP has no genotype field')
            gt_position = format_fields.index('GT')
            values = np.full(len(genotypes), -1, dtype=np.int8)
            for destination, source in zip(matched_at, source_columns):
                call_fields = source_calls[source].split(':')
                if gt_position >= len(call_fields):
                    raise ValueError('Genotype value has fewer fields than FORMAT declares')
                call = call_fields[gt_position]
                values[destination] = _dosage(call)
                gt_counts[call] += 1
            n_ref, n_het, n_alt = [int(np.count_nonzero(values == code)) for code in [0, 1, 2]]
            n_called = n_ref+n_het+n_alt
            n_classes = sum(n > 0 for n in [n_ref, n_het, n_alt])
            reference_base = seqs[at][position-starts[at]]
            annotation = dict(piece.split('=', 1) for piece in info.split(';') if '=' in piece)
            record = {
                'variant_id': f'chr{chromosome}:{position+1}:{ref}:{alt}', 'ACR': acr_ids[at],
                'chrom': chromosome, 'pos_1based': position+1, 'pos_0based': position,
                'REF': ref, 'ALT': alt, 'b73_acr_base': reference_base,
                'ref_matches_b73': ref == reference_base, 'QUAL': qual, 'FILTER': filt,
                'FORMAT': fmt, 'MQ': _number(annotation.get('MQ')),
                'source_CVP': _number(annotation.get('CVP')),
                'n_hom_ref': n_ref, 'n_het': n_het, 'n_hom_alt': n_alt,
                'n_called': n_called, 'n_missing': len(genotypes)-n_called,
                'development_call_rate': n_called/len(genotypes),
                'variable_across_development': n_classes > 1,
                'both_homozygotes_observed': n_ref > 0 and n_alt > 0,
                'at_least_five_each_homozygote': n_ref >= 5 and n_alt >= 5,
            }
            records.append(record)
            dosages.append(values)
            if progress and len(records) % 20000 == 0:
                progress(f'{counts["vcf_records"]:,} source records scanned; '
                         f'{len(records):,} ACR-overlapping SNP records')
    if not records:
        raise ValueError('No biallelic SNPs overlap ACRs; investigate source/build before proceeding')
    variants = pd.DataFrame(records).set_index('variant_id')
    if not variants.index.is_unique:
        raise ValueError('Duplicate overlapping chromosome/position/REF/ALT records')
    dosage = pd.DataFrame(np.stack(dosages), index=variants.index, columns=genotypes)
    if not dosage.columns.equals(genotypes) or not dosage.index.equals(variants.index):
        raise ValueError('Variant/genotype identity or order mismatch')
    acrs = coordinates.copy()
    flags = {
        'all_overlapping_snps': np.ones(len(variants), dtype=bool),
        'reference_matching_snps': variants.ref_matches_b73,
        'reference_matching_variable_snps': variants.ref_matches_b73 & variants.variable_across_development,
        'reference_matching_snps_with_five_each_homozygote': variants.ref_matches_b73
        & variants.at_least_five_each_homozygote,
    }
    for field, flag in flags.items():
        totals = variants.loc[flag].groupby('ACR').size()
        acrs[field] = totals.reindex(acrs.index, fill_value=0).astype(int)
    good = variants.ref_matches_b73.to_numpy()
    call_values = dosage.to_numpy()[good]
    qc = mapping.copy()
    qc['n_reference_matching_acr_snps'] = int(good.sum())
    qc['n_called'] = (call_values >= 0).sum(axis=0)
    qc['call_rate'] = qc.n_called/good.sum() if good.any() else np.nan
    qc['n_heterozygous'] = (call_values == 1).sum(axis=0)
    qc['heterozygous_fraction_of_called'] = np.divide(
        qc.n_heterozygous, qc.n_called, out=np.full(len(qc), np.nan), where=qc.n_called > 0)
    observed_fields = sorted({field for fmt in formats for field in fmt.split(':')})
    summary = {
        'source_filename': Path(path).name, 'chromosome': chromosome,
        'reference_header': 'B73_v5', 'source_sample_count': len(names),
        'requested_development_genotypes': len(genotypes),
        'matched_development_genotypes': int(mapping.matched.sum()),
        'unmatched_development_genotypes': mapping.index[~mapping.matched].tolist(),
        'n_acrs': len(acrs), **dict(counts),
        'format_record_counts': dict(formats), 'selected_genotype_GT_counts': dict(gt_counts),
        'declared_FORMAT_fields': re.findall(r'^##FORMAT=<ID=([^,>]+)',
                                            '\n'.join(header), re.MULTILINE),
        'observed_FORMAT_fields': observed_fields,
        'per_genotype_depth_field_available': 'DP' in observed_fields,
        'per_genotype_quality_field_available': 'GQ' in observed_fields,
        'acr_snp_records_with_mapping_quality': int(variants.MQ.notna().sum()),
        'acr_snp_records_with_site_QUAL': int(variants.QUAL.ne('.').sum()),
        'acr_snp_FILTER_counts': variants.FILTER.value_counts().to_dict(),
        'reference_matching_acr_snp_records': int(variants.ref_matches_b73.sum()),
        'reference_mismatching_acr_snp_records': int((~variants.ref_matches_b73).sum()),
        'reference_matching_variable_snp_records': int(flags['reference_matching_variable_snps'].sum()),
        'n_acrs_with_any_snp': int(acrs.all_overlapping_snps.gt(0).sum()),
        'n_acrs_with_reference_matching_variable_snp': int(acrs.reference_matching_variable_snps.gt(0).sum()),
        'n_acrs_with_reference_matching_snp_five_each_homozygote': int(
            acrs.reference_matching_snps_with_five_each_homozygote.gt(0).sum()),
        'median_development_call_rate_at_reference_matching_acr_snps': float(
            variants.loc[variants.ref_matches_b73, 'development_call_rate'].median()),
        'development_genotype_call_rate_range': [float(qc.call_rate.min()),
                                                 float(qc.call_rate.max())],
        'n_reference_matching_variable_snps_call_rate_ge_90pct': int((
            flags['reference_matching_variable_snps'] & variants.development_call_rate.ge(.9)).sum()),
        'n_acrs_variable_snp_call_rate_ge_90pct': int(variants.loc[
            flags['reference_matching_variable_snps'] & variants.development_call_rate.ge(.9), 'ACR'].nunique()),
        'illustrative_thresholds': '90% called and >=5 per homozygote are descriptive counts, not approved QC filters',
        'dosage_encoding': '-1 missing/partial missing; 0 homozygous REF; 1 heterozygous; 2 homozygous ALT; phasing not used',
        'phenotype_loaded': False, 'accessibility_loaded': False,
        'holdout_calls_analyzed': False, 'association_testing': False,
        'analysis_cohort_changed': False,
    }
    return VariantPilot(header, mapping, variants, dosage, acrs, qc, summary)
