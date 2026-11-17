"""Sequence availability audit only: no outcome-based filtering or model selection."""

import gzip
import time

import numpy as np
import pandas as pd

from .variant_pilot import _dosage, read_header, sample_mapping


def context_windows(coordinates, chromosome_length, width=2000):
    """Preserve cores; clip at the existing inter-block gap partitions, pad to width."""
    if (not coordinates.index.is_unique or coordinates.chrom.nunique() != 1
            or np.any(coordinates.end <= coordinates.start)
            or np.any(coordinates.end-coordinates.start > width)):
        raise ValueError('Need unique single-chromosome cores no wider than context')
    result = coordinates.copy()
    lower = np.zeros(len(result), dtype=int)
    upper = np.full(len(result), chromosome_length, dtype=int)
    ordered = np.argsort(result.start.to_numpy(), kind='stable')
    starts, ends, blocks = [result[c].to_numpy() for c in ['start', 'end', 'block']]
    if np.any(starts[ordered][1:] < ends[ordered][:-1]):
        raise ValueError('Overlapping cores')
    for left, right in zip(ordered[:-1], ordered[1:]):
        if blocks[left] != blocks[right]:
            cut = (int(ends[left])+int(starts[right]))//2
            upper[blocks == blocks[left]] = cut
            lower[blocks == blocks[right]] = cut
    desired = (starts+ends)//2-width//2
    result['desired_start'] = desired
    result['context_start'] = np.maximum(desired, lower)
    result['context_end'] = np.minimum(desired+width, upper)
    if np.any(result.context_start > starts) or np.any(result.context_end < ends):
        raise ValueError('Context clipping would truncate a core')
    result['core_offset'] = starts-desired
    result['context_clipped'] = ((result.context_start != desired)
                                 | (result.context_end != desired+width))
    return result


def merged_intervals(starts, ends):
    output = []
    for left, right in sorted(zip(starts, ends)):
        if right <= left:
            raise ValueError('Empty interval')
        if output and left <= output[-1][1]:
            output[-1][1] = max(output[-1][1], int(right))
        else:
            output.append([int(left), int(right)])
    return np.asarray(output, dtype=int)


def collect_context_variants(path, windows, genotypes, reference, progress=print):
    """Read all biallelic SNPs in the union, preserving selected sample order."""
    header, names = read_header(path)
    if '##reference=B73_v5' not in header:
        raise ValueError('Need explicit B73_v5 reference')
    mapping = sample_mapping(names, genotypes)
    if not mapping.matched.all():
        raise ValueError('Unmatched requested genotype')
    selected = mapping.source_sample_column.to_numpy()
    intervals = merged_intervals(windows.context_start, windows.context_end)
    chrom_expected = str(int(windows.chrom.iloc[0]))
    records, values, codes = [], [], {}
    at, previous, total, ignored = 0, -1, 0, 0
    start_time = time.monotonic()
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            total += 1
            fields = line.rstrip('\n').split('\t', 9)
            if len(fields) != 10:
                raise ValueError('VCF record missing sample columns')
            chrom, pos, _, ref, alt, _, _, _, fmt, calls = fields
            position = int(pos)-1
            if chrom.removeprefix('chr') != chrom_expected or position < previous:
                raise ValueError('Unexpected chromosome or unsorted VCF')
            previous = position
            while at < len(intervals) and position >= intervals[at, 1]:
                at += 1
            if at == len(intervals):
                break
            if position < intervals[at, 0]:
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
                ignored += 1
                continue
            if reference[position:position+1].decode() != ref:
                raise ValueError(f'B73 REF disagreement at {position+1}')
            if records and records[-1][0] == position:
                raise ValueError(f'Multiple SNP records at coordinate {position+1}')
            samples = calls.split('\t')
            if len(samples) != len(names):
                raise ValueError('VCF sample count mismatch')
            if fmt.split(':').count('GT') != 1:
                raise ValueError('Missing/duplicate GT field')
            gt_at = fmt.split(':').index('GT')
            row = np.empty(len(selected), dtype=np.int8)
            for g, col in enumerate(selected):
                pieces = samples[col].split(':')
                if gt_at >= len(pieces):
                    raise ValueError('Sample GT field missing')
                call = pieces[gt_at]
                if call not in codes:
                    codes[call] = _dosage(call)
                row[g] = codes[call]
            records.append((position, ref, alt))
            values.append(row)
            if progress and len(records) % 50000 == 0:
                progress(f'Collected {len(records):,} context SNPs; '
                         f'{time.monotonic()-start_time:.1f}s elapsed', flush=True)
    if not records:
        raise ValueError('No context SNP records')
    variants = pd.DataFrame(records, columns=['pos_0based', 'REF', 'ALT'])
    return variants, np.stack(values), mapping, dict(
        seconds=time.monotonic()-start_time, records_scanned=total,
        context_biallelic_snps=len(records), ignored_other_variant_records=ignored,
        all_context_REF_match_B73=True, parsed_GT_codes=codes)


def interval_call_counts(positions, calls, starts, ends):
    """Count missing, heterozygous and alternate calls per interval/genotype."""
    positions, calls = np.asarray(positions), np.asarray(calls)
    if (calls.ndim != 2 or len(calls) != len(positions)
            or np.any(np.diff(positions) <= 0) or not np.isin(calls, [-1, 0, 1, 2]).all()):
        raise ValueError('Variant positions/calls invalid')
    first, last = np.searchsorted(positions, starts), np.searchsorted(positions, ends)
    result = {'snps': last-first}
    for name, value in [('missing', -1), ('heterozygous', 1), ('alternate', 2)]:
        counts = np.empty((len(first), calls.shape[1]), dtype=np.int32)
        for g in range(calls.shape[1]):
            cumulative = np.r_[0, np.cumsum(calls[:, g] == value)]
            counts[:, g] = cumulative[last]-cumulative[first]
        result[name] = counts
    result['complete_homozygous'] = (result['missing'] == 0) & (result['heterozygous'] == 0)
    return result


def reconstruct_patterns(windows, reference, variants, calls, support, kind, width=2000):
    """Cache unique sequences within each ACR, with -1 for unavailable pairs.

    Availability requires complete homozygous listed SNPs and entirely ACGT DNA.
    B73 patterns are retained separately even if no genotype has that sequence.
    """
    positions = variants.pos_0based.to_numpy()
    alt = variants.ALT.to_numpy()
    context = kind == 'context'
    starts = windows.context_start.to_numpy() if context else windows.start.to_numpy()
    ends = windows.context_end.to_numpy() if context else windows.end.to_numpy()
    first, last = np.searchsorted(positions, starts), np.searchsorted(positions, ends)
    pattern_ids = np.full((len(windows), calls.shape[1]), -1, dtype=np.int32)
    reference_ids = np.full(len(windows), -1, dtype=np.int32)
    unambiguous = np.zeros(len(windows), bool)
    records = []
    start_time = time.monotonic()
    for i, acr in enumerate(windows.index):
        left, right = int(starts[i]), int(ends[i])
        raw = reference[left:right].decode()
        offset = left-int(windows.desired_start.iloc[i]) if context else 0
        seq = ('N'*offset+raw+'N'*(width-offset-len(raw))) if context else raw
        if context and len(seq) != width:
            raise ValueError('Context width mismatch')
        unambiguous[i] = set(seq) <= set('ACGT')
        if not unambiguous[i]:
            continue
        local_start = int(windows.core_offset.iloc[i]) if context else 0
        local_end = local_start+int(windows.end.iloc[i]-windows.start.iloc[i])
        seen = {}

        def add(sequence, is_reference):
            if sequence not in seen:
                seen[sequence] = len(records)
                records.append(dict(pattern_id=len(records), ACR=acr, acr_index=i,
                                    sequence=sequence, is_b73=is_reference,
                                    core_start=local_start, core_end=local_end))
            return seen[sequence]

        reference_ids[i] = add(seq, True)
        rows = slice(int(first[i]), int(last[i]))
        relative = positions[rows]-left+offset
        alleles = alt[rows]
        for g in np.flatnonzero(support['complete_homozygous'][i]):
            bases = list(seq)
            changed = np.flatnonzero(calls[rows, g] == 2)
            for v in changed:
                bases[int(relative[v])] = alleles[v]
            pattern_ids[i, g] = add(''.join(bases), False)
    patterns = pd.DataFrame(records)
    if not records:
        raise ValueError('No reconstructable sequence patterns')
    if not np.array_equal(pattern_ids >= 0, support['complete_homozygous'] & unambiguous[:, None]):
        raise AssertionError('Availability/reconstruction mismatch')
    return patterns, pattern_ids, reference_ids, unambiguous, time.monotonic()-start_time


def availability_summary(support, unambiguous, ids, patterns):
    complete = support['complete_homozygous']
    available = ids >= 0
    both = (support['missing'] > 0) & (support['heterozygous'] > 0)
    n = available.sum(axis=1)
    used = np.unique(ids[available])
    return dict(total_pairs=int(complete.size), complete_homozygous_pairs=int(complete.sum()),
                available_pairs=int(available.sum()), available_fraction=float(available.mean()),
                pairs_with_missing_calls=int((support['missing'] > 0).sum()),
                pairs_with_heterozygous_calls=int((support['heterozygous'] > 0).sum()),
                pairs_with_both=int(both.sum()),
                complete_pairs_lost_to_ambiguous_or_padded_DNA=int((complete & ~unambiguous[:, None]).sum()),
                unambiguous_acrs=int(unambiguous.sum()), acrs_with_any_available_pair=int((n > 0).sum()),
                acrs_with_ge_20_genotypes=int((n >= 20).sum()),
                acrs_with_ge_50_genotypes=int((n >= 50).sum()),
                available_genotypes_per_acr_quantiles=np.quantile(n, [0, .25, .5, .75, 1]).tolist(),
                unique_patterns_including_B73=len(patterns),
                unique_personalized_patterns=len(used),
                pair_to_unique_pattern_ratio=float(available.sum()/len(used)) if len(used) else None,
                pairs_with_sequence_different_from_B73=int(
                    (~patterns.is_b73.to_numpy()[ids[available]]).sum()))


def width_sensitivity(windows, variants, calls, reference, widths=(300, 500, 1000, 1500, 2000)):
    """Descriptive alternatives, without adopting a new width or selecting samples."""
    rows = []
    for width in widths:
        if width <= 0 or width > 2000:
            raise ValueError('Sensitivity windows must lie inside audited 2,000-bp windows')
        desired = ((windows.start+windows.end)//2-width//2).to_numpy()
        starts = np.maximum(desired, windows.context_start)
        ends = np.minimum(desired+width, windows.context_end)
        fits = (windows.end-windows.start <= width).to_numpy()
        support = interval_call_counts(variants.pos_0based, calls, starts, ends)
        unambiguous = np.array([
            fits[i] and int(starts.iloc[i]) == desired[i]
            and int(ends.iloc[i]) == desired[i]+width
            and set(reference[int(starts.iloc[i]):int(ends.iloc[i])]) <= set(b'ACGT')
            for i in range(len(windows))])
        available = support['complete_homozygous'] & unambiguous[:, None]
        n = available.sum(axis=1)
        rows.append(dict(width_bp=width, available_pairs=int(available.sum()),
            fraction_of_original_pairs=float(available.mean()),
            acrs_longer_than_window=int((~fits).sum()),
            acrs_with_any_pair=int((n > 0).sum()), acrs_with_ge_20_genotypes=int((n >= 20).sum()),
            acrs_with_ge_50_genotypes=int((n >= 50).sum()),
            median_available_genotypes_per_acr=float(np.median(n))))
    return pd.DataFrame(rows)
