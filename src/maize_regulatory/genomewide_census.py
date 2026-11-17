"""Outcome-independent census of homozygous SNP-substituted sequence support."""

import numpy as np
import pandas as pd

from .personalized_context_audit import interval_call_counts


def sequence_pattern_statistics(positions, calls, starts, ends, available):
    """Exact sequence classes without constructing DNA; IDs are LOCAL to each ACR.

    Unique SNP positions with REF != ALT make homozygous dosage vectors equivalent
    to substituted DNA identity within a fixed interval. Calls outside available
    pairs are never used to construct a sequence class. No-SNP intervals have one
    reference class if any genotype is available, otherwise zero classes.
    """
    positions, calls, available = map(np.asarray, (positions, calls, available))
    if (available.shape != (len(starts), calls.shape[1])
            or len(positions) != len(calls) or np.any(np.diff(positions) <= 0)):
        raise ValueError('Sequence pattern axes/positions disagree')
    first, last = np.searchsorted(positions, starts), np.searchsorted(positions, ends)
    ids = np.full(available.shape, -1, dtype=np.int16)
    records = []
    for i, (left, right) in enumerate(zip(first, last)):
        selected = np.flatnonzero(available[i])
        n, reference_n, classes = len(selected), 0, 0
        largest, second, alt_pairs = 0, 0, 0
        if n:
            panel = calls[left:right, selected].T
            if not np.isin(panel, [0, 2]).all():
                raise ValueError('Available sequence includes missing/heterozygous calls')
            if right > left:
                unique, inverse, counts = np.unique(panel, axis=0, return_inverse=True, return_counts=True)
                reference_n = int(counts[np.all(unique == 0, axis=1)].sum())
            else:
                inverse, counts = np.zeros(n, int), np.array([n])
                reference_n = n
            ids[i, selected] = inverse
            classes = len(counts)
            sorted_counts = np.sort(counts)
            largest = int(sorted_counts[-1])
            second = int(sorted_counts[-2]) if classes > 1 else 0
            alt_pairs = n-reference_n
        records.append(dict(n_available=n, n_sequences=classes,
                            largest_sequence_n=largest, second_sequence_n=second,
                            B73_sequence_n=reference_n, non_B73_pairs=alt_pairs))
    return pd.DataFrame(records), ids


def census_from_variants(windows, variants, calls, reference):
    """Core/context support and diversity, preserving every ACR and genotype axis."""
    n, ng = len(windows), calls.shape[1]
    positions = variants.pos_0based.to_numpy()
    starts = np.r_[windows.start, windows.context_start]
    ends = np.r_[windows.end, windows.context_end]
    all_counts = interval_call_counts(positions, calls, starts, ends)
    per_acr = windows.copy()
    arrays, summaries, per_genotype = {}, {}, pd.DataFrame(index=np.arange(ng))
    availability = {}
    for k, kind in enumerate(['core', 'context']):
        support = {key: value[k*n:(k+1)*n] for key, value in all_counts.items()}
        left, right = starts[k*n:(k+1)*n], ends[k*n:(k+1)*n]
        unambiguous = np.array([set(reference[int(a):int(b)]) <= set(b'ACGT')
                                for a, b in zip(left, right)])
        if kind == 'context':
            unambiguous &= ~windows.context_clipped.to_numpy()
        valid = support['complete_homozygous'] & unambiguous[:, None]
        availability[kind] = valid
        stats, pattern_ids = sequence_pattern_statistics(positions, calls, left, right, valid)
        stats.index = windows.index
        for column in stats:
            per_acr[f'{kind}_{column}'] = stats[column]
        per_acr[f'{kind}_n_SNPs'] = support['snps']
        per_acr[f'{kind}_unambiguous_DNA'] = unambiguous
        arrays[f'{kind}_available'] = valid
        arrays[f'{kind}_pattern_id_within_ACR'] = pattern_ids
        for field in ['missing', 'heterozygous', 'alternate']:
            if support[field].max(initial=0) > np.iinfo(np.uint16).max:
                raise ValueError('Call count exceeds uint16 storage')
            arrays[f'{kind}_{field}_SNP_count'] = support[field].astype(np.uint16)
        missing, het = support['missing'] > 0, support['heterozygous'] > 0
        summaries[kind] = dict(total_pairs=n*ng, available_pairs=int(valid.sum()),
            complete_homozygous_pairs=int(support['complete_homozygous'].sum()),
            pairs_with_missing_calls=int(missing.sum()), pairs_with_heterozygous_calls=int(het.sum()),
            pairs_with_both=int((missing & het).sum()),
            pairs_lost_to_ambiguous_or_padded_DNA=int((support['complete_homozygous'] & ~unambiguous[:, None]).sum()),
            acrs_with_any_pair=int((stats.n_available > 0).sum()),
            acrs_with_two_sequences=int((stats.n_sequences >= 2).sum()),
            distinct_available_sequences=int(stats.n_sequences.sum()),
            non_B73_pairs=int(stats.non_B73_pairs.sum()))
        per_genotype[f'{kind}_available_ACRs'] = valid.sum(axis=0)
        per_genotype[f'{kind}_missing_call_ACRs'] = missing.sum(axis=0)
        per_genotype[f'{kind}_heterozygous_call_ACRs'] = het.sum(axis=0)
    common = availability['core'] & availability['context']
    if not np.array_equal(common, availability['context']):
        raise AssertionError('Available context should imply available core')
    common_stats, common_ids = sequence_pattern_statistics(
        positions, calls, windows.start, windows.end, common)
    common_stats.index = windows.index
    for column in common_stats:
        per_acr[f'common_core_{column}'] = common_stats[column]
    arrays['common_available'] = common
    arrays['common_core_pattern_id_within_ACR'] = common_ids
    summaries['common_pairs'] = int(common.sum())
    return per_acr, per_genotype, arrays, summaries


def coverage_diversity_grid(frame):
    """Descriptive screens only; no threshold is adopted and no cohort is sampled."""
    rows = []
    for scope, coverage in [('core', 'core'), ('context', 'context'), ('common_core', 'context')]:
        for n_genotypes in [1, 20, 50, 80]:
            for second_sequence_n in [0, 5, 10]:
                mask = ((frame[f'{coverage}_n_available'] >= n_genotypes)
                        & (frame[f'{scope}_second_sequence_n'] >= second_sequence_n))
                selected = frame.loc[mask]
                rows.append(dict(sequence_scope=scope, min_available_genotypes=n_genotypes,
                    min_second_sequence_carriers=second_sequence_n, n_ACRs=len(selected),
                    available_pairs=int(selected[f'{coverage}_n_available'].sum()),
                    occupied_5Mb_blocks=int(selected.block.nunique()),
                    chromosomes=int(selected.chrom.nunique())))
    return pd.DataFrame(rows)


def spatial_block_summary(frame, n_genotypes):
    rows = []
    for block, part in frame.groupby('block', sort=True):
        chrom = int(part.chrom.iloc[0])
        bins = (part.midpoint//5_000_000).astype(int)
        if part.chrom.nunique() != 1 or bins.nunique() != 1:
            raise ValueError('Frozen block does not represent one chromosome-specific 5-Mb interval')
        rows.append(dict(block=int(block), chromosome=chrom, block_number=int(bins.iloc[0]),
            start_bp=int(bins.iloc[0])*5_000_000, end_bp=(int(bins.iloc[0])+1)*5_000_000,
            n_ACRs=len(part), potential_pairs=len(part)*n_genotypes,
            core_available_pairs=int(part.core_n_available.sum()),
            context_available_pairs=int(part.context_n_available.sum()),
            context_variable_ACRs=int((part.context_n_sequences >= 2).sum()),
            core_variable_in_common_pairs_ACRs=int((part.common_core_n_sequences >= 2).sum())))
    return pd.DataFrame(rows)
