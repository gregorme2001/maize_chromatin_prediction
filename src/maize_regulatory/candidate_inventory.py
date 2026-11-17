"""Descriptive sequence/cell-support inventory; no model or outcome selection."""

import numpy as np
import pandas as pd


def prepare_variant_support(variants, dosage, genotypes, acrs, *, call_rate=.9):
    """Validate identities and retain all panel SNPs for sequence completeness.

    MAF is based on called diploid alleles. Missing and heterozygous calls do
    not produce a reconstructable single sequence. Low-frequency/low-call-rate
    SNPs still contribute to this completeness check, even if not screening SNPs.
    """
    if (not variants.index.equals(dosage.index)
            or not dosage.columns.equals(genotypes)
            or not variants.index.is_unique or not acrs.is_unique):
        raise ValueError('Variant, genotype or ACR identities/order disagree')
    if not variants.ref_matches_b73.all():
        raise ValueError('Variant reference check failed')
    locations = acrs.get_indexer(variants.ACR)
    if np.any(locations < 0):
        raise ValueError('A variant has no measured ACR')
    calls = dosage.to_numpy()
    if not np.isin(calls, [-1, 0, 1, 2]).all():
        raise ValueError('Unexpected dosage coding')
    called = calls >= 0
    n_called = called.sum(axis=1)
    alternate = np.where(called, calls, 0).sum(axis=1)
    mac = np.minimum(alternate, 2 * n_called - alternate)
    maf = np.divide(mac, 2 * n_called, out=np.zeros(len(calls)), where=n_called > 0)
    complete = np.ones((len(acrs), len(genotypes)), dtype=bool)
    homozygous = complete.copy()
    rows_by_acr = []
    # Source variant order is retained; no sorting or inferred alignment.
    for i in range(len(acrs)):
        rows = np.flatnonzero(locations == i)
        rows_by_acr.append(rows)
        if len(rows):
            complete[i] = called[rows].all(axis=0)
            homozygous[i] = np.isin(calls[rows], [0, 2]).all(axis=0)
    return dict(calls=calls, locations=locations, maf=maf,
                rate_ok=n_called >= call_rate * len(genotypes),
                complete=complete, homozygous=homozygous,
                rows_by_acr=rows_by_acr, acrs=acrs, genotypes=genotypes)


def support_at_cell_threshold(support, nuclei, min_nuclei, min_maf):
    """Recheck MAF among sequence-usable, adequately sampled genotypes per ACR.

    A screening SNP must pass both full-development call-rate/MAF criteria and
    MAF among the actual usable genotypes. This prevents an apparent common SNP
    being rare/absent in the cell-state data we can use.
    """
    nuclei = np.asarray(nuclei)
    if nuclei.shape != (len(support['genotypes']),) or np.any(nuclei < 0):
        raise ValueError('Nucleus counts do not match genotype axis')
    cell_ok = nuclei >= min_nuclei
    usable = support['homozygous'] & cell_ok
    n_usable = usable.sum(axis=1)
    variant_usable = usable[support['locations']]
    alt = ((support['calls'] == 2) & variant_usable).sum(axis=1)
    n = n_usable[support['locations']]
    minor_lines = np.minimum(alt, n - alt)
    local_maf = np.divide(minor_lines, n, out=np.zeros(len(n)), where=n > 0)
    global_good = support['rate_ok'] & (support['maf'] >= min_maf)
    final_good = global_good & (local_maf >= min_maf) & (n > 0)
    size = len(support['acrs'])
    return dict(
        usable=usable, n_usable=n_usable, n_cell_eligible=int(cell_ok.sum()),
        n_complete=(support['complete'] & cell_ok).sum(axis=1),
        n_global_snps=np.bincount(support['locations'][global_good], minlength=size),
        n_usable_common_snps=np.bincount(support['locations'][final_good], minlength=size),
        minor_lines=minor_lines, final_good=final_good,
    )


def candidate_grid(support, counts, groups, *, min_cells=(20, 50, 100),
                   mafs=(.05, .1, .2), min_snps=(1, 2, 5),
                   min_genotypes=(30, 50, 75)):
    """Inventory all states; return a grid plus one provisional N50/MAF10% panel.

    The provisional panel requires >=2 screening SNPs and >=50 usable genotypes.
    It never filters on observed accessibility, its variance or an association.
    """
    if counts.shape != (len(groups), len(support['acrs'])):
        raise ValueError('Accessibility axes do not match metadata')
    keys = pd.MultiIndex.from_frame(groups[['genotype', 'cell_state']])
    if not keys.is_unique:
        raise ValueError('Duplicate genotype/state measurements')
    grid, candidates, observations, stages = [], [], [], []
    for state in groups.cell_state.unique():
        rows = np.flatnonzero(groups.cell_state.eq(state).to_numpy())
        if not pd.Index(groups.genotype.iloc[rows], name=support['genotypes'].name).equals(
                support['genotypes']):
            raise ValueError('Cell-state genotype order mismatch')
        nuclei = groups.n_nuclei.to_numpy()[rows]
        y = counts[rows].toarray()
        if np.any(y < 0) or np.any(y > nuclei[:, None]):
            raise ValueError('Detected nucleus count outside [0, N]')
        for minimum in min_cells:
            for maf in mafs:
                s = support_at_cell_threshold(support, nuclei, minimum, maf)
                for k in min_snps:
                    genetic = s['n_global_snps'] >= k
                    for ng in min_genotypes:
                        enough = genetic & (s['n_usable'] >= ng)
                        final = enough & (s['n_usable_common_snps'] >= k)
                        grid.append(dict(cell_state=state, min_nuclei=minimum,
                            min_maf=maf, min_snps=k, min_usable_genotypes=ng,
                            n_cell_eligible_genotypes=s['n_cell_eligible'],
                            n_genetic_acrs=int(genetic.sum()),
                            n_acrs_enough_complete_homozygous=int(enough.sum()),
                            n_candidate_acrs=int(final.sum()),
                            n_acr_genotype_observations=int(s['n_usable'][final].sum())))
                if minimum != 50 or maf != .1:
                    continue
                genetic = s['n_global_snps'] >= 2
                final = genetic & (s['n_usable'] >= 50) & (s['n_usable_common_snps'] >= 2)
                stages.append(dict(cell_state=state,
                    n_cell_eligible_genotypes=s['n_cell_eligible'],
                    n_acrs_global_two_snps=int(genetic.sum()),
                    n_potential_observations=int(genetic.sum() * s['n_cell_eligible']),
                    n_complete_observations=int(s['n_complete'][genetic].sum()),
                    n_complete_homozygous_observations=int(s['n_usable'][genetic].sum()),
                    n_acrs_50_complete=int((genetic & (s['n_complete'] >= 50)).sum()),
                    n_acrs_50_complete_homozygous=int((genetic & (s['n_usable'] >= 50)).sum()),
                    n_final_acrs=int(final.sum()),
                    n_final_observations=int(s['n_usable'][final].sum())))
                for i in np.flatnonzero(final):
                    good_g = np.flatnonzero(s['usable'][i])
                    snp_rows = support['rows_by_acr'][i]
                    patterns, pattern_counts = np.unique(
                        support['calls'][snp_rows][:, good_g].T, axis=0, return_counts=True)
                    good_snp_rows = snp_rows[s['final_good'][snp_rows]]
                    candidates.append(dict(ACR=support['acrs'][i], cell_state=state,
                        n_usable_genotypes=len(good_g),
                        n_panel_snps=len(snp_rows),
                        n_global_common_snps=int(s['n_global_snps'][i]),
                        n_usable_common_snps=int(s['n_usable_common_snps'][i]),
                        n_distinct_panel_sequences=len(patterns),
                        n_sequence_groups_ge_5_genotypes=int((pattern_counts >= 5).sum()),
                        smallest_minor_group_among_screening_snps=int(
                            s['minor_lines'][good_snp_rows].min()),
                        n_nuclei_total=int(nuclei[good_g].sum()),
                        n_zero_detection_observations=int((y[good_g, i] == 0).sum())))
                    observations.append(pd.DataFrame(dict(
                        ACR=support['acrs'][i], cell_state=state,
                        genotype=support['genotypes'][good_g],
                        n_nuclei=nuclei[good_g], n_detected=y[good_g, i])))
        print(f'Candidate inventory: {state}', flush=True)
    obs = (pd.concat(observations, ignore_index=True) if observations else
           pd.DataFrame(columns=['ACR', 'cell_state', 'genotype', 'n_nuclei', 'n_detected']))
    return pd.DataFrame(grid), pd.DataFrame(candidates), obs, pd.DataFrame(stages)
