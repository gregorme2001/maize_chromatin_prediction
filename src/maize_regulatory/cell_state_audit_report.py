"""Descriptive audit tables and clearly labeled plots; no statistical testing."""

import base64
import html
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


def accessibility_tables(aggregation, output):
    """Reduce chunked summaries without loading a genotype-state-ACR tensor."""
    aggregation, output = Path(aggregation), Path(output)
    groups = pd.read_csv(aggregation / 'genotype_cell_state_nuclei.csv')
    libraries = pd.read_csv(aggregation / 'genotype_cell_state_library_nuclei.csv')
    acrs = pd.read_csv(aggregation / 'audited_acrs.csv')
    manifest = json.loads((aggregation / 'aggregation.json').read_text())
    thresholds = manifest['config']['minimum_nuclei']
    sampling = []
    for state, g in groups.groupby('cell_state', sort=True):
        observed = g.n_nuclei[g.n_nuclei > 0]
        record = {'cell_state': state, 'n_genotypes_observed': len(observed),
                  'n_nuclei': int(observed.sum()), 'median_nuclei_observed': float(observed.median()),
                  'min_nuclei_observed': int(observed.min()), 'max_nuclei': int(observed.max())}
        for n in thresholds:
            record[f'n_genotypes_ge_{n}_nuclei'] = int((g.n_nuclei >= n).sum())
        sampling.append(record)
    sampling = pd.DataFrame(sampling)
    sampling.to_csv(output / 'cell_state_sampling.csv', index=False)
    observed_n = groups.n_nuclei[groups.n_nuclei > 0]
    overview = {'n_possible_genotype_state_groups': len(groups),
                'n_observed_genotype_state_groups': len(observed_n),
                'n_groups_without_cells': int((groups.n_nuclei == 0).sum()),
                'n_library_labels': int(libraries.library.nunique()),
                'nuclei_per_observed_group_quantiles': {
                    str(q): float(observed_n.quantile(q)) for q in [0, .1, .25, .5, .75, .9, 1]},
                'below_nuclei_thresholds': {str(n): {
                    'observed_groups_below': int((observed_n < n).sum()),
                    'fraction_observed_below': float((observed_n < n).mean()),
                    'groups_including_absent_below': int((groups.n_nuclei < n).sum())}
                    for n in thresholds}}
    (output / 'sampling_overview.json').write_text(json.dumps(overview, indent=2) + '\n')
    groups.to_csv(output / 'genotype_cell_state_nuclei.csv', index=False)
    libraries.to_csv(output / 'genotype_cell_state_library_nuclei.csv', index=False)
    cats = pd.read_csv(aggregation / 'state_detection_sparsity.csv')
    histogram = pd.read_parquet(aggregation / 'nonzero_detection_frequency_histogram.parquet')
    for (state, n), h in histogram.groupby(['cell_state', 'min_nuclei']):
        cumulative = h.n_nonzero_blocks.cumsum().to_numpy()
        if not len(cumulative) or cumulative[-1] == 0:
            continue
        mask = cats.cell_state.eq(state) & cats.min_nuclei.eq(n)
        for label, quantile in [('q05', .05), ('q25', .25), ('median', .5), ('q75', .75), ('q95', .95)]:
            at = min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(h) - 1)
            cats.loc[mask, f'nonzero_frequency_{label}_lower'] = h.lower_frequency.iloc[at]
            cats.loc[mask, f'nonzero_frequency_{label}_upper'] = h.upper_frequency.iloc[at]
    cats.to_csv(output / 'cell_state_detection_summary.csv', index=False)
    repeated = {}
    variable = {}
    across = {(n, k): np.zeros(len(acrs), np.int16) for n in thresholds for k in [3, 5, 10]}
    acr_index = pd.Index(acrs.ACR)
    parquet = pq.ParquetFile(aggregation / 'acr_state_accessibility.parquet')
    columns = ['cell_state', 'min_nuclei', 'ACR', 'n_genotypes_eligible',
               'sd_detection_frequency', 'range_detection_frequency'] + [
                   f'n_genotypes_events_ge_{k}' for k in [3, 5, 10]]
    for batch in parquet.iter_batches(batch_size=131072, columns=columns):
        frame = batch.to_pandas()
        for (state, minimum), f in frame.groupby(['cell_state', 'min_nuclei'], sort=False):
            key = (state, int(minimum))
            rec = variable.setdefault(key, {'n_acrs_with_any_frequency_variation': 0,
                'n_acrs_sd_ge_one_percentage_point': 0,
                'n_acrs_sd_ge_one_percentage_point_and_5_detections_in_10_genotypes': 0})
            rec['n_acrs_with_any_frequency_variation'] += int((f.range_detection_frequency > 0).sum())
            sd = f.sd_detection_frequency >= .01
            rec['n_acrs_sd_ge_one_percentage_point'] += int(sd.sum())
            rec['n_acrs_sd_ge_one_percentage_point_and_5_detections_in_10_genotypes'] += int(
                (sd & (f.n_genotypes_events_ge_5 >= 10)).sum())
            at = acr_index.get_indexer(f.ACR)
            if np.any(at < 0):
                raise ValueError('Unknown ACR in chunk summary')
            for events, genotypes in itertools.product([3, 5, 10], [5, 10, 20]):
                key2 = (*key, events, genotypes)
                repeated[key2] = repeated.get(key2, 0) + int((f[f'n_genotypes_events_ge_{events}'] >= genotypes).sum())
            for events in [3, 5, 10]:
                across[int(minimum), events][at] += (f[f'n_genotypes_events_ge_{events}'] >= 5).to_numpy()
    var = pd.DataFrame([{'cell_state': k[0], 'min_nuclei': k[1], **v} for k, v in variable.items()])
    var.to_csv(output / 'cell_state_observed_variation.csv', index=False)
    repeat = pd.DataFrame([{'cell_state': k[0], 'min_nuclei': k[1], 'min_events': k[2],
                           'min_genotypes': k[3], 'n_acrs': v} for k, v in repeated.items()])
    repeat.to_csv(output / 'repeated_detection_acrs.csv', index=False)
    for (n, k), values in across.items():
        acrs[f'n_states_N{n}_events{k}_in_5_genotypes'] = values
    acrs.to_parquet(output / 'acr_multistate_support.parquet', index=False)
    return groups, libraries, sampling, cats


def library_diagnostics(aggregation, output):
    """Technical pool consistency and detected-ACR depth proxies, not biological replication."""
    aggregation, output = Path(aggregation), Path(output)
    cells = pd.read_csv(aggregation / 'source_cell_metadata.csv')
    selected = pd.read_csv(aggregation / 'selected_cell_rows.csv')
    qc = pd.read_csv(aggregation / 'cell_detected_acrs.csv.gz')
    if not np.array_equal(qc.matrix_row, selected.matrix_row):
        raise ValueError('Cell QC/development selection order mismatch')
    cells = cells.iloc[selected.matrix_row.to_numpy() - 1].copy()
    cells['detected_acrs'] = qc.detected_acrs.to_numpy()
    cells['celltype_full'] = cells.celltype_full.fillna('unannotated')
    cells['library'] = cells.library.fillna('unassigned_library')
    depth = cells.groupby(['normalized_genotype', 'celltype_full', 'library']).detected_acrs.agg(
        ['size', 'min', 'median', 'mean', 'max']).reset_index()
    depth.columns = ['genotype', 'cell_state', 'library', 'n_nuclei',
                     'min_detected_acrs', 'median_detected_acrs', 'mean_detected_acrs', 'max_detected_acrs']
    depth.to_csv(output / 'cell_depth_proxy_by_genotype_state_library.csv', index=False)
    lib = pd.read_csv(aggregation / 'genotype_cell_state_library_depth.csv')
    if int(qc.detected_acrs.sum()) != int(lib.total_detected_events.sum()):
        raise ValueError('Library aggregation/cell depth totals disagree')
    sample = sparse.load_npz(aggregation / 'library_qc_sample_counts.npz').tocsr()
    pairs = []
    for (genotype, state), rows in lib.groupby(['genotype', 'cell_state']).indices.items():
        rows = np.array([r for r in rows if lib.n_nuclei.iloc[r] >= 20])
        if len(rows) < 2:
            continue
        frequencies = sample[rows].toarray() / lib.n_nuclei.to_numpy()[rows, None]
        for a, b in itertools.combinations(range(len(rows)), 2):
            p, q = frequencies[a], frequencies[b]
            corr = float(np.corrcoef(p, q)[0, 1]) if p.std() > 0 and q.std() > 0 else np.nan
            pairs.append({'genotype': genotype, 'cell_state': state,
                          'library_1': lib.library.iloc[rows[a]], 'library_2': lib.library.iloc[rows[b]],
                          'n_nuclei_1': int(lib.n_nuclei.iloc[rows[a]]),
                          'n_nuclei_2': int(lib.n_nuclei.iloc[rows[b]]),
                          'n_sample_acrs': sample.shape[1], 'detection_frequency_correlation': corr,
                          'mean_absolute_frequency_difference': float(np.abs(p - q).mean())})
    columns = ['genotype', 'cell_state', 'library_1', 'library_2', 'n_nuclei_1', 'n_nuclei_2',
               'n_sample_acrs', 'detection_frequency_correlation', 'mean_absolute_frequency_difference']
    pairs = pd.DataFrame(pairs, columns=columns)
    pairs.to_csv(output / 'library_pair_consistency.csv', index=False)
    return depth, pairs


def candidate_panel(best, count):
    """Unique ACRs ranked by measurement support and call completeness, never effects."""
    if best.empty:
        return best.copy()
    best = best.copy()
    best['n_benchmark_supported_states'] = best.groupby('ACR').cell_state.transform('size')
    states = best.groupby('ACR').cell_state.agg(lambda values: ';'.join(sorted(values)))
    best['benchmark_supported_states'] = best.ACR.map(states)
    ranked = best.sort_values(['min_detected_genotypes', 'min_allele_genotypes',
        'state_call_rate', 'call_rate_audit', 'n_benchmark_supported_states', 'ACR', 'cell_state'],
        ascending=[False, False, False, False, False, True, True])
    ranked = ranked.drop_duplicates('ACR').head(count).reset_index(drop=True)
    ranked.insert(0, 'candidate_rank', np.arange(1, len(ranked) + 1))
    return ranked


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plots(output, groups, sampling, cats, acr_variants, joint, depth):
    output = Path(output)
    states = sampling.sort_values('n_genotypes_ge_100_nuclei', ascending=False).cell_state.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [2, 1]})
    heat = groups.pivot(index='cell_state', columns='genotype', values='n_nuclei').reindex(states)
    im = axes[0].imshow(np.log1p(heat.to_numpy()), aspect='auto', interpolation='nearest', cmap='viridis')
    axes[0].set_yticks(np.arange(len(states)), [s.replace('_', ' ') for s in states], fontsize=8)
    at = np.arange(0, len(heat.columns), 10)
    axes[0].set_xticks(at, heat.columns[at], rotation=90, fontsize=7)
    axes[0].set_xlabel('Development genotype (labels shown every tenth column; alphabetical order)')
    axes[0].set_ylabel('Cell-state annotation')
    axes[0].set_title('Nuclei per genotype and cell state; zero means no observed cells')
    fig.colorbar(im, ax=axes[0], label='Natural log(1 + number of nuclei)', shrink=.6)
    positive = groups.n_nuclei[groups.n_nuclei > 0]
    axes[1].hist(positive, bins=np.geomspace(1, positive.max() + 1, 40), color='#376b8d')
    axes[1].set_xscale('log')
    for threshold in [20, 50, 100]:
        axes[1].axvline(threshold, linestyle='--', label=f'{threshold} nuclei')
    axes[1].set_xlabel('Nuclei per observed genotype × cell-state group (log scale)')
    axes[1].set_ylabel('Number of observed groups')
    axes[1].set_title('Groups with zero nuclei are excluded from this histogram')
    axes[1].legend()
    fig.tight_layout()
    _save(fig, output / 'sampling.png')
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    c = cats.loc[cats.min_nuclei.eq(50)].set_index('cell_state').reindex(states)
    left = np.zeros(len(states))
    for label, title, color in zip(['zero', 'one', 'two_to_four', 'five_to_nine', 'at_least_ten'],
            ['0 nuclei detected', '1', '2–4', '5–9', '≥10'], ['#ddd', '#fed98e', '#fe9929', '#d95f0e', '#993404']):
        values = c[f'fraction_{label}'].fillna(0).to_numpy()
        axes[0].barh(np.arange(len(states)), values, left=left, label=title, color=color)
        left += values
    axes[0].set_yticks(np.arange(len(states)), [s.replace('_', ' ') for s in states], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Fraction of genotype × cell-state × ACR observations')
    axes[0].set_title('Detection counts; only groups with ≥50 nuclei')
    for row, state in enumerate(states):
        if c.loc[state, 'n_genotypes'] == 0:
            axes[0].text(.01, row, 'No genotype has ≥50 nuclei', va='center', fontsize=7, color='#555')
    axes[0].legend(fontsize=8, loc='lower right')
    repeat = pd.read_csv(output / 'repeated_detection_acrs.csv')
    for n, color in zip([20, 50, 100], ['#8ab6d6', '#376b8d', '#172f46']):
        r = repeat.loc[(repeat.min_nuclei == n) & (repeat.min_events == 5) & (repeat.min_genotypes == 10)]
        values = r.set_index('cell_state').n_acrs.reindex(states, fill_value=0)
        eligible = sampling.set_index('cell_state')[f'n_genotypes_ge_{n}_nuclei'].reindex(states)
        values = values.astype(float).where(eligible > 0)
        axes[1].plot(values, np.arange(len(states)), 'o', label=f'≥{n} nuclei/genotype', color=color)
    axes[1].set_yticks(np.arange(len(states)), [], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Number of ACRs with ≥5 detections in ≥10 genotypes')
    axes[1].set_title('Repeated detection across development genotypes')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output / 'detection.png')
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    bins = np.arange(0, max(20, int(acr_variants.n_panel_snps.quantile(.99))) + 2)
    axes[0].hist(np.minimum(acr_variants.n_panel_snps, bins[-1] - 1), bins=bins,
                 histtype='step', label='All listed SNPs', linewidth=2)
    axes[0].hist(np.minimum(acr_variants.n_mac_ge_10_callrate_ge_90pct, bins[-1] - 1),
                 bins=bins, histtype='step', label='Variable dosage, MAC ≥10, ≥90% called', linewidth=2)
    axes[0].set_xlabel(f'SNPs per audited chromosome-10 ACR (last bin includes ≥{bins[-2]})')
    axes[0].set_ylabel('Number of ACRs')
    axes[0].set_title('Observed panel coverage, including zero-SNP ACRs')
    axes[0].legend(fontsize=8)
    main = joint.loc[joint.name.eq('benchmark')].sort_values('n_joint_acrs', ascending=False).head(15)
    y = np.arange(len(main))
    for offset, field, label in [(-.25, 'n_genetic_acrs', 'Common, sufficiently called SNP'),
                                (0, 'n_allele_represented_acrs', 'Both homozygote groups represented'),
                                (.25, 'n_joint_acrs', 'Also enough detections in both groups')]:
        axes[1].barh(y + offset, main[field], height=.25, label=label)
    axes[1].set_yticks(y, main.cell_state.str.replace('_', ' '), fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Number of chromosome-10 ACRs')
    axes[1].set_title('Benchmark feasibility, top 15 states; not association results')
    axes[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    _save(fig, output / 'genetic_accessibility_overlap.png')
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    weighted = depth.assign(weighted_events=depth.mean_detected_acrs * depth.n_nuclei)
    totals = weighted.groupby(['genotype', 'cell_state'])[['weighted_events', 'n_nuclei']].sum()
    means = (totals.weighted_events / totals.n_nuclei).unstack()
    image = axes[0].imshow(np.log1p(means.reindex(columns=states).to_numpy().T),
                           aspect='auto', cmap='magma')
    axes[0].set_yticks(np.arange(len(states)), [s.replace('_', ' ') for s in states], fontsize=8)
    axes[0].set_xlabel('Development genotypes, alphabetical order')
    axes[0].set_ylabel('Cell state')
    axes[0].set_title('Mean detected ACRs per nucleus: depth proxy, not fragment depth')
    fig.colorbar(image, ax=axes[0], label='Natural log(1 + mean detected ACRs per nucleus)', shrink=.65)
    panel = pd.read_csv(output / 'candidate_pilot_acrs.csv')
    if not panel.empty:
        axes[1].hist(panel.accessibility_percentile_in_state_chr10 * 100, bins=np.arange(0, 105, 5),
                     alpha=.6, label='Accessibility percentile')
        axes[1].hist(panel.variability_percentile_in_state_chr10 * 100, bins=np.arange(0, 105, 5),
                     histtype='step', linewidth=2, label='Variability percentile')
    axes[1].set_xlabel('Percentile among chromosome-10 ACRs in the selected cell state')
    axes[1].set_ylabel('Number of unique candidate ACRs')
    axes[1].set_title('Pilot selection bias; no effect sizes or p-values used')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output / 'depth_and_selection_bias.png')


def write_report(output, aggregation, manifest, groups, sampling, cats, acr_variants,
                 joint, candidates, pairs, variant_summary):
    output, aggregation = Path(output), Path(aggregation)
    benchmark = joint.loc[joint.name.eq('benchmark')].sort_values('n_joint_acrs', ascending=False)
    observed = groups.n_nuclei[groups.n_nuclei > 0]
    top = benchmark.head(8)
    columns = ['cell_state', 'n_genotypes_eligible', 'n_accessibility_supported_acrs',
               'n_allele_represented_acrs', 'n_joint_acrs']
    table = top[columns].to_csv(index=False)
    composition = candidates.cell_state.value_counts().to_dict() if len(candidates) else {}
    bias = (f"The candidate median accessibility percentile is {candidates.accessibility_percentile_in_state_chr10.median():.1%}; "
            f"the median variability percentile is {candidates.variability_percentile_in_state_chr10.median():.1%}. "
            "Percentiles compare each region with all audited chromosome-10 ACRs in its selected state. "
            "This quantifies selection toward strongly measured regions; it does not establish biological importance."
            if len(candidates) else 'No regions met the benchmark, so no candidate pilot panel was selected.')
    verification = output / 'independent_bulk_count_check.json'
    verification_text = ''
    if verification.exists():
        check = json.loads(verification.read_text())
        verification_text = (f" An independent comparison against the existing genotype-by-ACR raw count artifact "
                             f"checked {check['compared_entries']:,} development-genotype/chromosome-10 ACR entries "
                             f"after summing across states: {check['mismatches']} mismatches. "
                             "See independent_bulk_count_check.json and validation/source_genotype_mapping.csv.")
    source = pd.read_csv(aggregation / 'stream_validation.csv').iloc[0]
    sections = [
        ('Scope and principal findings',
         f"Analyzed {manifest['n_cells']:,} nuclei from {manifest['n_genotypes']} eligible development genotypes, "
         f"{manifest['n_states']} annotation labels and {manifest['n_acrs']:,} ACRs. "
         f"There are {len(observed):,}/{len(groups):,} observed genotype–state combinations; "
         f"median nuclei among observed combinations: {observed.median():.0f}. "
         f"The genetic intersection covers {len(acr_variants):,} chromosome-10 ACRs only. "
         f"{variant_summary['acrs_variable']:,} have observed variable SNP dosages; "
         f"{variant_summary['acrs_common_called']:,} have a dosage-variable SNP with MAC ≥10 and ≥90% of development genotypes called. "
         f"Selected {len(candidates)} unique pilot ACRs under the benchmark. "
         "The other chromosomes have not been genetically audited; unexamined chromosomes are not classified as sequence-invariant."),
        ('Data and identity checks',
         "Inputs: data/intermediate/cells/cell_by_acr.sparse.rds, cell_metadata.csv and acr_index.csv; "
         "data/maize_acr_sequences.parquet; data/intermediate/panzea_v5_pilot/chr10_variants.parquet and chr10_alt_dosage.parquet. "
         "Original and normalized cell genotype labels, cell IDs and matrix row order are checked explicitly. "
         "The permanent development order is checked before genetic/accessibility joins. "
         "VCF coordinates are 1-based; ACRs are 0-based half-open B73-v5 intervals. "
         "Every used SNP REF and position is checked again against the stored B73 ACR sequence, and cached call counts against dosages. "
         f"Positive source entries other than one among audited cells/ACRs: {int(source.nonbinary_positive_source_entries):,}. "
         "All positive entries are counted once per nucleus as detections. Zero entries are observed nondetections, not missing records."
         + verification_text),
        ('Definitions and denominators',
         "N is the number of observed nuclei in one genotype and cell state. Y is the number of those nuclei with a positive event at one ACR. "
         "p = Y/N is an empirical detection frequency, not a perfectly observed biological accessibility probability. "
         "Groups with N=0 have undefined p and are excluded from frequency summaries; their absence remains explicit in the nuclei table. "
         "For each state and N threshold (20, 50, 100), the sparsity denominator is eligible genotypes × all audited ACRs. "
         "Zero-detection blocks are counted analytically. Frequency means/SDs weight eligible genotypes equally. "
         "Nonzero frequency quantiles are bounded using fixed 0.001-wide histogram bins. "
         "Observed SD and range combine biology, sampling and technical noise; an SD ≥0.01 means one percentage point and is a descriptive threshold only."),
        ('Genetic definitions',
         "Missing dosage is -1, reference homozygote 0, heterozygote 1 and alternate homozygote 2. No calls are imputed. "
         "MAC is the smaller of the observed REF and ALT allele counts; its denominator is twice the number of called diploid genotypes. "
         "MAC counts alleles, not lines. Dosage variability is checked separately so identical heterozygous calls across lines do not count as genotype-dependent variation. "
         "Complete local dosage patterns use only genotypes called at every listed SNP in that ACR; missing calls are never treated as alleles. "
         "They are unphased dosage patterns, not reconstructed haplotypes. A zero-SNP ACR has unknown sequence variation, not established invariance. "
         "This panel contains biallelic SNPs, not a complete indel/structural-variation catalogue. "
         "GT-only calls lack per-genotype DP/GQ and populated site QUAL/MQ; call-rate and frequency criteria cannot certify variant accuracy."),
        ('Joint feasibility criteria',
         "Each ACR is accepted only if at least one single SNP satisfies all criteria in that state. "
         "The benchmark requires MAC ≥10, ≥90% call rate overall and among state-eligible genotypes, ≥50 nuclei per eligible genotype, "
         "≥10 reference-homozygous and ≥10 alternate-homozygous eligible genotypes, and ≥5 genotypes in EACH group with ≥5 detected nuclei at the ACR. "
         "Heterozygotes count as called but are excluded from the two homozygote comparison groups. "
         "Lenient: MAC ≥5, call rate ≥80%, ≥20 nuclei, ≥5 genotypes per group, ≥3 genotypes per group with ≥3 detections. "
         "Conservative: MAC ≥10, call rate ≥90%, ≥100 nuclei, ≥10 genotypes per group, ≥10 genotypes per group with ≥10 detections. "
         "Additional N/events sensitivity combinations retain the benchmark group-size and call-rate criteria. "
         "These are exploratory measurement criteria, not established biological standards. "
         "Requiring detections in both groups deliberately excludes some potential complete-loss-of-accessibility cases; it is a conservative measured-pilot screen, not a census of all testable effects."),
        ('Pilot ranking and selection bias',
         "Within each ACR/state, choose the qualifying SNP with the most detected genotypes in the smaller homozygote group, "
         "then the most eligible genotypes in that group, then call completeness, with deterministic ID ties. "
         "Rank unique ACRs by the same measurement support, call completeness and number of supported states. "
         "No association significance, allele-effect size, motif score or phenotype enters ranking. "
         f"Candidate state composition: {composition}. " + bias),
        ('Libraries, depth and replication',
         "Aggregation first preserves genotype × cell-state × library groups. Full chromosome-10 library-level counts are retained as sparse chunks; "
         "whole-audit library totals and a fixed random ACR sample are also retained. Libraries are labeled pools, not verified biological replicates. "
         "Detailed fragment depth, TSS enrichment, FRiP, donor/plant identifiers and independent biological-replicate labels are absent. "
         "Detected ACRs per nucleus is a depth proxy, not sequencing depth. "
         f"Compared {len(pairs):,} within-genotype/state library pairs with ≥20 nuclei per library on a fixed random ACR sample. "
         "These differences reflect sampling, technical and possible biological effects; they do not estimate pure biological-replicate variance. "
         "Nuclei are nested observations and do not increase the number of independent genotypes."),
        ('Recommendation and limits',
         ("A restricted chromosome-10, well-sampled-state pilot is supported by the measurement criteria. "
          if len(candidates) else "The benchmark does not support a pilot panel; inspect the lenient criteria and measurement limitations before proceeding. ")
         + f"The strongest benchmark state is {benchmark.iloc[0].cell_state}, with {int(benchmark.iloc[0].n_joint_acrs):,} supported ACRs. "
         "State differences in support can reflect nucleus counts and detection depth, not just biology. "
         "The global ranking is not designed to balance cell states; a state-balanced pilot would require an explicitly different selection rule. "
         "Proceed next with inspection/validation of the selected SNP calls and library/depth effects before genotype-specific motif scoring or inference. "
         "Any eventual association analysis must address technical variation, population structure, relatedness and ACR ascertainment. "
         "This audit does not show genetic causality, TF binding, regulatory grammar or phenotype prediction. "
         "CML341 is excluded, the original 30-genotype holdout is unchanged and unanalyzed, and no DTS values were loaded. "
         "No external data were downloaded, sequences reconstructed, association tests fitted or FDR tests performed."),
        ('Reproduction and outputs',
         "Run .venv/bin/python scripts/audit_cell_state_accessibility.py --output-dir results/cell_state_accessibility_audit_NEW. "
         "The default output directory must be new or empty. Use --chromosome 10 --max-acrs 64 for a bounded smoke run. "
         f"Aggregation cache: {aggregation.resolve()}. Use --aggregation-dir with that directory and a new --output-dir to reuse validated aggregation. "
         "run.json contains input/code hashes and the exact configuration; aggregation/stream.log contains R progress. "
         "The code processes bounded ACR chunks and never creates a dense cell-by-ACR matrix or full genotype-state-ACR tensor. "
         "The per-ACR/state accessibility summary is Parquet; chromosome-10 library/pooled counts are sparse NPZ, not raw cell-level exports.")
    ]
    captions = {
        'sampling.png': 'Sampling: each heatmap tile is one genotype/state nucleus count; the histogram excludes combinations with no cells. Zero-cell groups are retained in the CSV.',
        'detection.png': 'Detection: Y is the number of nuclei with a positive event. Left: fractions of ACR/group observations by Y category for N≥50. Right: ACRs with Y≥5 in at least ten eligible genotypes; eligibility uses N≥20, 50 or 100.',
        'genetic_accessibility_overlap.png': 'Genetic coverage and intersection: MAC is minor allele count among called diploid genotypes. Counts concern chromosome 10 only. Benchmark bars show progressively stricter data-availability criteria; they are not association discoveries.',
        'depth_and_selection_bias.png': 'Depth proxy and selection bias: heatmap colors show natural log(1 + mean detected ACRs per nucleus); white tiles have no observed nuclei. Detected-ACR counts are not fragment sequencing depth. Candidate percentiles are relative to chromosome-10 ACRs in each candidate’s selected state; high percentiles indicate a measurement-selected panel.'
    }
    md = ['# Sparse cell-state accessibility and chromosome-10 variation audit', '']
    for title, paragraph in sections:
        md.extend([f'## {title}', '', paragraph, ''])
        if title == 'Joint feasibility criteria':
            md.extend(['Top benchmark states (CSV-formatted excerpt):', '', '```csv', table.rstrip(), '```', ''])
    for filename, caption in captions.items():
        md.extend([f'![{caption}]({filename})', '', caption, ''])
    md.extend(['## Output tables', '', *[f'- `{p.name}`' for p in sorted(output.iterdir()) if p.suffix in ['.csv', '.parquet']], ''])
    (output / 'report.md').write_text('\n'.join(md))
    parts = ['<!doctype html><meta charset="utf-8"><title>Cell-state accessibility audit</title>',
             '<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}h2{margin-top:2em}</style>',
             '<h1>Sparse accessibility and chromosome-10 variation audit</h1>']
    for title, paragraph in sections:
        parts += [f'<h2>{html.escape(title)}</h2><p>{html.escape(paragraph)}</p>']
        if title == 'Joint feasibility criteria':
            parts.append(top[columns].to_html(index=False))
    for filename, caption in captions.items():
        encoded = base64.b64encode((output / filename).read_bytes()).decode()
        parts.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    parts += ['<h2>Tables</h2><ul>']
    parts += [f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in sorted(output.iterdir()) if p.suffix in ['.csv', '.parquet']]
    parts.append('</ul>')
    (output / 'report.html').write_text('\n'.join(parts))
