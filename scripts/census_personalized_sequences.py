#!/usr/bin/env python
"""Resume a genome-wide sequence-availability census; no model fitting or sampling."""

import argparse
import gc
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from maize_regulatory.cohort import development_genotypes
from maize_regulatory.genomewide_census import (
    census_from_variants, coverage_diversity_grid, spatial_block_summary,
)
from maize_regulatory.personalized_context_audit import collect_context_variants, context_windows
from maize_regulatory.reference_sequence import digest


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_axes(root):
    previous = root/'results/sequence_cnn_benchmark'
    coordinates = pd.read_csv(previous/'acr_coordinates.csv', index_col='ACR')
    coverage = pd.read_csv(previous/'genotype_coverage.csv')
    if coverage.genotype.tolist() != development_genotypes().tolist():
        raise ValueError('Development cohort order differs from existing coverage')
    genotypes = pd.Index(coverage.loc[coverage.eligible, 'genotype'], name='genotype')
    receipt = json.loads((root/'data/intermediate/sequence_cnn/receipt.json').read_text())
    if receipt['source']['genotype_order'] != genotypes.tolist():
        raise ValueError('Eligible prediction genotype order differs')
    sequences = pd.read_parquet(root/'data/maize_acr_sequences.parquet')
    if not sequences.index.equals(coordinates.index):
        raise ValueError('Global B73 sequence and ACR axes differ')
    coordinates['global_acr_index'] = np.arange(len(coordinates))
    return coordinates, sequences, genotypes


def run(root, output, cache, chromosomes):
    from maize_regulatory.sequence_cnn import IndexedFasta
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    coordinates, sequences, genotypes = load_axes(root)
    source_manifest = json.loads((root/'configs/panzea_v5_sources.json').read_text())
    source_by_chrom = {entry['chromosome']: entry for entry in source_manifest['files']
                       if entry['name'].endswith('.vcf.gz')}
    code_paths = ['scripts/census_personalized_sequences.py',
                  'src/maize_regulatory/genomewide_census.py',
                  'src/maize_regulatory/personalized_context_audit.py',
                  'src/maize_regulatory/variant_pilot.py',
                  'src/maize_regulatory/sequence_cnn.py']
    state = dict(width_bp=2000, genotypes=genotypes.tolist(),
        source_manifest_sha256=digest(root/'configs/panzea_v5_sources.json'),
        coordinates_sha256=digest(root/'results/sequence_cnn_benchmark/acr_coordinates.csv'),
        sequence_sha256=digest(root/'data/maize_acr_sequences.parquet'),
        fasta_sha256=digest(root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa'),
        fasta_index_sha256=digest(root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa.fai'),
        code_sha256={relative: digest(root/relative) for relative in code_paths},
        selection='Existing 137 development prediction genotypes; no MAF, phenotype, or accessibility screen',
        pattern_ids='Local to each ACR; -1 unavailable; never global feature-table IDs',
        holdout_analyzed=False, DTS_read=False, accessibility_values_read=False,
        model_fitted=False, cohort_sampled=False, CV_changed=False)
    state_path = output/'run_config.json'
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError('Census code/input/config changed; choose a new output/cache directory')
    save_json(state_path, state)
    pd.DataFrame({'genotype': genotypes}).to_csv(output/'genotypes.csv', index=False)
    for relative in code_paths:
        target = output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, target)
    for chromosome in chromosomes:
        destination = cache/f'chr{chromosome}'
        destination.mkdir(parents=True, exist_ok=True)
        receipt_path = destination/'receipt.json'
        if receipt_path.exists():
            if json.loads(receipt_path.read_text())['state'] != state:
                raise ValueError('Chromosome cache state mismatch')
            print(f'Keeping verified chromosome {chromosome} census', flush=True)
            continue
        source = source_by_chrom[chromosome]
        vcf = root/'data/raw/panzea_v5'/source['name']
        if not vcf.exists():
            raise FileNotFoundError(f'Download has not completed: {vcf}')
        print(f'Chromosome {chromosome}: sequence-only census', flush=True)
        started = time.monotonic()
        selected = coordinates.loc[coordinates.chrom.eq(chromosome)].copy()
        fasta = IndexedFasta(root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa')
        try:
            reference = fasta.fetch(f'chr{chromosome}', 0, fasta.entries[f'chr{chromosome}'][0])
        finally:
            fasta.handle.close()
        for acr, row in selected.iterrows():
            if reference[int(row.start):int(row.end)].decode() != sequences.loc[acr, 'sequence']:
                raise ValueError(f'Core does not match B73: {acr}')
        windows = context_windows(selected, len(reference))
        variants, calls, mapping, scan = collect_context_variants(vcf, windows, genotypes, reference)
        variants.to_parquet(destination/'context_variants.parquet', index=False)
        np.save(destination/'context_dosage.npy', calls)
        mapping.to_csv(destination/'vcf_sample_mapping.csv')
        frame, genotype_frame, arrays, summaries = census_from_variants(windows, variants, calls, reference)
        frame.to_parquet(destination/'per_acr.parquet')
        genotype_frame.index = genotypes
        genotype_frame.to_csv(destination/'per_genotype.csv')
        np.savez_compressed(destination/'pair_support.npz', **arrays)
        comparisons = {}
        if chromosome == 10:
            old_dir = root/'results/personalized_chr10_audit'
            old_windows = pd.read_csv(old_dir/'windows.csv', index_col='ACR')
            if not old_windows.index.equals(frame.index):
                raise ValueError('Prior chromosome-10 ACR order differs')
            for kind in ['core', 'context']:
                previous = np.load(old_dir/f'{kind}_support.npz')['pattern_ids'] >= 0
                if not np.array_equal(previous, arrays[f'{kind}_available']):
                    raise AssertionError('Chromosome-10 availability differs from earlier audit')
                comparisons[f'{kind}_pair_masks_identical_to_previous_audit'] = True
        receipt = dict(state=state, chromosome=chromosome, n_ACRs=len(selected),
            source=source, vcf_sha256=digest(vcf), scan=scan, summaries=summaries,
            sequence_patterns_counted_without_materializing_DNA=True,
            chromosome10_comparison=comparisons, seconds=time.monotonic()-started)
        save_json(receipt_path, receipt)
        print(json.dumps(dict(chromosome=chromosome, seconds=receipt['seconds'],
            core_pairs=summaries['core']['available_pairs'],
            context_pairs=summaries['context']['available_pairs'],
            context_variable_ACRs=summaries['context']['acrs_with_two_sequences'])), flush=True)
        del reference, variants, calls, arrays, frame, genotype_frame
        gc.collect()


def aggregate(root, output, cache):
    _, _, genotypes = load_axes(root)
    directories = [cache/f'chr{chromosome}' for chromosome in range(1, 11)]
    if not all((directory/'receipt.json').exists() for directory in directories):
        raise ValueError('All ten completed chromosome receipts are required')
    frames = [pd.read_parquet(directory/'per_acr.parquet') for directory in directories]
    frame = pd.concat(frames).sort_values('global_acr_index')
    expected = pd.read_csv(root/'results/sequence_cnn_benchmark/acr_coordinates.csv').ACR.tolist()
    if frame.index.tolist() != expected:
        raise AssertionError('Aggregated ACR order differs from frozen global source')
    frame.to_parquet(output/'per_acr_census.parquet')
    frame.to_csv(output/'per_acr_census.csv')
    receipts = [json.loads((directory/'receipt.json').read_text()) for directory in directories]
    chromosome_rows = []
    for receipt, part in zip(receipts, frames):
        row = dict(chromosome=receipt['chromosome'], n_ACRs=receipt['n_ACRs'],
                   listed_context_SNPs=receipt['scan']['context_biallelic_snps'])
        for kind in ['core', 'context']:
            row.update({f'{kind}_{key}': value for key, value in receipt['summaries'][kind].items()})
            row[f'{kind}_median_genotypes_per_ACR'] = float(part[f'{kind}_n_available'].median())
        chromosome_rows.append(row)
    chromosome_frame = pd.DataFrame(chromosome_rows)
    chromosome_frame.to_csv(output/'per_chromosome.csv', index=False)
    genotype_frames = [pd.read_csv(directory/'per_genotype.csv', index_col='genotype')
                       for directory in directories]
    if any(part.index.tolist() != genotypes.tolist() for part in genotype_frames):
        raise ValueError('Per-chromosome genotype axes differ')
    genotype_frame = sum(genotype_frames[1:], genotype_frames[0].copy())
    genotype_frame.to_csv(output/'per_genotype.csv')
    grid = coverage_diversity_grid(frame)
    grid.to_csv(output/'coverage_diversity_grid.csv', index=False)
    blocks = spatial_block_summary(frame, len(genotypes))
    blocks.to_csv(output/'spatial_blocks.csv', index=False)
    summary = dict(n_ACRs=len(frame), n_genotypes=len(genotypes), total_possible_pairs=len(frame)*len(genotypes),
        chromosomes=10, width_bp=2000, occupied_5Mb_blocks=len(blocks),
        source_context_SNPs=int(chromosome_frame.listed_context_SNPs.sum()),
        all_context_SNP_REF_match_B73=all(r['scan']['all_context_REF_match_B73'] for r in receipts),
        census_complete=True, phenotype_or_accessibility_values_read=False,
        holdout_analyzed=False, models_fitted=False, final_cohort_sampled=False,
        total_chromosome_processing_seconds=sum(r['seconds'] for r in receipts))
    for kind in ['core', 'context']:
        summary[kind] = dict(available_pairs=int(frame[f'{kind}_n_available'].sum()),
            available_fraction=float(frame[f'{kind}_n_available'].sum()/(len(frame)*len(genotypes))),
            ACRs_with_any_pair=int((frame[f'{kind}_n_available'] > 0).sum()),
            median_genotypes_per_ACR=float(frame[f'{kind}_n_available'].median()),
            ACRs_with_two_sequences=int((frame[f'{kind}_n_sequences'] >= 2).sum()),
            distinct_available_sequences=int(frame[f'{kind}_n_sequences'].sum()),
            non_B73_pairs=int(frame[f'{kind}_non_B73_pairs'].sum()))
    save_json(output/'summary.json', summary)
    return frame, genotype_frame, chromosome_frame, blocks, grid, summary


def report(root, output, cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame, genotypes, chromosomes, blocks, grid, summary = aggregate(root, output, cache)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    max_bin = int(blocks.block_number.max())+1
    for ax, field, title, label, vmax in [
        (axes[0], 'context_available_pairs', 'Complete 2,000-bp sequence availability',
         'Available ACR–genotype pairs (%)', 100),
        (axes[1], 'context_variable_ACRs', 'ACRs with at least two available context sequences',
         'ACRs per 5-Mb interval', None)]:
        data = np.full((10, max_bin), np.nan)
        for row in blocks.itertuples():
            value = getattr(row, field)
            if field == 'context_available_pairs':
                value = 100*value/row.potential_pairs
            data[row.chromosome-1, row.block_number] = value
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
        image = ax.imshow(data, aspect='auto', origin='upper', cmap=cmap, vmin=0, vmax=vmax,
                          extent=[0, max_bin*5, 10.5, .5])
        ax.set(title=title, xlabel='Genomic position (Mb; each column is 5 Mb)', ylabel='Chromosome')
        ax.set_yticks(range(1, 11)); fig.colorbar(image, ax=ax, label=label)
    fig.savefig(output/'spatial_coverage.png', dpi=160); plt.close(fig)
    # Embed the figure for a standalone, shareable offline HTML report.
    import base64
    encoded = base64.b64encode((output/'spatial_coverage.png').read_bytes()).decode()
    main_table = chromosomes[['chromosome', 'n_ACRs', 'core_available_pairs',
        'context_available_pairs', 'context_median_genotypes_per_ACR', 'context_acrs_with_two_sequences']]
    main_table.columns = ['Chromosome', 'ACRs', 'Core pairs', '2,000-bp pairs',
                          'Median genotypes/ACR (context)', 'ACRs with ≥2 context sequences']
    compact_grid = grid.loc[grid.min_available_genotypes.isin([20, 50, 80])
                           & grid.min_second_sequence_carriers.isin([0, 5])]
    parts = ['<!doctype html><html><meta charset="utf-8"><title>Genome-wide sequence census</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;line-height:1.5}'
        'table{border-collapse:collapse}td,th{padding:5px 9px;border:1px solid #ccc}'
        'img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Genome-wide personalized-sequence census</h1>',
        '<p>Availability and sequence diversity only. No motif scoring, model fitting, phenotype '
        'analysis or final cohort selection. The 30-genotype holdout remains untouched.</p>',
        f'<p>{len(frame):,} ACRs × {summary["n_genotypes"]} existing development prediction genotypes '
        f'= {summary["total_possible_pairs"]:,} potential pairs. Cortex measurement eligibility '
        'is unchanged (≥50 pooled nuclei). No accessibility values were read.</p>',
        '<h2>What qualifies?</h2><p>A sequence must have homozygous, nonmissing calls at every listed '
        'biallelic SNP and contain only A/C/G/T reference DNA. No MAF, SNP-count, phenotype or '
        'accessibility-variance screen. Context is 2,000 bp total, with the same frozen block-boundary '
        'clipping; clipped/padded or ambiguous windows are unavailable. Core DNA is never truncated.</p>',
        '<pre>'+html.escape(json.dumps(summary, indent=2))+'</pre>', main_table.to_html(index=False),
        '<h2>Spatial coverage</h2>', '<img alt="Chromosome maps of sequence availability and variable ACR counts" '
        f'src="data:image/png;base64,{encoded}">',
        '<p>Top: available pairs divided by all 137 × ACRs in each interval. Bottom: number of ACRs '
        'with at least two distinct sequences among available genotypes. Gray means no source ACRs '
        'in that interval (or outside that chromosome), not zero availability.</p>',
        '<h2>Coverage and sequence-diversity inventory</h2>',
        '<p>These are descriptive thresholds, not adopted filters. “second sequence carriers” is '
        'the number of genotypes carrying the second most frequent distinct sequence. A value ≥5 '
        'guarantees at least two sequence classes each represented by five genotypes. '
        '“core” uses core-complete pairs; “context” uses context-complete pairs; “common_core” counts '
        'distinct core sequences among those same context-complete pairs, for a matched experiment.</p>',
        compact_grid.to_html(index=False),
        '<h2>Sampling recommendation</h2><p>Sample ACRs first, within prespecified genomic intervals, '
        'then retain their available genotypes. Uniform sampling of pairs favors ACRs with many '
        'available genotypes. Approximately equal ACR quotas per occupied 5-Mb interval favor spatial '
        'coverage; chromosome allocations should reflect represented genomic length, not equal quotas '
        'per chromosome. Sparse intervals need a stated fallback. Use fixed seeds and retain the '
        'existing block CV assignments. No sampling or new loss weighting is applied by this census.</p>',
        '<p>Spatial uniformity is not the same as equal inclusion probability among all eligible '
        'ACRs. Choose the target population before sampling and save selection probabilities. '
        'Consider equal-ACR and pair-level evaluation summaries; do not change fitting weights silently.</p>',
        '<h2>Limits and saved files</h2><p>Complete means complete for this listed SNP panel, not a '
        'complete personal genome. Indels, structural variants and unlisted SNPs are absent. The '
        'export lacks populated per-call depth/quality fields. Call completeness can select particular '
        'genotypes and regions. REF checks do not validate ALT calls. Distinct genotype labels are '
        'not independent sequence observations when their local sequences are identical.</p>',
        '<p>Saved: per_acr_census.csv/parquet, per_genotype.csv, per_chromosome.csv, spatial_blocks.csv, '
        'coverage_diversity_grid.csv and summary.json. Per-chromosome call arrays, explicit sample maps, '
        'availability masks and local sequence-class IDs are in data/intermediate/genomewide_sequence_census. '
        'IDs are local to each ACR; −1 means unavailable. All chromosome receipts and source snapshots '
        'are retained. The original chromosome-10 availability masks are independently compared exactly.</p></html>']
    (output/'report.html').write_text('\n'.join(parts))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['census', 'report'], required=True)
    parser.add_argument('--chromosomes', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--output-dir', default='results/genomewide_sequence_census')
    parser.add_argument('--cache-dir', default='data/intermediate/genomewide_sequence_census')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, cache = root/args.output_dir, root/args.cache_dir
    if args.stage == 'census':
        run(root, output, cache, args.chromosomes)
    else:
        report(root, output, cache)


if __name__ == '__main__':
    main()
