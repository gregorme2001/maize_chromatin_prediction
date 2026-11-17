#!/usr/bin/env python
"""Freeze the agreed ACR sample and prepare matched sequence features, without fitting."""

import argparse
import gc
import hashlib
import html
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd

from maize_regulatory.cohort import development_genotypes
from maize_regulatory.personalized_context_audit import reconstruct_patterns
from maize_regulatory.quasi_motifs import quasi_word_kernel, quasi_word_lme, read_pfm_catalog
from maize_regulatory.reference_sequence import digest
from maize_regulatory.sampled_sequence_features import (
    canonical_word_counts, pool_log_mean_lr, spatial_sample,
)

CODE = ['scripts/prepare_sampled_sequence_features.py',
        'src/maize_regulatory/sampled_sequence_features.py',
        'src/maize_regulatory/personalized_context_audit.py',
        'src/maize_regulatory/quasi_motifs.py', 'src/maize_regulatory/scoring.py',
        'src/maize_regulatory/sequence_cnn.py']


def save_json(path, value):
    temporary = Path(str(path)+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def input_state(root):
    census = root/'results/genomewide_sequence_census'
    paths = [census/'per_acr_census.parquet', census/'genotypes.csv', census/'run_config.json',
             root/'results/sequence_summary_comparison/fold_manifest.json',
             root/'results/personalized_chr10_audit/quasi_motif_definition.json',
             root/'results/personalized_chr10_audit/quasi_motif_kernel.npy',
             root/'data/raw/jaspar_plants_2026.txt', root/'data/maize_acr_motif_scores.parquet',
             root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa',
             root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa.fai']
    for chromosome in range(1, 11):
        directory = root/f'data/intermediate/genomewide_sequence_census/chr{chromosome}'
        paths.extend(directory/name for name in ['receipt.json', 'per_acr.parquet',
            'context_variants.parquet', 'context_dosage.npy', 'pair_support.npz', 'vcf_sample_mapping.csv'])
    return dict(parameters=dict(seed=42, minimum_context_complete_genotypes=20,
                                quota_per_5Mb_block=25, total_context_width=2000),
        inputs={str(p.relative_to(root)): digest(p) for p in paths},
        code={relative: digest(root/relative) for relative in CODE},
        no_DTS_or_accessibility_values_read=True, holdout_analyzed=False,
        supervised_fitting=False, CV_changed=False, fitting_weights_changed=False)


def check_state(root, output):
    actual = input_state(root)
    if actual != json.loads((output/'input_state.json').read_text()):
        raise ValueError('Frozen input/code state changed; use a new output/cache for intentional changes')
    return actual


def sample(root, output, cache):
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    state = input_state(root)
    state_path = output/'input_state.json'
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError('Existing frozen state differs')
    if (output/'sample_receipt.json').exists():
        for name, expected in json.loads((output/'sample_receipt.json').read_text())['outputs'].items():
            if digest(root/name) != expected:
                raise ValueError('Existing frozen sample output changed')
        print('Keeping verified frozen sample', flush=True)
        return
    save_json(state_path, state)
    for relative in CODE:
        target = output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/relative, target)
    census_dir = root/'results/genomewide_sequence_census'
    census = pd.read_parquet(census_dir/'per_acr_census.parquet')
    chosen = spatial_sample(census)
    chosen['core_variation_rich'] = chosen.common_core_second_sequence_n >= 5
    chosen['context_variation_rich'] = chosen.context_second_sequence_n >= 5
    genotypes = pd.Index(pd.read_csv(census_dir/'genotypes.csv').genotype, name='genotype')
    coverage = pd.read_csv(root/'results/sequence_cnn_benchmark/genotype_coverage.csv')
    if (coverage.genotype.tolist() != development_genotypes().tolist()
            or coverage.loc[coverage.eligible, 'genotype'].tolist() != genotypes.tolist()):
        raise ValueError('Genotype identity/order differs from frozen prediction cohort')
    folds_path = root/'results/sequence_summary_comparison/fold_manifest.json'
    folds = json.loads(folds_path.read_text())
    block_folds = {}
    for spec in folds:
        if set(spec['training_blocks']) & set(spec['validation_blocks']):
            raise ValueError('Outer training/validation overlap')
        for block in spec['validation_blocks']:
            if block in block_folds:
                raise ValueError('Block belongs to more than one outer validation fold')
            block_folds[block] = spec['number']
    chosen['outer_fold'] = [block_folds[int(b)] for b in chosen.block]
    mask = np.zeros((len(chosen), len(genotypes)), bool)
    for chromosome, part in chosen.groupby('chrom', sort=True):
        directory = root/f'data/intermediate/genomewide_sequence_census/chr{chromosome}'
        parent = pd.read_parquet(directory/'per_acr.parquet')
        at = parent.index.get_indexer(part.index)
        if np.any(at < 0) or not parent.index[at].equals(part.index):
            raise ValueError('Sample/census ACR alignment failed')
        with np.load(directory/'pair_support.npz') as support:
            selected_mask = support['common_available'][at]
            if not np.array_equal(selected_mask, support['context_available'][at]):
                raise ValueError('Common/context mask mismatch')
            mask[part.sample_acr_index.to_numpy()] = selected_mask
    if not np.array_equal(mask.sum(axis=1), chosen.context_n_available):
        raise ValueError('Sample mask counts differ from census')
    a, g = np.where(mask)
    pairs = pd.DataFrame(dict(pair_index=np.arange(len(a)), sample_acr_index=a.astype('int32'),
        genotype_index=g.astype('int16'), ACR=chosen.index.to_numpy()[a],
        genotype=genotypes.to_numpy()[g], outer_fold=chosen.outer_fold.to_numpy()[a]))
    chosen.to_parquet(output/'selected_acrs.parquet')
    chosen.to_csv(output/'selected_acrs.csv')
    pairs.to_parquet(output/'pair_manifest.parquet', index=False)
    pd.DataFrame({'genotype': genotypes}).to_csv(output/'genotypes.csv', index=False)
    np.save(cache/'common_pair_mask.npy', mask)
    shutil.copyfile(folds_path, output/'fold_manifest.json')
    summary = dict(n_ACRs=len(chosen), n_genotypes=len(genotypes), n_pairs=len(pairs),
        blocks=int(chosen.block.nunique()), chromosomes=int(chosen.chrom.nunique()),
        core_variation_rich_ACRs=int(chosen.core_variation_rich.sum()),
        context_variation_rich_ACRs=int(chosen.context_variation_rich.sum()),
        ACRs_with_no_listed_context_SNPs=int((chosen.context_n_SNPs == 0).sum()),
        numpy_version_used_for_draw=np.__version__, platform=platform.platform(),
        sample_is_frozen=True, no_response_or_prediction_values_used=True)
    files = [output/name for name in ['selected_acrs.parquet', 'selected_acrs.csv',
        'pair_manifest.parquet', 'genotypes.csv', 'fold_manifest.json']]+[cache/'common_pair_mask.npy']
    save_json(output/'sample_receipt.json', dict(summary=summary,
        outputs={str(p.relative_to(root)): digest(p) for p in files}))
    print(json.dumps(summary, indent=2), flush=True)


def sequences(root, output, cache):
    from maize_regulatory.sequence_cnn import IndexedFasta
    check_state(root, output)
    if (output/'sequence_receipt.json').exists():
        receipt = json.loads((output/'sequence_receipt.json').read_text())
        if any(digest(root/name) != expected for name, expected in receipt['outputs'].items()):
            raise ValueError('Sequence cache changed')
        print('Keeping verified sequence cache', flush=True); return
    chosen = pd.read_parquet(output/'selected_acrs.parquet')
    genotypes = pd.Index(pd.read_csv(output/'genotypes.csv').genotype, name='genotype')
    common = np.load(cache/'common_pair_mask.npy')
    ids = {kind: np.full(common.shape, -1, np.int32) for kind in ['core', 'context']}
    ref_ids = {kind: np.full(len(chosen), -1, np.int32) for kind in ids}
    records = {kind: [] for kind in ids}; totals = {kind: 0 for kind in ids}
    started = time.monotonic()
    for chromosome, part in chosen.groupby('chrom', sort=True):
        print(f'Reconstructing selected chromosome {chromosome} sequences', flush=True)
        directory = root/f'data/intermediate/genomewide_sequence_census/chr{chromosome}'
        variants = pd.read_parquet(directory/'context_variants.parquet')
        calls = np.load(directory/'context_dosage.npy', mmap_mode='r')
        mapping = pd.read_csv(directory/'vcf_sample_mapping.csv', index_col='genotype')
        if mapping.index.tolist() != genotypes.tolist() or calls.shape != (len(variants), len(genotypes)):
            raise ValueError('Cached variant/genotype identity mismatch')
        fasta = IndexedFasta(root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa')
        try:
            reference = fasta.fetch(f'chr{chromosome}', 0, fasta.entries[f'chr{chromosome}'][0])
        finally:
            fasta.handle.close()
        positions = variants.pos_0based.to_numpy()
        if np.any(np.diff(positions) <= 0):
            raise ValueError('Cached SNP positions are not unique/sorted')
        reference_bases = np.frombuffer(reference, dtype='S1')[positions]
        if not np.array_equal(reference_bases, variants.REF.to_numpy(dtype='S1')):
            raise ValueError('Cached SNP REF disagrees with B73')
        local_mask = common[part.sample_acr_index.to_numpy()]
        for i, row in enumerate(part.itertuples()):
            first, last = np.searchsorted(positions, [row.context_start, row.context_end])
            if not np.isin(calls[first:last, np.flatnonzero(local_mask[i])], [0, 2]).all():
                raise ValueError('Selected pair has missing or heterozygous calls')
        for kind in ids:
            patterns, local_ids, local_refs, unambiguous, _ = reconstruct_patterns(
                part, reference, variants, calls, {'complete_homozygous': local_mask}, kind)
            if not unambiguous.all() or not np.array_equal(local_ids >= 0, local_mask):
                raise ValueError('Reconstruction changed pair availability')
            expected_classes = part.common_core_n_sequences if kind == 'core' else part.context_n_sequences
            counts = np.array([len(np.unique(row[row >= 0])) for row in local_ids])
            if not np.array_equal(counts, expected_classes):
                raise ValueError('Reconstructed sequence classes differ from census')
            offset = totals[kind]
            ids[kind][part.sample_acr_index.to_numpy()] = np.where(local_ids >= 0, local_ids+offset, -1)
            ref_ids[kind][part.sample_acr_index.to_numpy()] = local_refs+offset
            patterns['pattern_id'] += offset
            patterns['sample_acr_index'] = part.sample_acr_index.to_numpy()[patterns.acr_index]
            patterns = patterns.drop(columns='acr_index')
            patterns['GC'] = patterns.sequence.map(lambda s: (s.count('G')+s.count('C'))/len(s))
            records[kind].append(patterns); totals[kind] += len(patterns)
        del reference, calls; gc.collect()
    patterns = {kind: pd.concat(records[kind], ignore_index=True) for kind in ids}
    for kind in ids:
        if not np.array_equal(patterns[kind].pattern_id, np.arange(len(patterns[kind]))):
            raise ValueError('Global pattern IDs are not explicit row positions')
        patterns[kind].to_parquet(cache/f'{kind}_patterns.parquet', index=False)
        np.save(cache/f'{kind}_pattern_ids.npy', ids[kind])
        np.save(cache/f'{kind}_B73_pattern_ids.npy', ref_ids[kind])
    core_lookup = {(row.ACR, row.sequence): row.pattern_id for row in patterns['core'].itertuples()}
    context_to_core = np.array([core_lookup[(row.ACR, row.sequence[row.core_start:row.core_end])]
                               for row in patterns['context'].itertuples()], dtype=np.int32)
    if not np.array_equal(context_to_core[ids['context'][common]], ids['core'][common]):
        raise ValueError('Core/context sequences disagree for a matched genotype pair')
    np.save(cache/'context_to_core_pattern.npy', context_to_core)
    pairs = pd.read_parquet(output/'pair_manifest.parquet')
    a, g = pairs.sample_acr_index.to_numpy(), pairs.genotype_index.to_numpy()
    for kind in ids:
        pairs[f'{kind}_personalized_pattern_id'] = ids[kind][a, g]
        pairs[f'{kind}_B73_pattern_id'] = ref_ids[kind][a]
    pairs.to_parquet(output/'feature_lookup.parquet', index=False)
    files = list(cache.glob('*patterns.parquet'))+list(cache.glob('*pattern_ids.npy'))+[
        cache/'context_to_core_pattern.npy', output/'feature_lookup.parquet']
    receipt = dict(seconds=time.monotonic()-started,
        core_patterns_including_B73=totals['core'], context_patterns_including_B73=totals['context'],
        all_matched_pair_core_context_sequences_agree=True,
        all_selected_calls_homozygous=True, all_cached_REF_match_B73=True,
        sequence_class_counts_match_census=True,
        outputs={str(p.relative_to(root)): digest(p) for p in files})
    save_json(output/'sequence_receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'outputs'}, indent=2), flush=True)


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def score(root, output, cache, chunk_size=256):
    from maize_regulatory.scoring import MotifScorer
    check_state(root, output)
    sequence_receipt = json.loads((output/'sequence_receipt.json').read_text())
    if any(digest(root/name) != expected for name, expected in sequence_receipt['outputs'].items()):
        raise ValueError('Sequence cache digest mismatch')
    catalog = read_pfm_catalog(root/'data/raw/jaspar_plants_2026.txt')
    pwms = {key: np.log2((.99*c/c.sum(axis=0, keepdims=True)+.0025)/.25) for key, c in catalog.items()}
    scorer = MotifScorer(pwms)
    definition_path = root/'results/personalized_chr10_audit/quasi_motif_definition.json'
    definition = json.loads(definition_path.read_text())
    kernel, words = quasi_word_kernel(definition['preferred_probabilities'])
    np.testing.assert_array_equal(kernel, np.load(root/'results/personalized_chr10_audit/quasi_motif_kernel.npy'))
    lengths = np.array([pwms[mid].shape[1] for mid in scorer.motif_ids])
    features = cache/'features'; features.mkdir(exist_ok=True)
    chunks = output/'score_chunks'; chunks.mkdir(exist_ok=True)
    columns = dict(TF_motif_ids=scorer.motif_ids, canonical_five_mers=words,
        conceptual_five_mer_columns=1024, stored_five_mer_columns=512,
        five_mer_duplicate_multiplicity=2,
        future_ridge_note='For equivalence to 1024 standardized word columns, word coefficients '
                          'use half the TF coefficient penalty; no fitting is done here.',
        definition=definition, summary='log2(mean likelihood ratio over valid positions and both strands)',
        flank_rule='Pool placements within left/right fragments; exclude sites overlapping the core; '
                   'do not concatenate fragments. Whole context includes boundary-crossing sites.')
    save_json(output/'feature_columns.json', columns)
    config = dict(chunk_size=chunk_size, input_state_sha256=digest(output/'input_state.json'),
                  sequence_receipt_sha256=digest(output/'sequence_receipt.json'),
                  numpy_version=np.__version__, platform=platform.platform())
    if (output/'score_config.json').exists() and json.loads((output/'score_config.json').read_text()) != config:
        raise ValueError('Score resume config/environment changed')
    save_json(output/'score_config.json', config)
    timings = []
    for kind in ['core', 'context', 'flanks']:
        patterns = pd.read_parquet(cache/f'{"core" if kind == "core" else "context"}_patterns.parquet')
        if not np.array_equal(patterns.pattern_id, np.arange(len(patterns))):
            raise ValueError('Pattern identity/order mismatch')
        arrays = {}
        for name, width in [('TF', len(scorer.motif_ids)), ('word', 512)]:
            path = features/f'{kind}_{name}.npy'
            if path.exists():
                value = np.load(path, mmap_mode='r+')
                if value.shape != (len(patterns), width) or value.dtype != np.float32:
                    raise ValueError('Existing feature shape/dtype mismatch')
            else:
                value = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(patterns), width))
                value[:] = np.nan; value.flush()
            arrays[name] = value
        started = time.monotonic()
        for first in range(0, len(patterns), chunk_size):
            last = min(first+chunk_size, len(patterns)); receipt_path = chunks/f'{kind}_{first:07d}.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if any(array_digest(a[first:last]) != receipt['sha256'][name] for name, a in arrays.items()):
                    raise ValueError('Completed score chunk digest mismatch')
                continue
            counts = []
            for j in range(first, last):
                row = patterns.iloc[j]
                fragments = ([row.sequence[:row.core_start], row.sequence[row.core_end:]]
                             if kind == 'flanks' else [row.sequence])
                if min(map(len, fragments)) < max(8, scorer.max_length):
                    raise ValueError('Fragment shorter than scanner requirement')
                values = np.stack([scorer.score_acr(fragment) for fragment in fragments])
                arrays['TF'][j] = (values[0] if len(fragments) == 1 else
                                  pool_log_mean_lr(values, list(map(len, fragments)), lengths))
                counts.append(canonical_word_counts(fragments))
            arrays['word'][first:last] = quasi_word_lme(np.array(counts), kernel).astype('float32')
            for value in arrays.values():
                if not np.isfinite(value[first:last]).all():
                    raise FloatingPointError('Nonfinite sequence scores')
                value.flush()
            save_json(receipt_path, dict(first=first, last=last,
                sha256={name: array_digest(a[first:last]) for name, a in arrays.items()}))
            if first % (chunk_size*4) == 0 or last == len(patterns):
                print(f'{kind}: {last:,}/{len(patterns):,} distinct patterns; '
                      f'{time.monotonic()-started:.1f}s', flush=True)
        timings.append(dict(kind=kind, n_patterns=len(patterns), seconds_this_invocation=time.monotonic()-started,
            minimum_TF_score=float(arrays['TF'].min()), maximum_TF_score=float(arrays['TF'].max()),
            feature_files={name: dict(shape=list(a.shape), sha256=digest(features/f'{kind}_{name}.npy'))
                           for name, a in arrays.items()}))
        del arrays, patterns; gc.collect()
    save_json(output/'scoring_receipt.json', dict(complete=True, timings=timings,
        model_fitted=False, source_score_definition_unchanged=True,
        no_threshold_or_score_clipping=True, output_dtype='float32'))


def verify(root, output, cache):
    from maize_regulatory.scoring import direct_motif_score
    check_state(root, output)
    chosen = pd.read_parquet(output/'selected_acrs.parquet')
    columns = json.loads((output/'feature_columns.json').read_text())
    existing = pd.read_parquet(root/'data/maize_acr_motif_scores.parquet',
        filters=[('ACR', 'in', chosen.index.tolist())])
    at = existing.index.get_indexer(chosen.index)
    if np.any(at < 0) or not existing.index[at].equals(chosen.index):
        raise ValueError('Stored B73 motif-score ACR identity mismatch')
    col = existing.columns.get_indexer(columns['TF_motif_ids'])
    if np.any(col < 0) or not existing.columns[col].tolist() == columns['TF_motif_ids']:
        raise ValueError('Stored B73 motif-score TF identity mismatch')
    old = existing.iloc[at, col].to_numpy()
    core_tf = np.load(cache/'features/core_TF.npy', mmap_mode='r')
    refs = np.load(cache/'core_B73_pattern_ids.npy')
    difference = float(np.max(np.abs(core_tf[refs]-old)))
    np.testing.assert_allclose(core_tf[refs], old, atol=1e-5, rtol=1e-5)
    catalog = read_pfm_catalog(root/'data/raw/jaspar_plants_2026.txt')
    pwms = {mid: np.log2((.99*c/c.sum(axis=0, keepdims=True)+.0025)/.25) for mid, c in catalog.items()}
    # Every motif length is checked against a direct position-by-position scan.
    picks = [next(mid for mid, pwm in pwms.items() if pwm.shape[1] == length)
             for length in sorted({pwm.shape[1] for pwm in pwms.values()})]
    rng = np.random.default_rng(71); n_checks = 0
    for kind in ['core', 'context', 'flanks']:
        patterns = pd.read_parquet(cache/f'{"core" if kind == "core" else "context"}_patterns.parquet')
        saved = np.load(cache/f'features/{kind}_TF.npy', mmap_mode='r')
        personal = np.flatnonzero(~patterns.is_b73.to_numpy())
        rows = rng.choice(personal, min(4, len(personal)), replace=False)
        for i in rows:
            row = patterns.iloc[i]
            fragments = ([row.sequence[:row.core_start], row.sequence[row.core_end:]]
                         if kind == 'flanks' else [row.sequence])
            for mid in picks:
                length = pwms[mid].shape[1]
                weights = np.array([len(fragment)-length+1 for fragment in fragments])
                scores = np.array([direct_motif_score(fragment, pwms[mid]) for fragment in fragments])
                expected = np.log2(np.sum(weights*np.exp2(scores))/weights.sum())
                np.testing.assert_allclose(saved[i, columns['TF_motif_ids'].index(mid)], expected,
                                           atol=1e-5, rtol=1e-5)
                n_checks += 1
    receipt = json.loads((output/'scoring_receipt.json').read_text())
    for item in receipt['timings']:
        for name, info in item['feature_files'].items():
            path = cache/f'features/{item["kind"]}_{name}.npy'
            if digest(path) != info['sha256'] or not np.isfinite(np.load(path, mmap_mode='r')).all():
                raise ValueError('Completed feature file integrity mismatch')
    validation = dict(all_selected_B73_core_TF_scores_match_existing_parquet=True,
        n_B73_values_compared=int(old.size), max_absolute_B73_difference=difference,
        direct_personalized_TF_checks=n_checks, all_motif_lengths_checked=True,
        all_feature_hashes_and_finiteness_verified=True, holdout_or_response_read=False)
    save_json(output/'validation.json', validation)
    print(json.dumps(validation, indent=2), flush=True)


def report(output):
    sample_info = json.loads((output/'sample_receipt.json').read_text())['summary']
    sequence_info = json.loads((output/'sequence_receipt.json').read_text())
    scoring = json.loads((output/'scoring_receipt.json').read_text())
    validation = json.loads((output/'validation.json').read_text())
    chosen = pd.read_parquet(output/'selected_acrs.parquet')
    by_chrom = chosen.groupby('chrom').agg(ACRs=('block', 'size'), blocks=('block', 'nunique'),
        matched_pairs=('context_n_available', 'sum'), core_variation_rich_ACRs=('core_variation_rich', 'sum'))
    by_chrom.to_csv(output/'sample_by_chromosome.csv')
    parts = ['<!doctype html><html><meta charset="utf-8"><title>Frozen sequence feature dataset</title>',
        '<style>body{font:16px system-ui;max-width:1050px;margin:36px auto;line-height:1.5}'
        'td,th{border:1px solid #ccc;padding:6px 12px}table{border-collapse:collapse}'
        'pre{white-space:pre-wrap}</style><h1>Frozen spatial sample and sequence features</h1>',
        f'<p><strong>{sample_info["n_ACRs"]:,} ACRs; {sample_info["n_pairs"]:,} matched ACR–genotype pairs; '
        f'{sample_info["blocks"]} genomic blocks; {sample_info["n_genotypes"]} eligible development genotypes.</strong></p>',
        '<p>Seed 42; at least 20 usable genotypes; up to 25 ACRs sampled uniformly within each 5-Mb block. '
        'Keep all available genotypes for each selected ACR. No accessibility outcomes, predictions or '
        'DTS were used. The frozen holdout remains untouched. No model was fitted.</p>', by_chrom.to_html(),
        '<p>Inclusion probabilities are saved. Spatial quotas do not imply uniform inclusion over all '
        'ACRs or remove genotype-dependent missingness. No fitting weights were changed.</p>',
        '<h2>What was scored?</h2><p>Every distinct core/context sequence within each selected ACR, '
        'including its B73 control, has 927 TF motif scores and 512 distinct five-mer quasi-motif scores '
        '(equivalent to 1,024 conceptual reverse-complement-paired columns). All summaries are '
        'log2(mean likelihood ratio) over valid placements and both strands; negative scores are retained.</p>',
        '<p>Core, full 2,000-bp context, and flank-only summaries are saved. Flank-only scores pool '
        'valid placements in the left and right fragments without joining them or including matches '
        'crossing the measured-core boundary. Full-context scores include all genuine windows. '
        'Separate core/flank predictors are available for the planned context-enabled ridge models.</p>',
        '<p>Scores are stored once per sequence pattern. feature_lookup.parquet maps every matched '
        'pair to its personalized and B73 feature rows; context IDs also index flank scores. '
        'The sequence cache retains core masks and DNA for future CNN training.</p>',
        '<h2>Validation</h2><pre>'+html.escape(json.dumps(validation, indent=2))+'</pre>',
        '<h2>Counts and runtime</h2><pre>'+html.escape(json.dumps(dict(sample=sample_info,
            sequences={k:v for k,v in sequence_info.items() if k != 'outputs'},
            scoring=scoring), indent=2))+'</pre>',
        '<h2>Limitations</h2><p>These are SNP-substituted B73 sequences, not complete personal genomes. '
        'Missing and heterozygous calls were not imputed; unlisted variants and indels/SVs are absent. '
        'Regions with no listed SNPs remain in the agreed sample and are identified explicitly. '
        'The variation-rich indicators are evaluation annotations, not extra sampling filters.</p></html>']
    (output/'report.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['sample', 'sequences', 'score', 'verify', 'report'], required=True)
    parser.add_argument('--output-dir', default='results/sampled_sequence_features')
    parser.add_argument('--cache-dir', default='data/intermediate/sampled_sequence_features')
    parser.add_argument('--chunk-size', type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, cache = root/args.output_dir, root/args.cache_dir
    if args.stage == 'report':
        report(output)
    else:
        dict(sample=sample, sequences=sequences, score=score, verify=verify)[args.stage](
            root, output, cache, **({'chunk_size': args.chunk_size} if args.stage == 'score' else {}))


if __name__ == '__main__':
    main()
