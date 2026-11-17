#!/usr/bin/env python
"""Bounded sequence feasibility/timing audit; never runs prediction evaluation."""

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
from maize_regulatory.kmer_pilot import count_kmers, kmer_dictionary
from maize_regulatory.personalized_context_audit import (
    availability_summary, collect_context_variants, context_windows,
    interval_call_counts, reconstruct_patterns,
    width_sensitivity,
)
from maize_regulatory.quasi_motifs import (
    calibrated_word_definition, quasi_word_kernel, quasi_word_lme,
    read_pfm_catalog, word_probability_matrices,
)
from maize_regulatory.reference_sequence import digest


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(root, output):
    if (output/'summary.json').exists() or (output/'windows.csv').exists():
        raise ValueError('Preparation output exists; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    catalog_path = root/'data/raw/jaspar_plants_2026.txt'
    catalog = read_pfm_catalog(catalog_path)
    definition = calibrated_word_definition(catalog)
    definition['source_sha256'] = digest(catalog_path)
    save_json(output/'quasi_motif_definition.json', definition)
    matrices, words = word_probability_matrices(definition['preferred_probabilities'])
    np.save(output/'quasi_motif_probabilities.npy', matrices)
    kernel, names = quasi_word_kernel(definition['preferred_probabilities'])
    np.save(output/'quasi_motif_kernel.npy', kernel)
    save_json(output/'quasi_motif_columns.json', dict(all_words=words, canonical_words=names))
    with (output/'quasi_motifs.jaspar').open('w') as handle:
        for word, matrix in zip(words, matrices):
            handle.write(f'>WORD5_{word}\t{word}\n')
            for base, row in zip('ACGT', matrix):
                handle.write(base+' [ '+' '.join(f'{v:.12g}' for v in row)+' ]\n')
    print('Preferred probabilities:', definition['preferred_probabilities'], flush=True)

    previous = root/'results/sequence_cnn_benchmark'
    coords = pd.read_csv(previous/'acr_coordinates.csv', index_col='ACR')
    coverage = pd.read_csv(previous/'genotype_coverage.csv')
    development = development_genotypes()
    if coverage.genotype.tolist() != development.tolist():
        raise ValueError('Coverage/development identity order mismatch')
    genotypes = pd.Index(coverage.loc[coverage.eligible, 'genotype'], name='genotype')
    source = json.loads((root/'data/intermediate/sequence_cnn/receipt.json').read_text())['source']
    if source['genotype_order'] != genotypes.tolist():
        raise ValueError('Previous prediction cohort mismatch')
    chr_indices = np.flatnonzero(coords.chrom.eq(10))
    selected = coords.iloc[chr_indices].copy()
    sequences = pd.read_parquet(root/'data/maize_acr_sequences.parquet')
    if not sequences.index.equals(coords.index):
        raise ValueError('B73 sequence/global coordinate identity mismatch')
    from maize_regulatory.sequence_cnn import IndexedFasta
    fasta_path = root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa'
    fasta = IndexedFasta(fasta_path)
    try:
        reference = fasta.fetch('chr10', 0, fasta.entries['chr10'][0])
    finally:
        fasta.handle.close()
    windows = context_windows(selected, len(reference))
    for acr, row in windows.iterrows():
        if reference[int(row.start):int(row.end)].decode() != sequences.loc[acr, 'sequence']:
            raise ValueError(f'B73 core sequence mismatch: {acr}')
    windows['global_acr_index'] = chr_indices
    windows.to_csv(output/'windows.csv')
    pd.DataFrame({'genotype': genotypes}).to_csv(output/'genotypes.csv', index=False)
    vcf_path = root/'data/raw/panzea_v5/panzea_hapmap_282_chr10.vcf.gz'
    print(f'Auditing {len(windows):,} ACRs × {len(genotypes)} eligible genotypes', flush=True)
    variants, calls, mapping, scan = collect_context_variants(
        vcf_path, windows, genotypes, reference)
    variants.to_parquet(output/'context_variants.parquet', index=False)
    np.save(output/'context_dosage.npy', calls)
    mapping.to_csv(output/'vcf_sample_mapping.csv')

    # Independent source check against the previous ACR-only variant cache.
    old = pd.read_parquet(root/'data/intermediate/panzea_v5_pilot/chr10_variants.parquet')
    old_calls = pd.read_parquet(root/'data/intermediate/panzea_v5_pilot/chr10_alt_dosage.parquet')
    if not old_calls.index.equals(old.index) or not old_calls.columns.equals(development):
        raise ValueError('Old SNP cache identity mismatch')
    indices = np.searchsorted(variants.pos_0based, old.pos_0based)
    if (not np.array_equal(variants.pos_0based.to_numpy()[indices], old.pos_0based)
            or not np.array_equal(variants.REF.to_numpy()[indices], old.REF)
            or not np.array_equal(variants.ALT.to_numpy()[indices], old.ALT)
            or not np.array_equal(calls[indices], old_calls.loc[:, genotypes].to_numpy())):
        raise ValueError('New VCF extraction disagrees with old core cache')
    print(f'Validated all {len(old):,} old core SNPs and selected calls', flush=True)
    summary = dict(n_acrs=len(windows), n_genotypes=len(genotypes), context_width=2000,
                   scan=scan, old_core_snps_verified=len(old), old_core_calls_verified=int(len(old)*len(genotypes)),
                   context_clipped_acrs=int(windows.context_clipped.sum()),
                   scoring_scope='All listed biallelic SNPs, including rare/monomorphic; no MAF filter',
                   holdout_used=False, DTS_loaded=False, accuracy_evaluation=False,
                   cohort_changed=False, response_changed=False)
    per_acr = windows[['chrom', 'start', 'end', 'block']].copy()
    per_genotype = pd.DataFrame(index=genotypes)
    ids_by_kind = {}
    for kind, starts, ends in [
            ('core', windows.start, windows.end),
            ('context', windows.context_start, windows.context_end)]:
        support = interval_call_counts(variants.pos_0based, calls, starts, ends)
        patterns, ids, reference_ids, unambiguous, seconds = reconstruct_patterns(
            windows, reference, variants, calls, support, kind)
        patterns.to_parquet(output/f'{kind}_patterns.parquet', index=False)
        np.savez_compressed(output/f'{kind}_support.npz', **support, pattern_ids=ids,
                            reference_pattern_ids=reference_ids, unambiguous_DNA=unambiguous)
        summary[kind] = availability_summary(support, unambiguous, ids, patterns)
        summary[kind]['reconstruction_seconds'] = seconds
        summary[kind]['total_pattern_bases_including_B73'] = int(patterns.sequence.str.len().sum())
        ids_by_kind[kind] = ids
        per_acr[f'{kind}_SNPs'] = support['snps']
        per_acr[f'{kind}_available_genotypes'] = (ids >= 0).sum(axis=1)
        per_acr[f'{kind}_unique_personalized_sequences'] = [
            len(np.unique(row[row >= 0])) for row in ids]
        per_genotype[f'{kind}_available_ACRs'] = (ids >= 0).sum(axis=0)
        per_genotype[f'{kind}_missing_call_ACRs'] = (support['missing'] > 0).sum(axis=0)
        per_genotype[f'{kind}_heterozygous_call_ACRs'] = (support['heterozygous'] > 0).sum(axis=0)
        print(kind, json.dumps(summary[kind]), flush=True)
    common = (ids_by_kind['core'] >= 0) & (ids_by_kind['context'] >= 0)
    if np.any((ids_by_kind['context'] >= 0) & (ids_by_kind['core'] < 0)):
        raise AssertionError('Available context must imply available core')
    np.save(output/'common_pair_mask.npy', common)
    per_acr['common_available_genotypes'] = common.sum(axis=1)
    per_acr.to_csv(output/'per_acr_availability.csv')
    per_genotype.to_csv(output/'per_genotype_availability.csv')
    summary['common_pairs'] = int(common.sum())
    summary['total_preparation_seconds'] = time.monotonic()-started
    save_json(output/'summary.json', summary)
    paths = [vcf_path, catalog_path, fasta_path, Path(str(fasta_path)+'.fai'),
             previous/'acr_coordinates.csv', previous/'genotype_coverage.csv',
             root/'results/sequence_summary_comparison/fold_manifest.json',
             root/'src/maize_regulatory/resources/analysis_exclusions.json']
    save_json(output/'provenance.json', dict(
        inputs={str(p.relative_to(root)): digest(p) for p in paths},
        code={str(p.relative_to(root)): digest(p) for p in [Path(__file__).resolve(),
              root/'src/maize_regulatory/quasi_motifs.py',
              root/'src/maize_regulatory/personalized_context_audit.py']},
        genotype_order=genotypes.tolist(), reference_sequence_only=True,
        excluded_sample_below_existing_50_nucleus_threshold=coverage.loc[~coverage.eligible, 'genotype'].tolist(),
        supervised_transforms='None; all eligibility and encodings are sequence-only',
        NOTE='Personalized means SNP-substituted B73; unlisted variants, indels and SVs are absent.'))


def timing_sequences(root, output, kind, maximum=64):
    windows = pd.read_csv(output/'windows.csv', index_col='ACR')
    folds = json.loads((root/'results/sequence_summary_comparison/fold_manifest.json').read_text())
    training_acrs = np.flatnonzero(windows.block.isin(folds[0]['inner'][0]['training_blocks']))
    patterns = pd.read_parquet(output/f'{kind}_patterns.parquet')
    selected = patterns.loc[patterns.acr_index.isin(training_acrs)]
    rng = np.random.default_rng(42)
    return selected.iloc[rng.choice(len(selected), min(maximum, len(selected)), replace=False)]


def score_timing(root, output):
    # Legacy environment has Numba; the GPU environment intentionally stays separate.
    from maize_regulatory.scoring import MotifScorer, direct_motif_score
    catalog = read_pfm_catalog(root/'data/raw/jaspar_plants_2026.txt')
    pwms = {key: np.log2((.99*c/c.sum(axis=0, keepdims=True)+.0025)/.25)
            for key, c in catalog.items()}
    started = time.monotonic()
    scorer = MotifScorer(pwms)
    setup = time.monotonic()-started
    summary = json.loads((output/'summary.json').read_text())
    kernel = np.load(output/'quasi_motif_kernel.npy')
    mapping = kmer_dictionary(5)[3]
    rows = []
    for kind in ['core', 'context']:
        sample = timing_sequences(root, output, kind)
        seqs = sample.sequence.tolist()
        before = time.monotonic()
        warm = scorer.score_acr(seqs[0])
        warmup = time.monotonic()-before
        for length in sorted({p.shape[1] for p in pwms.values()}):
            mid = next(mid for mid, p in pwms.items() if p.shape[1] == length)
            expected = direct_motif_score(seqs[0], pwms[mid])
            np.testing.assert_allclose(warm[scorer.motif_ids.index(mid)], expected,
                                       atol=1e-5, rtol=1e-5)
        before = time.monotonic()
        for seq in seqs:
            scorer.score_acr(seq)
        tf_seconds = time.monotonic()-before
        before = time.monotonic()
        counts = np.array([np.bincount(mapping, weights=count_kmers(seq), minlength=512)
                           for seq in seqs])
        count_seconds = time.monotonic()-before
        # Warm BLAS separately; no accessibility values enter scoring.
        quasi_word_lme(counts[:2], kernel)
        repeats = 5
        before = time.monotonic()
        for _ in range(repeats):
            quasi_word_lme(counts, kernel)
        quasi_seconds = (time.monotonic()-before)/repeats
        n_patterns = summary[kind]['unique_patterns_including_B73']
        rows.append(dict(kind=kind, sampled_patterns=len(seqs), n_TF_motifs=len(pwms),
            TF_warmup_seconds=warmup, TF_measured_seconds=tf_seconds,
            TF_seconds_per_pattern=tf_seconds/len(seqs), count_measured_seconds=count_seconds,
            quasi_measured_seconds=quasi_seconds,
            extrapolated_TF_all_patterns_minutes=tf_seconds/len(seqs)*n_patterns/60,
            extrapolated_quasi_all_patterns_minutes=(count_seconds+quasi_seconds)/len(seqs)*n_patterns/60))
        print(rows[-1], flush=True)
    save_json(output/'scoring_timing.json', dict(setup_seconds=setup, rows=rows,
        TF_method='Unchanged legacy full-catalog Numba scanner; measured after compilation warmup',
        estimates='Linear extrapolations to cached distinct patterns, including B73; exclude I/O and future changes',
        validation='Independent direct PWM scan for one motif of every length, both sequence extents',
        sample_scope='Only frozen first inner-training ACR blocks; no accessibility loaded'))


def training_timing(root, output, batch_size=512, steps=20):
    import torch
    from maize_regulatory.sequence_cnn import EncodedSequences, SequenceCNN, reverse_complement, synchronize
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cpu':
        raise RuntimeError('GPU timing requires native Metal access; refusing misleading CPU extrapolation')
    windows = pd.read_csv(output/'windows.csv', index_col='ACR')
    folds = json.loads((root/'results/sequence_summary_comparison/fold_manifest.json').read_text())
    selected_blocks = folds[0]['inner'][0]['training_blocks']
    eligible_acrs = windows.block.isin(selected_blocks).to_numpy()
    common = np.load(output/'common_pair_mask.npy')
    train_a, train_g = np.where(common & eligible_acrs[:, None])
    if not len(train_a):
        raise ValueError('No common training pairs')
    # Read only training entries from the existing response/covariate caches.
    cache = root/'data/intermediate/sequence_cnn'
    y = np.load(cache/'response.npy', mmap_mode='r')
    z = np.load(cache/'covariates.npy', mmap_mode='r')
    pcs = np.load(cache/'pcs.npy', mmap_mode='r')
    global_rows = windows.global_acr_index.to_numpy()
    target = np.asarray(y[global_rows[train_a], train_g], dtype='float32')
    cov = np.column_stack([z[global_rows[train_a]], pcs[train_g]])
    scale = cov.std(axis=0); scale[scale < 1e-12] = 1
    cov = ((cov-cov.mean(axis=0))/scale).astype('float32')
    target = (target-target.mean())/target.std()
    rng = np.random.default_rng(42)
    # Real pairs, with replacement only for fixed-step runtime sampling. No fit is retained.
    sampled = rng.integers(len(train_a), size=(steps+5, batch_size))
    zz = torch.tensor(cov, device=device)
    yy = torch.tensor(target, device=device)
    result = []
    for kind in ['core', 'context']:
        support = np.load(output/f'{kind}_support.npz')
        pair_ids = support['pattern_ids'][train_a, train_g]
        used, inverse = np.unique(pair_ids, return_inverse=True)
        patterns = pd.read_parquet(output/f'{kind}_patterns.parquet').iloc[used]
        # Core timing uses the existing minimum 256-bp bucket for short cores.
        width = 2000 if kind == 'context' else int(max(256, np.ceil((patterns.sequence.str.len().max()+2)/128)*128))
        codes = np.full((len(patterns), width), 4, dtype=np.uint8)
        starts, ends = [], []
        lookup = np.full(256, 4, np.uint8)
        lookup[np.frombuffer(b'ACGT', np.uint8)] = np.arange(4)
        for i, row in enumerate(patterns.itertuples()):
            offset = 0 if kind == 'context' else (width-len(row.sequence))//2
            codes[i, offset:offset+len(row.sequence)] = lookup[np.frombuffer(row.sequence.encode(), np.uint8)]
            starts.append(offset+row.core_start); ends.append(offset+row.core_end)
        seq = EncodedSequences(codes, np.array(starts), np.array(ends), device)
        for network_width, depth in [(32, 2), (64, 3)]:
            torch.manual_seed(42)
            model = SequenceCNN(cov.shape[1], width=network_width, depth=depth).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            scaler = torch.amp.GradScaler('mps')
            model.train()
            started = None
            losses = []
            for step, rows in enumerate(sampled):
                if step == 5:
                    synchronize(device); started = time.monotonic()
                pid = inverse[rows]
                x = seq.batch(pid)
                if step % 2:
                    x = reverse_complement(x)
                at = torch.tensor(rows, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('mps', dtype=torch.float16):
                    pred = model(x, zz[at])
                    loss = (pred-yy[at]).square().mean()+1e-5*model.weight_penalty()
                losses.append(loss.detach())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer); scaler.update()
            synchronize(device)
            seconds = time.monotonic()-started
            if not torch.isfinite(torch.stack(losses)).all().item():
                raise FloatingPointError('Nonfinite timing loss')
            updates = int(next(iter(optimizer.state.values()))['step'].item())
            if updates != steps+5:
                raise FloatingPointError('Gradient scaler skipped timing updates')
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError('Nonfinite timing weights')
            # Inference throughput uses only training inputs; no predictions/metrics saved.
            model.eval(); synchronize(device); before = time.monotonic()
            with torch.no_grad(), torch.autocast('mps', dtype=torch.float16):
                for rows in sampled[5:]:
                    x = seq.batch(inverse[rows])
                    at = torch.tensor(rows, device=device)
                    value = (model(x, zz[at])+model(reverse_complement(x), zz[at]))/2
            synchronize(device)
            inference_seconds = time.monotonic()-before
            if not torch.isfinite(value).all().item():
                raise FloatingPointError('Nonfinite timing inference')
            result.append(dict(kind=kind, width=network_width, depth=depth,
                n_parameters=sum(p.numel() for p in model.parameters()), input_width=width,
                measured_steps=steps, batch_size=batch_size, measured_seconds=seconds,
                verified_optimizer_updates=updates,
                seconds_per_1000_pairs=seconds/(steps*batch_size)*1000,
                inference_seconds_per_1000_pairs=inference_seconds/(steps*batch_size)*1000,
                extrapolated_first_inner_training_epoch_seconds=seconds/(steps*batch_size)*len(train_a)))
            print(result[-1], flush=True)
            del model, optimizer, scaler; gc.collect(); torch.mps.empty_cache()
        del seq; gc.collect(); torch.mps.empty_cache()
    save_json(output/'training_timing.json', dict(device=device, torch_version=torch.__version__,
        rows=result, common_first_inner_training_pairs=len(train_a),
        common_first_inner_training_acrs=int(np.unique(train_a).size),
        role='Fixed-step runtime measurement only; no validation, early stopping, accuracy, or retained models',
        model='SequenceCNN plus global nuisance/ancestry inputs; real training-pair CPM; pair batches',
        estimates='Naive pair-wise evaluation, no duplicate-sequence gradient aggregation; '
                  'core uses one padded bucket (conservative); excludes validation, setup and I/O'))


def sensitivity(root, output):
    from maize_regulatory.sequence_cnn import IndexedFasta
    windows = pd.read_csv(output/'windows.csv', index_col='ACR')
    variants = pd.read_parquet(output/'context_variants.parquet')
    calls = np.load(output/'context_dosage.npy', mmap_mode='r')
    fasta = IndexedFasta(root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa')
    try:
        reference = fasta.fetch('chr10', 0, fasta.entries['chr10'][0])
    finally:
        fasta.handle.close()
    frame = width_sensitivity(windows, variants, calls, reference)
    expected = json.loads((output/'summary.json').read_text())['common_pairs']
    if int(frame.loc[frame.width_bp.eq(2000), 'available_pairs'].iloc[0]) != expected:
        raise AssertionError('Sensitivity endpoint differs from original context audit')
    frame.to_csv(output/'width_sensitivity.csv', index=False)
    print(frame.to_string(index=False), flush=True)


def report(output):
    summary = json.loads((output/'summary.json').read_text())
    definition = json.loads((output/'quasi_motif_definition.json').read_text())
    fields = ['total_pairs', 'complete_homozygous_pairs', 'available_pairs', 'available_fraction',
              'pairs_with_missing_calls', 'pairs_with_heterozygous_calls', 'pairs_with_both',
              'complete_pairs_lost_to_ambiguous_or_padded_DNA', 'acrs_with_any_available_pair',
              'acrs_with_ge_20_genotypes', 'acrs_with_ge_50_genotypes',
              'unique_patterns_including_B73', 'unique_personalized_patterns',
              'pairs_with_sequence_different_from_B73', 'reconstruction_seconds']
    table = pd.DataFrame({kind: {key: summary[kind][key] for key in fields}
                          for kind in ['core', 'context']})
    parts = ['<!doctype html><html><meta charset="utf-8"><title>Personalized chromosome-10 audit</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;line-height:1.5}'
             'table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #ccc}'
             'pre{white-space:pre-wrap}</style><h1>Personalized chromosome-10 sequence audit</h1>',
             '<p>Sequence availability and runtime only. No prediction accuracy or association tests.</p>',
             f'<p>{summary["n_acrs"]:,} ACRs × {summary["n_genotypes"]} existing eligible development '
             'genotypes; cortex CPM unchanged. Fixed holdout and DTS untouched.</p>',
             '<p><strong>Main finding:</strong> strict 2,000-bp personalization retains '
             f'{summary["context"]["available_pairs"]:,} pairs '
             f'({100*summary["context"]["available_fraction"]:.1f}% of the original pair grid), '
             'with a median of six usable genotypes per ACR. Computation is feasible; '
             'missing/heterozygous calls are the principal limitation. The full 927-motif '
             'catalog was timed without motif clustering. No full prediction benchmark has begun.</p>',
             '<h2>Availability criteria</h2><p>Existing prediction cohort → listed biallelic SNPs '
             'inside each sequence → every listed call homozygous and nonmissing → all sequence bases A/C/G/T. '
             'Context is 2,000 bp total. No MAF/minimum-SNP/accessibility screen. '
             'Missing and heterozygous failure counts overlap: subtract their intersection once. '
             'Padding and genomic N bases are reported separately. B73 is not used to fill missing calls.</p>',
             table.to_html(float_format=lambda x: f'{x:,.4f}'),
             '<p>“Unique personalized patterns” counts distinct sequences within each ACR among available '
             'genotypes; a B73 sequence can be one of them. The larger cache also retains a B73 pattern '
             'per unambiguous ACR. No genotype exclusion or common evaluation cohort is selected here.</p>',
             '<h2>Five-mer quasi-motifs</h2>',
             '<p>Exactly five-base JASPAR motifs are equally weighted. Normalize each count column, apply '
             'the existing 99% motif / 1% uniform smoothing, average the largest probability at each position, '
             'then pool positions 1/5 and 2/4. Assign that probability to each word’s preferred nucleotide '
             'and divide the remainder equally among the other three. Do not smooth again.</p>',
             '<pre>'+html.escape(json.dumps({k: definition[k] for k in [
                 'n_source_motifs', 'original_position_means', 'preferred_probabilities',
                 'other_base_probabilities']}, indent=2))+'</pre>',
             '<p>All 1,024 profiles are saved; 512 reverse-complement-equivalent columns suffice for '
             'strand-averaged log2(mean likelihood ratio) scores. This is a new versioned definition; '
             'older five-mer results are preserved.</p>']
    for name in ['scoring_timing', 'training_timing']:
        path = output/f'{name}.json'
        if path.exists():
            data = json.loads(path.read_text())
            parts += ['<h2>'+name.replace('_', ' ').title()+'</h2>',
                      pd.DataFrame(data.pop('rows')).to_html(index=False, float_format=lambda x: f'{x:,.4f}'),
                      '<pre>'+html.escape(json.dumps(data, indent=2))+'</pre>']
    if (output/'width_sensitivity.csv').exists():
        parts += ['<h2>Width sensitivity: availability only</h2>',
                  '<p>No width is selected here. A core longer than the proposed window is unavailable '
                  'at that width; cores are never truncated. These windows remain inside the audited '
                  '2,000-bp interval and use identical missingness/heterozygosity rules.</p>',
                  pd.read_csv(output/'width_sensitivity.csv').to_html(index=False)]
    parts += ['<h2>Limits</h2><p>These are SNP-substituted B73 sequences, not assembled personal genomes. '
              'The VCF lacks populated per-call quality/depth fields. REF checks do not validate ALT calls. '
              'Timing extrapolations are planning estimates, not completed fits or performance claims. '
              'All timing samples come from the first frozen inner-training partition.</p></html>']
    (output/'report.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'scoring', 'training', 'sensitivity', 'report'], required=True)
    parser.add_argument('--output-dir', default='results/personalized_chr10_audit')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--timing-steps', type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/args.output_dir
    # Freeze actual stage code without changing the preparation receipt or older results.
    snapshot = output/'source_snapshot'/args.stage
    for relative in ['scripts/audit_personalized_chr10.py',
                     'src/maize_regulatory/quasi_motifs.py',
                     'src/maize_regulatory/personalized_context_audit.py',
                     'src/maize_regulatory/sequence_cnn.py',
                     'src/maize_regulatory/scoring.py']:
        destination = snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Existing preparation snapshot is the version that actually generated those outputs.
        if args.stage == 'prepare' and destination.exists():
            continue
        shutil.copyfile(root/relative, destination)
    if args.stage == 'prepare':
        prepare(root, output)
    elif args.stage == 'scoring':
        score_timing(root, output)
    elif args.stage == 'training':
        training_timing(root, output, args.batch_size, args.timing_steps)
    elif args.stage == 'sensitivity':
        sensitivity(root, output)
    else:
        report(output)


if __name__ == '__main__':
    main()
