"""Matched B73-only sequence prediction across genomic regions, cortex only."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import sparse

from .bulk_motif_association import gc_spline_basis
from .cohort import development_genotypes, read_development_parquet
from .kmer_pilot import count_kmers, kmer_dictionary
from .pooled_sequence import Moments
from .spatial import parse_acr_coordinates


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cortex_cache(root, directory):
    """Pool development cortex cells once; verify genome totals and previous counts.

    Source RDS is sparse; only selected development cortex rows enter arithmetic.
    A cache created by an interrupted/unreceipted run is verified before adoption.
    """
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = root/'data/intermediate/cells'
    audit = root/'results/cell_state_accessibility_audit/aggregation'
    genotypes = development_genotypes()
    metadata = pd.read_csv(source/'cell_metadata.csv')
    acrs = pd.read_csv(source/'acr_index.csv')
    if (not metadata.cellID.is_unique or not acrs.ACR.is_unique
            or not np.array_equal(metadata.matrix_row, np.arange(1, len(metadata)+1))
            or not np.array_equal(acrs.matrix_column, np.arange(1, len(acrs)+1))):
        raise ValueError('Source cell/ACR identities are not explicit and unique')
    keep = metadata.normalized_genotype.isin(genotypes) & metadata.celltype_full.eq('cortex')
    selected = metadata.loc[keep, ['matrix_row', 'normalized_genotype']].copy()
    selected['genotype_index'] = genotypes.get_indexer(selected.normalized_genotype)
    expected_n = np.bincount(selected.genotype_index, minlength=len(genotypes))
    selected_path = directory/'selected_cells.csv'
    if selected_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(selected_path), selected.reset_index(drop=True))
    else:
        selected.to_csv(selected_path, index=False)
    genotype_path = directory/'genotypes.csv'
    if genotype_path.exists() and pd.read_csv(genotype_path).genotype.tolist() != genotypes.tolist():
        raise ValueError('Cached genotype order changed')
    pd.DataFrame({'genotype': genotypes}).to_csv(genotype_path, index=False)
    paths = [source/p for p in ['cell_by_acr.sparse.rds', 'cell_metadata.csv', 'acr_index.csv']]
    paths += [root/'scripts/pool_reference_cortex.R', audit/'genotype_cell_state_library_depth.csv']
    inputs = {str(p.relative_to(root)): digest(p) for p in paths}
    receipt_path = directory/'receipt.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous['inputs'] != inputs or previous['genotypes'] != genotypes.tolist():
            raise ValueError('Cortex cache inputs changed; use a new cache directory')
    binary = directory/'counts.int32'
    adopted = binary.exists() and not receipt_path.exists()
    if not binary.exists():
        with (directory/'pooling.log').open('w') as log:
            subprocess.run(['Rscript', str(root/'scripts/pool_reference_cortex.R'),
                str(source/'cell_by_acr.sparse.rds'), str(source/'cell_metadata.csv'),
                str(selected_path), str(source/'acr_index.csv'), str(binary), str(len(genotypes))],
                check=True, stdout=log, stderr=log)
    if binary.stat().st_size != len(acrs)*len(genotypes)*4:
        raise ValueError('Pooled binary size does not match explicit ACR/genotype axes')
    y = np.memmap(binary, dtype='<i4', mode='r', shape=(len(acrs), len(genotypes)))
    if np.any(y < 0) or np.any(y > expected_n[None, :]):
        raise ValueError('Detection counts outside nucleus-count bounds')
    depth = pd.read_csv(audit/'genotype_cell_state_library_depth.csv')
    cortex = depth.loc[depth.cell_state.eq('cortex')]
    g = genotypes.get_indexer(cortex.genotype)
    if np.any(g < 0):
        raise ValueError('Non-development genotype in cached cortex depths')
    t = np.bincount(g, weights=cortex.total_detected_events, minlength=len(genotypes)).astype(np.int64)
    n = np.bincount(g, weights=cortex.n_nuclei, minlength=len(genotypes)).astype(np.int64)
    if not np.array_equal(n, expected_n) or not np.array_equal(y.sum(axis=0), t):
        raise ValueError('Whole-genome pooled events/nuclei differ from previous sparse audit')
    old_groups = pd.read_csv(audit/'genotype_cell_state_nuclei.csv')
    old_cortex = old_groups.cell_state.eq('cortex')
    if old_groups.loc[old_cortex, 'genotype'].tolist() != genotypes.tolist():
        raise ValueError('Previous pooled genotype order differs')
    chr10 = pd.read_csv(audit/'chr10_acrs.csv')
    positions = pd.Index(acrs.ACR).get_indexer(chr10.ACR)
    previous_y = sparse.load_npz(audit/'chr10_pooled_counts.npz')[np.flatnonzero(old_cortex)].toarray().T
    if np.any(positions < 0) or not np.array_equal(y[positions], previous_y):
        raise ValueError('Chromosome-10 counts differ from previous independent aggregation')
    sample = pd.read_csv(audit/'library_qc_sample_acrs.csv')
    sample_positions = pd.Index(acrs.ACR).get_indexer(sample.ACR)
    library = sparse.load_npz(audit/'library_qc_sample_counts.npz')
    lib_positions = cortex.library_group.to_numpy()
    indicator = sparse.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))),
                                  shape=(len(genotypes), len(g)))
    sample_y = (indicator@library[lib_positions]).toarray().T
    if np.any(sample_positions < 0) or not np.array_equal(y[sample_positions], sample_y):
        raise ValueError('Genome-wide sampled ACR counts differ from previous audit')
    receipt = dict(inputs=inputs, genotypes=genotypes.tolist(), n_acrs=len(acrs),
        n_development_cortex_cells=len(selected), counts_sha256=digest(binary),
        chromosome10_values_verified=int(previous_y.size), genome_sample_values_verified=int(sample_y.size),
        all_whole_genome_totals_verified=True, verified_existing_unreceipted_cache=(
            previous.get('verified_existing_unreceipted_cache', False) if receipt_path.exists() else adopted),
        no_DTS_read=True, no_holdout_values_in_aggregation=True)
    if receipt_path.exists() and previous['counts_sha256'] != receipt['counts_sha256']:
        raise ValueError('Cached counts changed since receipt')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return genotypes, pd.Index(acrs.ACR, name='ACR'), y, n, t


def reference_moments(region_covariates, pcs, features, y):
    """Exact moments of all region × genotype observations without duplication.

    Order: region covariates (including intercept), ancestry PCs, sequence features.
    All regions have the same genotype set. No means of the response replace the
    prediction target: yy retains every pair's variation and ancestry is fitted.
    """
    b, p, x, y = map(lambda v: np.asarray(v, float), [region_covariates, pcs, features, y])
    nr, ng = y.shape
    if (b.shape[0] != nr or x.shape[0] != nr or p.shape[0] != ng
            or not np.all(b[:, 0] == 1) or not nr or not ng
            or not all(np.isfinite(v).all() for v in [b, p, x, y])):
        raise ValueError('Reference region/genotype axes or numeric values disagree')
    bp = np.outer(b.sum(axis=0), p.sum(axis=0))
    bx, px = ng*b.T@x, np.outer(p.sum(axis=0), x.sum(axis=0))
    aa = np.block([[ng*(b.T@b), bp], [bp.T, nr*(p.T@p)]])
    ax = np.vstack([bx, px])
    gram = np.block([[aa, ax], [ax.T, ng*(x.T@x)]])
    target = np.r_[b.T@y.sum(axis=1), p.T@y.sum(axis=0), x.T@y.sum(axis=1)]
    return Moments(nr*ng, gram, target, float(np.sum(y*y)))


def subset_moments(m, covariate_indices, n_all_covariates):
    ix = np.r_[covariate_indices, np.arange(n_all_covariates, len(m.xy))].astype(int)
    return Moments(m.n, m.xx[np.ix_(ix, ix)], m.xy[ix], m.yy)


def prepare_reference(root, cache, config):
    genotypes, acrs, counts, n, t = cortex_cache(root, cache)
    sequence = pd.read_parquet(root/'data/maize_acr_sequences.parquet')
    if not sequence.index.equals(acrs) or not sequence.sequence.str.fullmatch('[ACGT]+').all():
        raise ValueError('Reference sequences must exactly match all source ACRs and contain ACGT only')
    coordinate = parse_acr_coordinates(acrs)
    length = sequence.sequence.str.len().to_numpy()
    if not np.array_equal(length, (coordinate.end-coordinate.start).to_numpy()):
        raise ValueError('Reference lengths and genomic coordinates disagree')
    gc = np.array([(s.count('G')+s.count('C'))/len(s) for s in sequence.sequence])
    gc_basis = gc_spline_basis(gc, np.asarray(config['fixed_gc_knots']))
    chroms = sorted(coordinate.chrom.unique())
    chrom_columns = [coordinate.chrom.eq(c).to_numpy(float) for c in chroms[1:]]
    region = np.column_stack([np.ones(len(acrs)), gc, gc_basis, length, *chrom_columns])
    region_names = ['intercept', 'GC']+[f'GC_spline_{j+1}' for j in range(4)]+['length']+[
        f'chromosome_{c}' for c in chroms[1:]]
    linear = [0, 1, 6]+list(range(7, len(region_names)))
    flexible = [0]+list(range(2, len(region_names)))
    eligible = n >= config['min_pooled_nuclei']
    pcs = read_development_parquet(root/'data/maize_ancestry_pcs.parquet')
    if not pcs.index.equals(genotypes):
        raise ValueError('Ancestry genotype axes disagree')
    p = pcs.to_numpy()[eligible]
    y = np.asarray(counts[:, eligible], float)*1e6/t[eligible][None, :]
    pc_names = [f'ancestry_PC{j+1}' for j in range(p.shape[1])]
    all_names = region_names+pc_names
    pc_indices = list(range(len(region_names), len(all_names)))
    bases = dict(linear=np.array(linear+pc_indices), gc_spline=np.array(flexible+pc_indices))
    keys = pd.MultiIndex.from_arrays([coordinate.chrom, (coordinate.midpoint//config['block_width_bp']).astype(int)])
    block_ids, unique = pd.factorize(keys, sort=True)
    blocks = pd.DataFrame(unique.tolist(), columns=['chromosome', 'block_number'])
    blocks['block'] = np.arange(len(blocks))
    blocks['start_bp'] = blocks.block_number*config['block_width_bp']
    blocks['end_bp'] = blocks.start_bp+config['block_width_bp']
    coordinate['block'] = block_ids
    coordinate['GC'], coordinate['length'] = gc, length
    cohort = pd.DataFrame(dict(genotype=genotypes, n_nuclei=n, total_events=t, eligible=eligible))
    return dict(acrs=acrs, genotypes=genotypes[eligible], all_genotypes=genotypes,
        region=region, pcs=p, y=y, covariate_names=all_names, bases=bases,
        sequences=sequence, coordinates=coordinate, blocks=blocks, cohort=cohort)


def reference_features(root, data, kind):
    if kind in ['frequency', 'softlme']:
        counts, names, multiplier = reference_features(root, data, 'kmer')
        lengths = data['coordinates'].length.to_numpy()
        if kind == 'frequency':
            return counts/(lengths[:, None]-4), names, multiplier
        from .sequence_summaries import soft_word_lme
        return soft_word_lme(counts, lengths, alpha=.99), names, multiplier
    if kind == 'motif':
        motif = pd.read_parquet(root/'data/maize_acr_motif_scores.parquet')
        if not motif.index.equals(data['acrs']) or not motif.columns.is_unique or motif.shape[1] != 927:
            raise ValueError('Motif ACR order or motif identities disagree')
        x = motif.to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite reference motif scores')
        return x, motif.columns.tolist(), 1.
    if kind != 'kmer':
        raise ValueError('Unknown reference feature kind')
    if '_canonical_count_features' in data:
        return data['_canonical_count_features']
    words, _, canonical, mapping = kmer_dictionary(5)
    x = np.empty((len(data['acrs']), len(canonical)))
    for i, sequence in enumerate(data['sequences'].sequence):
        x[i] = np.bincount(mapping, weights=count_kmers(sequence), minlength=len(canonical))
    if not np.array_equal(x.sum(axis=1), data['coordinates'].length.to_numpy()-4):
        raise ValueError('Reference k-mer counts do not sum to viable windows')
    result = x, [words[j] for j in canonical], .5
    data['_canonical_count_features'] = result
    return result
