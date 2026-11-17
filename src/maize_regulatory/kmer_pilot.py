"""SNP-only sequence features and training-only support for the cortex pilot."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

from .candidate_inventory import prepare_variant_support
from .cohort import development_genotypes, read_development_parquet
from .spatial import parse_acr_coordinates


def kmer_dictionary(k=5):
    words = ['']
    for _ in range(k):
        words = [prefix + base for prefix in words for base in 'ACGT']
    lookup = {word: i for i, word in enumerate(words)}
    complement = str.maketrans('ACGT', 'TGCA')
    reverse = np.array([lookup[w.translate(complement)[::-1]] for w in words])
    representatives = np.minimum(np.arange(len(words)), reverse)
    canonical, mapping = np.unique(representatives, return_inverse=True)
    return words, reverse, canonical, mapping


def count_kmers(sequence, k=5):
    """Forward-strand counts, requiring an entirely A/C/G/T sequence."""
    if len(sequence) < k or any(base not in 'ACGT' for base in sequence):
        raise ValueError('A sequence must contain >=k unambiguous DNA bases')
    encoded = np.array(['ACGT'.index(base) for base in sequence], dtype=np.int32)
    code = np.zeros(len(sequence) - k + 1, dtype=np.int32)
    for j in range(k):
        code = 4 * code + encoded[j:j + len(code)]
    return np.bincount(code, minlength=4 ** k)


def build_sequence_features(variants, dosage, sequences, genotypes, acrs, output, k=5):
    """Cache all chr10 sequence patterns before selection, without accessibility.

    Canonical frequencies equal sums of reverse-complement frequencies. A full
    strand-averaged 1024-vector repeats each canonical frequency divided by two.
    The solver uses an exactly equivalent 512-column compression (see its penalty).
    Missing feature rows have zero storage AND an explicit unavailable pattern ID.
    """
    support = prepare_variant_support(variants, dosage, genotypes, acrs)
    if not sequences.index.equals(acrs):
        raise ValueError('Reference sequence order differs from ACR order')
    coordinates = parse_acr_coordinates(acrs)
    words, reverse, canonical, mapping = kmer_dictionary(k)
    n_features = len(canonical)
    pattern_ids = np.full((len(acrs), len(genotypes)), -1, dtype=np.int32)
    reference_ids = np.empty(len(acrs), dtype=np.int32)
    patterns, count_rows, delta_rows, delta_cols, delta_values = [], [], [], [], []
    for i, acr in enumerate(acrs):
        reference = sequences.sequence.iloc[i]
        if len(reference) != coordinates.end.iloc[i] - coordinates.start.iloc[i]:
            raise ValueError(f'Coordinate/sequence length mismatch: {acr}')
        ref_counts = np.bincount(mapping, weights=count_kmers(reference, k),
                                 minlength=n_features).astype(np.int32)
        reference_ids[i] = len(patterns)
        patterns.append(dict(pattern_id=len(patterns), ACR=acr, sequence=reference,
                             is_b73=True, n_windows=len(reference) - k + 1))
        count_rows.append(sparse.csr_matrix(ref_counts[None, :]))
        seen = {reference: (reference_ids[i], ref_counts)}
        rows = support['rows_by_acr'][i]
        positions = variants.pos_0based.to_numpy()[rows] - coordinates.start.iloc[i]
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f'Duplicate SNP coordinates inside {acr}')
        for row, position in zip(rows, positions):
            if reference[position] != variants.REF.iloc[row]:
                raise ValueError(f'SNP/reference disagreement: {variants.index[row]}')
        for g in np.flatnonzero(support['homozygous'][i]):
            bases = list(reference)
            for row, position in zip(rows, positions):
                if support['calls'][row, g] == 2:
                    bases[position] = variants.ALT.iloc[row]
            sequence = ''.join(bases)
            if sequence not in seen:
                counts = np.bincount(mapping, weights=count_kmers(sequence, k),
                                      minlength=n_features).astype(np.int32)
                pid = len(patterns)
                patterns.append(dict(pattern_id=pid, ACR=acr, sequence=sequence,
                                     is_b73=False, n_windows=len(sequence) - k + 1))
                count_rows.append(sparse.csr_matrix(counts[None, :]))
                seen[sequence] = (pid, counts)
            pid, counts = seen[sequence]
            pattern_ids[i, g] = pid
            difference = counts - ref_counts
            at = np.flatnonzero(difference)
            delta_rows.extend([i * len(genotypes) + g] * len(at))
            delta_cols.extend(at.tolist())
            delta_values.extend((difference[at] / (len(sequence) - k + 1)).tolist())
        if i % 1000 == 0:
            print(f'Sequence features: {i:,}/{len(acrs):,} ACRs', flush=True)
    delta = sparse.csr_matrix((delta_values, (delta_rows, delta_cols)),
                              shape=(len(acrs) * len(genotypes), n_features))
    if not np.array_equal(pattern_ids >= 0, support['homozygous']):
        raise AssertionError('Feature availability differs from all-SNP homozygous calls')
    if np.max(np.abs(np.asarray(delta.sum(axis=1)).ravel()), initial=0) > 1e-12:
        raise AssertionError('Frequency differences must sum to zero')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(output / 'delta_features.npz', delta)
    sparse.save_npz(output / 'pattern_kmer_counts.npz', sparse.vstack(count_rows, format='csr'))
    np.save(output / 'pattern_ids.npy', pattern_ids)
    np.save(output / 'reference_pattern_ids.npy', reference_ids)
    pd.DataFrame(patterns).to_parquet(output / 'sequence_patterns.parquet', index=False)
    pd.DataFrame({'kmer': words, 'reverse_complement_index': reverse,
                  'canonical_column': mapping}).to_csv(output / 'kmer_dictionary.csv', index=False)
    (output / 'axes.json').write_text(json.dumps(dict(
        acrs=acrs.tolist(), genotypes=genotypes.tolist(), k=k,
        canonical_kmers=[words[j] for j in canonical],
        conceptual_features=len(words), compressed_features=n_features,
        n_patterns=len(patterns), n_available_genotype_acr_pairs=int((pattern_ids >= 0).sum()),
        unavailable_pattern_id=-1, delta_shape=delta.shape,
        global_exception='Fixed sequence-only transforms; no accessibility or fitted scaling.',
    ), indent=2) + '\n')
    return delta, pattern_ids


@dataclass
class PilotData:
    genotypes: pd.Index
    acrs: pd.Index
    variants: pd.DataFrame
    dosage: pd.DataFrame
    support: dict
    sequences: pd.DataFrame
    pcs: np.ndarray
    pooled_n: np.ndarray
    library_rows: pd.DataFrame
    row_genotypes: np.ndarray
    row_libraries: np.ndarray
    libraries: list
    y: np.ndarray
    delta: sparse.csr_matrix | None = None


def load_pilot_data(root, cell_state='cortex'):
    root = Path(root)
    directory = root / 'results/cell_state_accessibility_audit/aggregation'
    genotypes = development_genotypes()
    acrs = pd.Index(pd.read_csv(directory / 'chr10_acrs.csv').ACR, name='ACR')
    sequences = pd.read_parquet(root / 'data/maize_acr_sequences.parquet')
    at = sequences.index.get_indexer(acrs)
    if np.any(at < 0) or not sequences.index[at].equals(acrs):
        raise ValueError('Chromosome-10 reference identity mapping failed')
    sequences = sequences.iloc[at]
    variants = pd.read_parquet(root / 'data/intermediate/panzea_v5_pilot/chr10_variants.parquet')
    dosage = pd.read_parquet(root / 'data/intermediate/panzea_v5_pilot/chr10_alt_dosage.parquet')
    support = prepare_variant_support(variants, dosage, genotypes, acrs)
    pcs = read_development_parquet(root / 'data/maize_ancestry_pcs.parquet').to_numpy()
    groups = pd.read_csv(directory / 'genotype_cell_state_nuclei.csv')
    state = groups.loc[groups.cell_state.eq(cell_state)]
    if not pd.Index(state.genotype, name=genotypes.name).equals(genotypes):
        raise ValueError('Pooled nucleus genotype order mismatch')
    pooled_n = state.n_nuclei.to_numpy()
    rows = pd.read_csv(directory / 'genotype_cell_state_library_nuclei.csv')
    if not np.array_equal(rows.library_group, np.arange(len(rows))):
        raise ValueError('Library matrix row axis is not explicit/sequential')
    selected = np.flatnonzero(rows.cell_state.eq(cell_state).to_numpy())
    chunks = json.loads((directory / 'chr10_library_counts/files.json').read_text())
    y = sparse.hstack([sparse.load_npz(directory / 'chr10_library_counts' / f)[selected]
                       for f in chunks], format='csr').toarray()
    rows = rows.iloc[selected].reset_index(drop=True)
    row_genotypes = genotypes.get_indexer(rows.genotype)
    libraries = sorted(rows.library.unique())
    row_libraries = pd.Index(libraries).get_indexer(rows.library)
    if np.any(row_genotypes < 0) or y.shape != (len(rows), len(acrs)):
        raise ValueError('Library accessibility identity mismatch')
    if (not np.array_equal(np.bincount(row_genotypes, weights=rows.n_nuclei,
                                      minlength=len(genotypes)), pooled_n)
            or np.any(y < 0) or np.any(y > rows.n_nuclei.to_numpy()[:, None])):
        raise ValueError('Library/pooled count denominators disagree')
    return PilotData(genotypes, acrs, variants, dosage, support, sequences,
                     pcs, pooled_n, rows, row_genotypes, row_libraries, libraries, y)


def select_acrs(data, training_genotypes, config):
    """Only training genotype calls and training cell-support metadata select ACRs."""
    g = np.asarray(training_genotypes)
    calls = data.support['calls'][:, g]
    n_called = (calls >= 0).sum(axis=1)
    alt = np.where(calls >= 0, calls, 0).sum(axis=1)
    maf = np.divide(np.minimum(alt, 2 * n_called - alt), 2 * n_called,
                    out=np.zeros(len(calls)), where=n_called > 0)
    common = (n_called >= config['min_call_rate'] * len(g)) & (maf >= config['min_maf'])
    usable = data.support['homozygous'][:, g] & (data.pooled_n[g] >= config['min_nuclei'])
    n_usable = usable.sum(axis=1)
    variant_usable = usable[data.support['locations']]
    n_alt_lines = ((calls == 2) & variant_usable).sum(axis=1)
    denominator = n_usable[data.support['locations']]
    local_maf = np.divide(np.minimum(n_alt_lines, denominator - n_alt_lines), denominator,
                          out=np.zeros(len(calls)), where=denominator > 0)
    good = common & (local_maf >= config['min_maf']) & (denominator > 0)
    n_snps = np.bincount(data.support['locations'][good], minlength=len(data.acrs))
    return np.flatnonzero((n_usable >= config['min_train_genotypes'])
                          & (n_snps >= config['min_snps']))


def prediction_metrics(data, acr_indices, row_indices, prediction, config):
    """Pool library predictions by N, then equally weight ACRs within genotype.

    N is used to reconstruct a genotype's predicted pooled detection frequency,
    not to weight that genotype's error against other genotypes.
    """
    a, r = np.asarray(acr_indices), np.asarray(row_indices)
    g = data.row_genotypes[r]
    n = data.library_rows.n_nuclei.to_numpy()[r]
    indicator = sparse.csr_matrix((np.ones(len(r)), (np.arange(len(r)), g)),
                                  shape=(len(r), len(data.genotypes)))
    observed = data.y[np.ix_(r, a)].T @ indicator
    expected = (prediction * n[None, :]) @ indicator
    relevant_g = np.unique(g)
    output = []
    for genotype in relevant_g:
        valid = (data.support['homozygous'][a, genotype]
                 & (data.pooled_n[genotype] >= config['min_nuclei']))
        if not valid.any():
            continue
        p_obs = observed[valid, genotype] / data.pooled_n[genotype]
        p_hat = expected[valid, genotype] / data.pooled_n[genotype]
        output.append(dict(genotype_index=int(genotype), genotype=data.genotypes[genotype],
            n_acrs=int(valid.sum()), mse=float(np.mean((p_hat - p_obs) ** 2)),
            mae=float(np.mean(np.abs(p_hat - p_obs)))))
    return pd.DataFrame(output), np.asarray(observed), np.asarray(expected)
