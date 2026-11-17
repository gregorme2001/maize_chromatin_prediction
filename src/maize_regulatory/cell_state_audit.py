"""Sparse detection auditing; no phenotype, association test, or sequence reconstruction."""

import itertools
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .preparation import normalize_reconstructed_genotype
from .spatial import parse_acr_coordinates


def cell_groups(metadata, genotypes):
    """Explicit cell/library/pooled-group maps, retaining zero-cell combinations."""
    required = {'cellID', 'genotype', 'normalized_genotype', 'matrix_row',
                'celltype_full', 'library'}
    if not required.issubset(metadata.columns):
        raise ValueError(f'Missing metadata fields: {required - set(metadata.columns)}')
    if (not metadata.cellID.is_unique or metadata.cellID.isna().any()
            or not np.array_equal(metadata.matrix_row, np.arange(1, len(metadata) + 1))):
        raise ValueError('Cell identity/order mismatch')
    if not genotypes.is_unique:
        raise ValueError('Duplicate development identities')
    cells = metadata.loc[metadata.normalized_genotype.isin(genotypes)].copy()
    normalized = cells.genotype.map(normalize_reconstructed_genotype)
    if not normalized.equals(cells.normalized_genotype):
        raise ValueError('Original/normalized genotype mapping disagrees')
    missing = cells.isna().sum().to_dict()
    cells['celltype_full'] = cells.celltype_full.fillna('unannotated')
    cells['library'] = cells.library.fillna('unassigned_library')
    states = sorted(cells.celltype_full.unique())
    pooled_index = pd.MultiIndex.from_product([genotypes, states], names=['genotype', 'cell_state'])
    keys = pd.MultiIndex.from_arrays([cells.normalized_genotype, cells.celltype_full])
    pooled_ids = pooled_index.get_indexer(keys)
    library_index = pd.MultiIndex.from_tuples(sorted(set(zip(
        pooled_ids, cells.library))), names=['pooled_group', 'library'])
    library_ids = library_index.get_indexer(pd.MultiIndex.from_arrays([pooled_ids, cells.library]))
    if np.any(pooled_ids < 0) or np.any(library_ids < 0):
        raise ValueError('Unmapped cell group')
    groups = pooled_index.to_frame(index=False)
    groups['n_nuclei'] = np.bincount(pooled_ids, minlength=len(groups))
    if (groups.groupby('genotype', sort=False).n_nuclei.sum() == 0).any():
        raise ValueError('A development genotype has no cells')
    groups['observed'] = groups.n_nuclei.gt(0)
    libraries = library_index.to_frame(index=False)
    libraries['n_nuclei'] = np.bincount(library_ids, minlength=len(libraries))
    libraries['genotype'] = groups.genotype.to_numpy()[libraries.pooled_group]
    libraries['cell_state'] = groups.cell_state.to_numpy()[libraries.pooled_group]
    libraries.index.name = 'library_group'
    selected = pd.DataFrame({'matrix_row': cells.matrix_row.to_numpy(), 'library_group': library_ids})
    projection = sparse.csr_matrix((np.ones(len(libraries), dtype=np.int32),
                                    (libraries.pooled_group, np.arange(len(libraries)))),
                                   shape=(len(groups), len(libraries)))
    return cells, selected, groups, libraries, projection, missing


def aggregate_detections(matrix, cell_ids, metadata, genotypes):
    """Small-matrix reference used by tests; production R bridge uses the same maps."""
    if not sparse.issparse(matrix) or not pd.Index(cell_ids).equals(pd.Index(metadata.cellID)):
        raise ValueError('Sparse matrix/cell identity mismatch')
    cells, selected, groups, libraries, projection, missing = cell_groups(metadata, genotypes)
    block = matrix.tocsr()[selected.matrix_row.to_numpy() - 1].copy()
    if np.any(~np.isfinite(block.data)) or np.any(block.data < 0):
        raise ValueError('Invalid accessibility values')
    block.eliminate_zeros()
    block.data[:] = 1
    indicator = sparse.csr_matrix((np.ones(len(selected), dtype=np.int32),
                                   (selected.library_group, np.arange(len(selected)))),
                                  shape=(len(libraries), len(selected)))
    lib = (indicator @ block).tocsr()
    return (projection @ lib).tocsr(), lib, groups, libraries


def _read_exact(handle, size):
    parts = bytearray()
    while len(parts) < size:
        piece = handle.read(size - len(parts))
        if not piece:
            raise ValueError('Truncated sparse R stream; inspect stream.log')
        parts.extend(piece)
    return bytes(parts)


def stream_library_counts(command, n_libraries, n_acrs, log_path):
    """Bounded binary CSC chunks; never export or load the entire cell matrix in Python."""
    fifo = Path(log_path).parent / '.sparse_stream_fifo'
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    with open(log_path, 'w') as log:
        process = subprocess.Popen([*command, str(fifo)], stdout=subprocess.PIPE, stderr=log)
        stream = None
        try:
            if process.stdout.readline() != b'READY\n':
                raise ValueError(f'R stream failed to open; see {log_path}')
            os.set_blocking(descriptor, True)
            stream = os.fdopen(descriptor, 'rb')
            if _read_exact(stream, 12) != b'MAIZEAUDIT1\n':
                raise ValueError('Unexpected R stream signature')
            shape = struct.unpack('<ii', _read_exact(stream, 8))
            if shape != (n_libraries, n_acrs):
                raise ValueError('R stream group/ACR dimensions disagree')
            next_start = 0
            while True:
                start, width, nnz = struct.unpack('<iii', _read_exact(stream, 12))
                if start == -1:
                    if next_start != n_acrs:
                        raise ValueError('Incomplete ACR stream')
                    break
                if start != next_start or width <= 0 or nnz < 0 or start + width > n_acrs:
                    raise ValueError('Unexpected chunk order/dimensions')
                ptr = np.frombuffer(_read_exact(stream, 4 * (width + 1)), dtype='<i4')
                rows = np.frombuffer(_read_exact(stream, 4 * nnz), dtype='<i4')
                values = np.frombuffer(_read_exact(stream, 4 * nnz), dtype='<i4')
                matrix = sparse.csc_matrix((values, rows, ptr), shape=(n_libraries, width))
                matrix.check_format(full_check=True)
                yield start, matrix
                next_start += width
            if stream.read(1):
                raise ValueError('Unexpected trailing R stream content')
            if process.wait() != 0:
                raise ValueError(f'R aggregation failed; see {log_path}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if stream is not None:
                stream.close()
            else:
                os.close(descriptor)
            fifo.unlink()


def detection_summary(values, nuclei):
    """One state/threshold/chunk; zeros are observations, absent groups are excluded."""
    values = np.asarray(values)
    nuclei = np.asarray(nuclei)
    if values.ndim != 2 or nuclei.shape != (len(values),) or np.any(nuclei <= 0):
        raise ValueError('Require observed groups with positive denominators')
    if (np.any(values < 0) or np.any(values > nuclei[:, None])
            or np.any(values != np.floor(values))):
        raise ValueError('Detection counts must be integer and <= nuclei')
    frequencies = values / nuclei[:, None]
    bins = np.array([np.sum(values == 0), np.sum(values == 1),
                     np.sum((values >= 2) & (values <= 4)),
                     np.sum((values >= 5) & (values <= 9)), np.sum(values >= 10)], dtype=np.int64)
    hist = np.histogram(frequencies[values > 0], bins=np.linspace(0, 1, 1001))[0]
    width = values.shape[1]
    metrics = {
        'n_genotypes_eligible': np.full(width, len(nuclei), dtype=np.int32),
        'total_detected_nuclei': values.sum(axis=0, dtype=np.int64),
        'n_nuclei_total': np.full(width, nuclei.sum(), dtype=np.int64),
        'mean_detection_frequency': frequencies.mean(axis=0),
        'sd_detection_frequency': frequencies.std(axis=0, ddof=1) if len(nuclei) > 1
        else np.full(width, np.nan),
        'range_detection_frequency': np.ptp(frequencies, axis=0),
    }
    for threshold in [1, 3, 5, 10]:
        metrics[f'n_genotypes_events_ge_{threshold}'] = (values >= threshold).sum(axis=0).astype(np.int32)
    return metrics, bins, hist


def run_aggregation(metadata, genotypes, acrs, matrix_path, config, output, rscript='Rscript'):
    """Stream library aggregates, summarize all ACRs, retain only chr10 and a QC sample."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cells, selected, groups, libraries, projection, missing = cell_groups(metadata, genotypes)
    selected.to_csv(output / 'selected_cell_rows.csv', index=False)
    groups.to_csv(output / 'genotype_cell_state_nuclei.csv', index=False)
    libraries.to_csv(output / 'genotype_cell_state_library_nuclei.csv')
    acrs.to_csv(output / 'audited_acrs.csv', index=False)
    coords = parse_acr_coordinates(acrs.ACR)
    states = sorted(groups.cell_state.unique())
    sampled = np.sort(np.random.default_rng(config['seed']).choice(
        len(acrs), min(config['sample_acrs_for_library_checks'], len(acrs)), replace=False))
    sample_parts, chr10_parts = [], []
    chr10_ids, library_chr10_files = [], []
    (output / 'chr10_library_counts').mkdir()
    accumulator = {(s, n): {'counts': np.zeros(5, np.int64), 'hist': np.zeros(1000, np.int64)}
                   for s in states for n in config['minimum_nuclei']}
    library_totals = np.zeros(len(libraries), dtype=np.int64)
    writer = None
    script = Path(__file__).resolve().parents[2] / 'scripts/stream_accessibility_groups.R'
    command = [rscript, str(script), str(matrix_path), str(output / 'source_cell_metadata.csv'),
               str(output / 'selected_cell_rows.csv'), str(output / 'audited_acrs.csv'),
               str(output), str(config['chunk_acrs']), str(len(libraries))]
    metadata.to_csv(output / 'source_cell_metadata.csv', index=False)
    try:
        for start, lib in stream_library_counts(command, len(libraries), len(acrs), output / 'stream.log'):
            end = start + lib.shape[1]
            if np.any(lib.data <= 0) or np.any(lib.data > libraries.n_nuclei.to_numpy()[lib.indices]):
                raise ValueError('Library detections exceed the number of measured nuclei')
            pooled = (projection @ lib).tocsr()
            if pooled[groups.n_nuclei.eq(0).to_numpy()].nnz:
                raise ValueError('An unobserved genotype/state group has detections')
            library_totals += np.asarray(lib.sum(axis=1)).ravel().astype(np.int64)
            selected_sample = sampled[(sampled >= start) & (sampled < end)] - start
            if len(selected_sample):
                sample_parts.append(lib[:, selected_sample])
            selected_chr10 = np.flatnonzero(coords.chrom.to_numpy()[start:end] == 10)
            if len(selected_chr10):
                filename = f'chunk_{start:06d}.npz'
                sparse.save_npz(output / 'chr10_library_counts' / filename, lib[:, selected_chr10])
                library_chr10_files.append(filename)
                chr10_parts.append(pooled[:, selected_chr10])
                chr10_ids.extend(acrs.ACR.iloc[start + selected_chr10].tolist())
            frames = []
            for state in states:
                rows = np.flatnonzero(groups.cell_state.eq(state).to_numpy())
                dense = pooled[rows].toarray()
                nuclei = groups.n_nuclei.to_numpy()[rows]
                for minimum in config['minimum_nuclei']:
                    keep = nuclei >= minimum
                    if not keep.any():
                        continue
                    metrics, categories, hist = detection_summary(dense[keep], nuclei[keep])
                    accumulator[state, minimum]['counts'] += categories
                    accumulator[state, minimum]['hist'] += hist
                    frame = pd.DataFrame(metrics)
                    frame.insert(0, 'ACR', acrs.ACR.iloc[start:end].to_numpy())
                    frame.insert(0, 'min_nuclei', minimum)
                    frame.insert(0, 'cell_state', state)
                    frames.append(frame)
            if frames:
                table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output / 'acr_state_accessibility.parquet', table.schema,
                                              compression='zstd')
                writer.write_table(table)
            print(f'Aggregated {end:,}/{len(acrs):,} ACRs', flush=True) if start == 0 or end % 5120 == 0 else None
    finally:
        if writer is not None:
            writer.close()
    if chr10_parts:
        sparse.save_npz(output / 'chr10_pooled_counts.npz', sparse.hstack(chr10_parts, format='csr'))
    pd.DataFrame({'ACR': chr10_ids}).to_csv(output / 'chr10_acrs.csv', index=False)
    (output / 'chr10_library_counts/files.json').write_text(json.dumps(library_chr10_files))
    sparse.save_npz(output / 'library_qc_sample_counts.npz', sparse.hstack(sample_parts, format='csr'))
    acrs.iloc[sampled].to_csv(output / 'library_qc_sample_acrs.csv', index=False)
    libraries['total_detected_events'] = library_totals
    libraries['mean_detected_acrs_per_nucleus'] = library_totals / libraries.n_nuclei
    libraries.to_csv(output / 'genotype_cell_state_library_depth.csv')
    records, histograms = [], []
    for (state, minimum), acc in accumulator.items():
        n = int(((groups.cell_state == state) & (groups.n_nuclei >= minimum)).sum())
        total = n * len(acrs)
        if int(acc['counts'].sum()) != total:
            raise ValueError('Implicit-zero denominator mismatch')
        record = {'cell_state': state, 'min_nuclei': minimum, 'n_genotypes': n,
                  'n_acrs': len(acrs), 'n_observed_acr_blocks': total}
        for label, count in zip(['zero', 'one', 'two_to_four', 'five_to_nine', 'at_least_ten'], acc['counts']):
            record[f'n_{label}'] = int(count)
            record[f'fraction_{label}'] = float(count / total) if total else np.nan
        records.append(record)
        histograms.append(pd.DataFrame({'cell_state': state, 'min_nuclei': minimum,
            'lower_frequency': np.arange(1000) / 1000, 'upper_frequency': np.arange(1, 1001) / 1000,
            'n_nonzero_blocks': acc['hist']}))
    pd.DataFrame(records).to_csv(output / 'state_detection_sparsity.csv', index=False)
    pd.concat(histograms, ignore_index=True).to_parquet(output / 'nonzero_detection_frequency_histogram.parquet', index=False)
    manifest = {'completed': True, 'n_cells': len(cells), 'n_acrs': len(acrs),
                'n_genotypes': len(genotypes), 'n_states': len(states), 'missing_metadata': missing,
                'config': config, 'genotype_order': genotypes.tolist()}
    (output / 'aggregation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def variant_inventory(variants, dosage, sequences, genotypes, audited_acrs):
    """Called diploid allele denominators, explicit REF/coordinate checks, no imputation."""
    if not variants.index.equals(dosage.index) or not dosage.columns.equals(genotypes):
        raise ValueError('Variant/development genotype identity or order mismatch')
    if not variants.index.is_unique or not sequences.index.is_unique:
        raise ValueError('Duplicate variants or ACR sequences')
    coords = parse_acr_coordinates(audited_acrs)
    chr10 = coords.loc[coords.chrom.eq(10)].copy()
    variants = variants.loc[variants.ACR.isin(chr10.index)].copy()
    dosage = dosage.loc[variants.index]
    calls = dosage.to_numpy()
    if not np.isin(calls, [-1, 0, 1, 2]).all():
        raise ValueError('Unexpected dosage coding')
    for row in variants.itertuples():
        ac = chr10.loc[row.ACR]
        if (row.chrom != 10 or row.pos_1based != row.pos_0based + 1
                or not ac.start <= row.pos_0based < ac.end
                or len(row.REF) != 1 or len(row.ALT) != 1
                or sequences.loc[row.ACR, 'sequence'][row.pos_0based - int(ac.start)] != row.REF
                or not row.ref_matches_b73):
            raise ValueError(f'Variant/reference alignment failed: {row.Index}')
    called = calls >= 0
    n_called = called.sum(axis=1)
    alt_count = np.where(called, calls, 0).sum(axis=1)
    mac = np.minimum(alt_count, 2 * n_called - alt_count)
    variable = np.array([len(np.unique(row[row >= 0])) > 1 for row in calls])
    variants['n_called_audit'] = n_called
    variants['call_rate_audit'] = n_called / len(genotypes)
    variants['minor_allele_count'] = mac
    variants['minor_allele_frequency'] = np.divide(mac, 2 * n_called,
        out=np.full(len(mac), np.nan), where=n_called > 0)
    variants['dosage_variable'] = variable
    variants['n_hom_ref_audit'] = (calls == 0).sum(axis=1)
    variants['n_hom_alt_audit'] = (calls == 2).sum(axis=1)
    for column, values in [('n_called', n_called), ('n_hom_ref', (calls == 0).sum(axis=1)),
                           ('n_hom_alt', (calls == 2).sum(axis=1))]:
        if column in variants and not np.array_equal(variants[column], values):
            raise ValueError('Cached variant annotations disagree with dosage values')
    for label, mask in {
        'n_panel_snps': np.ones(len(variants), bool),
        'n_polymorphic_snps': mac > 0,
        'n_dosage_variable_snps': variable,
        'n_mac_ge_5': (mac >= 5) & variable,
        'n_mac_ge_10': (mac >= 10) & variable,
        'n_mac_ge_10_callrate_ge_90pct': (mac >= 10) & variable & (n_called >= .9 * len(genotypes)),
    }.items():
        chr10[label] = variants.loc[mask].groupby('ACR').size().reindex(chr10.index, fill_value=0).astype(int)
    for acr, rows in variants.groupby('ACR', sort=False).indices.items():
        subset = calls[rows]
        chr10.loc[acr, 'missing_call_fraction'] = float((subset < 0).mean())
        complete = subset[:, np.all(subset >= 0, axis=0)].T
        chr10.loc[acr, 'n_complete_genotypes_all_panel_snps'] = len(complete)
        chr10.loc[acr, 'n_distinct_complete_dosage_patterns'] = len(np.unique(complete, axis=0)) if len(complete) else 0
    return variants, dosage, chr10


def joint_scenarios(config):
    """Primary descriptive criteria plus an explicit N/events sensitivity grid."""
    scenarios = list(config['scenarios'])
    for nuclei, events in itertools.product(config['minimum_nuclei'], [3, 5, 10]):
        scenarios.append({'name': f'N{nuclei}_events{events}', 'min_nuclei': nuclei,
                          'min_mac': 10, 'min_call_rate': .9,
                          'min_genotypes_per_allele': 10, 'min_events': events,
                          'min_detected_genotypes_per_allele': 5})
    return scenarios


def joint_feasibility(counts, groups, acr_ids, variants, dosage, genotypes, scenarios):
    """Require support for BOTH homozygote groups at the SAME SNP; no effect tests."""
    if not dosage.columns.equals(genotypes) or not dosage.index.equals(variants.index):
        raise ValueError('Joint variant/genotype axis mismatch')
    if counts.shape != (len(groups), len(acr_ids)) or not pd.Index(acr_ids).is_unique:
        raise ValueError('Joint accessibility shape/identity mismatch')
    locations = pd.Index(acr_ids).get_indexer(variants.ACR)
    if np.any(locations < 0):
        raise ValueError('Variant ACR missing from measured chromosome')
    calls = dosage.to_numpy()
    ref, alt, called = calls == 0, calls == 2, calls >= 0
    summaries, best = [], []
    for state in sorted(groups.cell_state.unique()):
        rows = np.flatnonzero(groups.cell_state.eq(state).to_numpy())
        if not pd.Index(groups.genotype.iloc[rows], name=genotypes.name).equals(genotypes):
            raise ValueError('State genotype order changed')
        y = counts[rows].toarray()
        n = groups.n_nuclei.to_numpy()[rows]
        measured = y[:, locations].T
        for scenario in scenarios:
            eligible = n >= scenario['min_nuclei']
            n_eligible = int(eligible.sum())
            genetic = (variants.dosage_variable.to_numpy()
                & (variants.minor_allele_count.to_numpy() >= scenario['min_mac'])
                & (variants.call_rate_audit.to_numpy() >= scenario['min_call_rate']))
            local_rate = (called[:, eligible].sum(axis=1) / n_eligible
                          if n_eligible else np.zeros(len(variants)))
            callable_state = genetic & (local_rate >= scenario['min_call_rate']) & (n_eligible > 0)
            n_ref = (ref & eligible).sum(axis=1)
            n_alt = (alt & eligible).sum(axis=1)
            represented = callable_state & (n_ref >= scenario['min_genotypes_per_allele']) & (n_alt >= scenario['min_genotypes_per_allele'])
            detected = (measured >= scenario['min_events']) & eligible
            d_ref = (ref & detected).sum(axis=1)
            d_alt = (alt & detected).sum(axis=1)
            joint = represented & (d_ref >= scenario['min_detected_genotypes_per_allele']) & (d_alt >= scenario['min_detected_genotypes_per_allele'])
            accessible = ((y[eligible] >= scenario['min_events']).sum(axis=0)
                          >= 2 * scenario['min_detected_genotypes_per_allele'])
            if n_eligible < 2 * scenario['min_genotypes_per_allele']:
                accessible[:] = False
            record = {'cell_state': state, **scenario, 'n_genotypes_eligible': n_eligible,
                      'n_chr10_acrs': len(acr_ids), 'n_accessibility_supported_acrs': int(accessible.sum())}
            for label, mask in [('genetic', genetic), ('state_callable', callable_state),
                                ('allele_represented', represented), ('joint', joint)]:
                record[f'n_{label}_snps'] = int(mask.sum())
                record[f'n_{label}_acrs'] = int(variants.loc[mask, 'ACR'].nunique())
            summaries.append(record)
            if scenario['name'] != 'benchmark' or not joint.any():
                continue
            subset = variants.loc[joint, ['ACR', 'REF', 'ALT', 'pos_1based', 'minor_allele_count', 'call_rate_audit']].copy()
            subset['variant_id'] = subset.index
            subset = subset.reset_index(drop=True)
            subset['cell_state'] = state
            subset['n_ref_genotypes'] = n_ref[joint]
            subset['n_alt_genotypes'] = n_alt[joint]
            subset['n_ref_detected_genotypes'] = d_ref[joint]
            subset['n_alt_detected_genotypes'] = d_alt[joint]
            subset['min_detected_genotypes'] = np.minimum(d_ref[joint], d_alt[joint])
            subset['min_allele_genotypes'] = np.minimum(n_ref[joint], n_alt[joint])
            subset['state_call_rate'] = local_rate[joint]
            subset['n_ref_nuclei'] = (ref[joint] * (n * eligible)).sum(axis=1)
            subset['n_alt_nuclei'] = (alt[joint] * (n * eligible)).sum(axis=1)
            subset['n_qualifying_snps_in_acr_state'] = subset.groupby('ACR').ACR.transform('size')
            subset = subset.sort_values(['min_detected_genotypes', 'min_allele_genotypes',
                'state_call_rate', 'call_rate_audit', 'variant_id'], ascending=[False, False, False, False, True])
            subset = subset.drop_duplicates('ACR').copy()
            # Frequency summaries describe measurement and selection bias, not effects.
            freq = y[eligible] / n[eligible, None]
            mean = freq.mean(axis=0)
            sd = freq.std(axis=0, ddof=1)
            at = pd.Index(acr_ids).get_indexer(subset.ACR)
            subset['mean_detection_frequency'] = mean[at]
            subset['sd_detection_frequency'] = sd[at]
            subset['accessibility_percentile_in_state_chr10'] = pd.Series(mean).rank(pct=True).to_numpy()[at]
            subset['variability_percentile_in_state_chr10'] = pd.Series(sd).rank(pct=True).to_numpy()[at]
            best.append(subset.reset_index(drop=True))
        print(f'Joint criteria checked: {state}', flush=True)
    return pd.DataFrame(summaries), pd.concat(best, ignore_index=True) if best else pd.DataFrame()
