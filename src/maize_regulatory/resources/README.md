# Permanent genotype identities

- `published_genotypes.txt`: 172 names in the published GEO matrix's original column order.
- `analysis_genotypes.txt`: 169 normalized names in the legacy analysis order.
- `holdout_genotypes.txt`: the original 30 holdout identities, in sklearn split order.
- `sample_identity.json`: source URL/member, header checksum, exclusions, split settings,
  and checksums of the identity lists.
- `analysis_exclusions.json`: current user-authorized QC exclusions, layered over
  the frozen lists. CML341 is excluded from analysis, leaving 138 development
  genotypes and the original 30 holdouts. This does not rewrite the source lists.

Extracted using `scripts/freeze_sample_ids.py`. Only the header of the published
matrix and the `Family_Inbred_Name` column of the supplied phenotype CSV were
used; no DTS values were read. The published exclusions were B73 (ambiguous
phenotype rows), ILLHY and YU796 (no unique matching phenotype row).

These small public identity lists belong in version control; raw data remain in
git-ignored `data/`. Normal downloads/preparation do not fetch the GEO matrix.
`--overwrite` never redefines these lists. Preparation and real-data loading
reject changed analysis identities/order; holdout IDs are checked against
`train_test_split(np.arange(169), test_size=30, random_state=42)`.
