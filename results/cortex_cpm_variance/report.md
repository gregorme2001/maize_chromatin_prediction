# Cortex CPM: genotype means versus library deviations

Two descriptive CPM quantities, calculated separately at each ACR and then summarized equally across ACRs. No F statistic or significance test.

## Definitions

Fix ACR i. g indexes genotype and l library. All cells are cortex. Y_igl is the number of nuclei detecting ACR i. T_gl is the total number of positive nucleus–ACR detections across ALL 108,843 ACRs in the SAME genotype/library cortex cells. R_igl = 1,000,000 × Y_igl/T_gl is a library CPM. C_ig = 1,000,000 × Σ_lY_igl/Σ_lT_gl is the genotype CPM after pooling libraries. Library weights w_gl = T_gl/Σ_l′T_gl′, so C_ig = Σ_l w_gl R_igl. Weights use total detections, not nucleus counts. G genotypes have equal weight; Cbar_i=(1/G)Σ_g C_ig.

## Coverage

108,843 ACRs on all ten chromosomes; 138 development genotypes, 384 genotype/library groups, and 93,651 cortex nuclei. No minimum-cell, SNP, MAF or accessibility filter is applied. H95 is retained in this descriptive analysis. CML341 and the fixed 30 holdouts remain excluded by existing policy. Seven genotypes have only one observed cortex library and therefore contribute zero observed library deviation. The separate multiple_libraries view uses the 131 genotypes with at least two libraries for BOTH quantities; this is not a new permanent exclusion.

## Interpretation

These are observed variations, not purified genetic or technical components. Library deviations include finite-cell sampling, preparation effects and possible plant differences. Genotype means also retain sampling and library-composition effects. Pooling libraries itself reduces variation, so a larger library quantity does not establish that genetic variance is absent. Very small groups can give extreme CPMs; event weighting reduces their influence within genotype but does not eliminate it. Every ACR has equal weight in the summary; high-variance ACRs can dominate the mean, so medians and quantiles are saved too. No regression adjustment, statistical test, noise subtraction, confidence interval, heritability or predictive performance claim is made. No DTS, holdout values or previous prediction folds are used.

## Validation

The library-count cache and source identities were verified before use. All 15,020,334 genome-wide genotype/ACR pooled counts match the earlier independent genotype cache. Genome-wide totals match the earlier audit. Event-weighted means reproduce ratio-of-pooled-count CPMs. Weighted library deviations plus genotype variance reconstruct total observed variance to floating-point precision. Source and code hashes are retained in summary.json.

     Genotype view                    Quantity  Mean variance (CPM²)  Median variance (CPM²)  √mean variance (CPM)  Median ACR SD (CPM)
   all_development weighted_library_deviations             55.886593               40.125868              7.475734             6.334498
   all_development  variance_of_genotype_means             37.015574               30.909338              6.084043             5.559617
multiple_libraries weighted_library_deviations             58.872899               42.269998              7.672868             6.501538
multiple_libraries  variance_of_genotype_means             34.379726               28.822429              5.863423             5.368652

Exact formulas are in report.html and docs/cortex_cpm_variance.md.
