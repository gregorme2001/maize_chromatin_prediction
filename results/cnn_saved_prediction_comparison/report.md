# Earlier CNN predictions on the current evaluation sample

No model was trained, tuned, refitted, or selected in this comparison. No phenotype or permanent-holdout values were read.

Every row evaluates the same 495,900 ACR-genotype observations at 9,666 ACRs and 137 development genotypes. The response is unchanged original-core, library-pooled cortex CPM.

Earlier predictions are saved outer-held-out predictions. ACRs and genotypes were explicitly matched by name. Frozen outer and inner fold manifests are identical; each evaluation ACR was absent from its corresponding earlier outer training fold. Recorded observed CPM agrees to 1e-12.

Earlier predictions equal the saved three-seed ACR prediction plus that outer fold's genotype-specific centered ancestry contribution. All ten saved ensembles were reconstructed from their three saved seed prediction arrays.

| Model | Observation RMSE | ACR-average RMSE | ACR-average R2 | Within-ACR R2 |
|---|---:|---:|---:|---:|
| Current covariates only | 7.748456 | 4.888121 | 0.027279 | 0.000208 |
| Current TF ridge: B73 core | 7.556236 | 4.527799 | 0.165399 | 0.000187 |
| Current TF ridge: personalized core | 7.555767 | 4.525719 | 0.166166 | 0.000092 |
| Earlier CNN: B73 core | 7.577269 | 4.563722 | 0.152103 | -0.000016 |
| Earlier CNN: B73 2,048 bp | 7.324438 | 4.091248 | 0.318578 | -0.000016 |
| Current CNN: B73 core | 7.746231 | 4.883676 | 0.029047 | 0.000159 |
| Current CNN: personalized core | 7.749307 | 4.889447 | 0.026751 | 0.000176 |
| Current CNN: B73 2,000 bp | 7.735766 | 4.858128 | 0.039179 | 0.000180 |
| Current CNN: personalized 2,000 bp | 7.742670 | 4.869006 | 0.034872 | 0.000075 |

This equalizes evaluation observations, not training data or architectures: earlier networks trained on roughly 108,843 ACRs across all folds, versus 9,666 in the current sample. Earlier models use B73 DNA, 2,048-bp context, an ACR-location channel, eight positional summary bins and three-seed averaging. Current models use 2,000-bp context, global mean/max pooling and one seed. Eligibility, available genotypes per ACR, learning rates and other settings also differ.

The comparison therefore cannot attribute a performance difference specifically to architecture, training-set size, or personalization. It is a retrospective comparison of completed procedures, not independent confirmation after model selection.

Observation RMSE averages squared errors over all available ACR-genotype observations. ACR-average RMSE averages observed and predicted CPM over the identical available genotypes at each ACR, then weights ACRs equally. Lower is better. Within-ACR R2 measures prediction of genotype deviations after centering each ACR for evaluation only.

