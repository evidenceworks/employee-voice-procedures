# Missing voice reports: conservative allocation bounds

## Question and scope

Do the released counts alone permit the 440 Q57-valid cases excluded for incomplete voice reports to reverse the raw configuration comparisons? This check uses no respondent records. It addresses the descriptive complete-case profile, not the adjusted interaction, survey-weighted population estimates or the missing-at-random assumption.

`T1_Procedure_by_voice.csv` contains the 12 observed configuration-by-response counts. `voice_item_response_counts.csv` supplies 210 excluded affirmative, 74 excluded negative and 156 excluded uncertain answers. Thus A = 210 and B = 230. The observed sample is 6,548 and the expanded Q57-valid domain is 6,988.

Every excluded case may be allocated to any configuration while its procedural answer remains fixed. Restrictions implied by an already observed voice item are deliberately relaxed. The intervals are therefore conservative outer bounds for the actual missing-voice problem, not necessarily its sharp identified set. Each endpoint is considered separately; endpoints for different configurations or contrasts need not occur in one common allocation.

## Derivation

For observed affirmative count y in a configuration of size n, assigning an affirmative case raises or leaves unchanged its affirmative share. Assigning a nonaffirmative case lowers or leaves unchanged that share. Monotonicity gives:

- Affirmative share: y/(n+B) to (y+A)/(n+A).
- Negative-or-uncertain share: (n-y)/(n+A) to (n-y+B)/(n+B).

For a contrast of configuration g against neither (indexed 0), the lower bound assigns all B nonaffirmative cases to g and all A affirmative cases to 0. The upper bound reverses those assignments:

- Lower: y_g/(n_g+B) − (y_0+A)/(n_0+A).
- Upper: (y_g+A)/(n_g+A) − y_0/(n_0+B).

These endpoints are feasible in the relaxed allocation space for each contrast separately. Multiplication by 100 yields percentages or percentage points as labeled. No sampling uncertainty is added to these deterministic sensitivity calculations.

## Results

| Quantity | Observed, unweighted | Lower | Upper |
|---|---:|---:|---:|
| Neither: negative or uncertain, % | 61.05 | 51.43 | 67.68 |
| Representation only: negative or uncertain, % | 48.08 | 38.17 | 59.58 |
| Meetings only: negative or uncertain, % | 39.78 | 34.18 | 48.94 |
| Both: negative or uncertain, % | 39.01 | 36.70 | 42.95 |
| Representation only minus neither: affirmative, pp | 12.97 | −8.15 | 29.50 |
| Meetings only minus neither: affirmative, pp | 21.27 | 2.48 | 33.50 |
| Both minus neither: affirmative, pp | 22.04 | 8.48 | 30.98 |

The representation-only comparison can reverse sign. The other two raw baseline comparisons cannot under these allocations. The both-channel nonaffirmative share remains substantial even when all excluded affirmative cases are assigned to that group. These are unweighted results: 39.01% is not a replacement for the article's weighted 38.25%.

The bounds do not cover the 50 routed cases lacking a valid Q57 answer, respondents outside the Q57 exposure route, sampling error, survey weights, covariates or causal effects. They neither establish ignorable missingness nor validate the adjusted representation interaction.

## Reproduction and verification

```sh
python code/missing_voice_bounds.py --results results --out results/missing_voice_allocation_bounds.csv
python -m unittest discover -s tests -v
```

The output records each endpoint's formula and full precision. Independent tests verify input partitions, exact rational endpoints and exhaustive allocations in a small example. The seven new noninferential rows remain separate from Data S1 and the original inferential results.
