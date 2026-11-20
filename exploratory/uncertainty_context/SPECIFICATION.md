# Cross-domain uncertainty comparison: executable specification

This specification documents the implemented exploratory comparison of explicit uncertainty responses across organizational topics. It describes the source, analytic domain, estimator and recorded sensitivity results.

## Source and access
EWCS 2024, second edition, UK Data Service study 9511, DOI 10.5255/UKDA-SN-9511-2. Expected authenticated Stata-file SHA-256: d5cb93c4f7f4ad148ad6f1628abe061b29252516b2c8eb98f05a05ccc9d2e49c. Expected dimensions: 36,644 observations and 394 variables. Only an authorized researcher may run the model locally. Do not upload or redistribute respondent records.

## Population and observations
Use the 27 EU countries listed in the code. Retain an employee when employee_selfdeclared=1 or paid_salary=1. Require at least one of tech_task_alloc, tech_hours and tech_monitor_perf in {1,2}, and aut_procedure_decision in {1,2,3}. This produces the archived 6,992 eligible observations before comparator completeness. Require trade_union and osh_delegate in {1,2,-999}, and work_accom_periods and work_accomodations_menopause in {1,2,3,-999}. The common sample is 6,674. No complete meeting response is required.

## Outcomes and contrasts
Each of the four comparator items is one only when its released numeric code is -999, otherwise zero for the allowed substantive codes. Representation uncertainty is the mean of the trade_union and osh_delegate indicators. Accommodation uncertainty is the mean of the two accommodation indicators. The paired outcome is representation uncertainty minus accommodation uncertainty. The focal contrast is Q57=3 minus Q57=2; Q57=3 minus Q57=1 is interpretive. Do not equate unobserved or system-missing answers with explicit uncertainty. The two Q79 organizational-climate items are not valid explicit-uncertainty comparators in this release.

## Adjustment
The design matrix contains three Q57 indicators and categorical controls, omitting one level per control. Age uses cut points [15,34,54,74], labeled 16-34, 35-54 and 55-74. Gender maps sex_id {1,2,3}; education groups ISCED 0-2, 3-4 and 5-8. Occupation and sector use their released values directly as categorical strings. Workplace size maps codes 1-3 to 1-9, 4 to 10-49, 5-6 to 50-249 and 7-8 to 250+. Contract maps 1 to permanent, 2-3 to fixed-term/agency, 4 and 995 to other/training, 5 to no contract and other values to unknown. Part-time retains 1/2 with other values coded unknown. Each Q56 function retains 1-4 with other values coded unknown. The age, occupation and other released-category coding must not be assumed identical to the main configuration model.

## Estimation and uncertainty
Fit weighted least squares with released calweight normalized by its sample mean. The full EU27 country-nested PSU/stratum frame includes zero scores outside the analytic domain. Use with-replacement stratum-centered PSU covariance, adding the average nonsingleton-stratum contribution for each singleton. Use 2,455 design degrees of freedom and pointwise t intervals. Compute the cross-outcome covariance jointly. A separate fit to the within-person paired outcome checks the estimate and its variance algebraically.

## Archived sensitivity rule
Additional checks are triggered when the absolute focal difference is at least 3 percentage points and its pointwise interval excludes zero. The threshold is an operational choice for this exploratory analysis. Checks use self-declared employees only, equal-country weights, gender-by-age categories and each of 27 country deletions. The reported output retains every result and compares the outcomes directly.

## Interpretation and output
The negative paired difference against Q57=2 indicates that explicit procedural uncertainty is also associated with uncertain responses about other organizational topics. Interpretation reflects differences in topic relevance, response options and baseline uncertainty. Outputs contain aggregate CSVs and execution metadata, subject to the study's disclosure protections.
