# Technical Contribution

## Problem addressed

Bettors have short recorded histories, so their observed sport and
bet-type preferences can be noisy. Also, the baseline contains 13
sport-share features, which can collectively influence clustering
more than smaller feature groups.

## Implemented adaptation

Our extension combines two changes before applying K-Means:

1. History-adjusted preferences:
   adjusted share = (observed count + alpha × overall share)
                    / (bettor's total bets + alpha)

   Shorter histories receive more adjustment toward the overall
   preference. We tested alpha values of 5, 10, and 20.

2. Feature-group balancing:
   We divide features into activity/diversity, stake, odds,
   sport preferences, and bet-type preferences. Each standardized
   feature is multiplied by 1 / sqrt(number of features in its group).

Financial outcomes are excluded from this process.

## Evaluation

We compare the baseline, preference adjustment alone, group
balancing alone, and their combination. We also repeat random seeds
and resample each bettor's recorded bets five times to examine
sensitivity to their observed history.

The baseline reproduces the original final-analysis memberships.

## Findings and limitations

The combined method changes the segmentation substantially.
Group balancing accounts for most of that change.

Mean agreement under history resampling was approximately 0.179
for the baseline and 0.274 for the combined method. Both indicate
weak robustness. Five resamples do not establish a reliable improvement.

Silhouette scores measured in different feature spaces are not
directly comparable. We also evaluate all assignments in the same
baseline feature space.

This is a project-specific adaptation of established methods,
not a claim that we invented a new clustering algorithm.
The simulated data do not establish real-world bettor types,
betting skill, or future profitability.

Original financial results still refer to the ORIGINAL baseline
groups, not the new method's groups.

## Run the extension

From the project folder:

python contribution_analysis.py

Results are saved in outputs/contribution/.