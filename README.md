# Sports Betting Behavioral Profiling

**Research question:** Can simulated sports betting records reveal distinct behavioral profiles, and how does observed betting performance vary across those profiles?

We analyze [Kaggle's simulated sports betting dataset](https://www.kaggle.com/datasets/emiliencoicaud/sports-betting-profiling-dataset): 100,000 bets from 5,000 bettors. Clustering uses behavioral features; financial outcomes are evaluated separately.

## Main findings

- Baseline K-Means (k=2) separates lower- and higher-stake groups, with weak separation (sampled silhouette 0.0829).
- Membership is consistent across tested seeds (ARI 0.9968–1.0000), but depends on preprocessing choices.
- Ward hierarchical clustering gives a different split tied to rare cycling bets (291 versus 4,709 bettors), with ARI 0.0019 relative to K-Means.
- Both baseline groups lose on average. Win rates and ROI are similar; higher-stake bettors have larger monetary losses. This does not establish a difference in skill.
- Results describe synthetic records and do not establish real-world bettor types or predict future returns.

## Reproduce the final analysis

Use Python 3.12 (tested with 3.12.14). From the repository root:

```bash
python -m pip install -r requirements.txt
python final_analysis.py
```

Alternatively, open `final_analysis.ipynb` in VS Code, select that Python environment, and run all cells. It runs the same script and displays the generated tables and figures. The script resolves data paths relative to its own location.

The analysis checks the data and payout identities, builds features, compares K-Means and Ward for k=2–8, repeats seeds, checks five feature/scaling specifications, and computes bettor-level performance and exploratory bootstrap contrasts. Settings are fixed for reproducibility. All final outputs go to `outputs/final/`; exact tested versions and run settings are saved in `run_metadata.json`.

## Files

| File or folder | Purpose |
|---|---|
| `data/bets.csv` | Original semicolon-separated dataset |
| `phase2_analysis.ipynb` | Preserved mid-project analysis |
| `outputs/` (top level) | Preserved Phase 2 results |
| `final_analysis.py` | Complete reproducible final pipeline |
| `final_analysis.ipynb` | Notebook interface to the final pipeline |
| `outputs/final/` | Final comparison, sensitivity, profile, and performance results; figures; validation and run metadata |
| `reports/final_report.md` | Final analytical report draft |
| `reports/presentation_and_demo_outline.md` | Presentation outline, demo steps, and submission checklist |
| `requirements.txt` | Tested analysis dependencies and notebook kernel dependency |

## Interpretation notes

Final K-Means IDs are ordered by mean stake: group 0 is lower stake and group 1 is higher stake. This reverses the Phase 2 numeric IDs while preserving membership. Ward IDs are separate and should not be interpreted as the same profiles.

`gain` is gross payout including returned stake. Bettor net profit is `gain − stake`, equivalent to negative GGR. ROI is profit / stake; CSV rate fields are fractions. Mean bettor ROI and pooled ROI use different weighting. No currency is specified.

Silhouette is sampled on the same 2,000 bettors within the baseline model comparison. Sensitivity silhouettes use different geometries and should not be ranked as if they measured the same space. Bootstrap intervals condition on fitted memberships and are descriptive; no claims of real-world generalization, causality, or betting skill are supported.

## Remaining submission work

Review the report with your group and apply the actual final course rubric; add names and contributions, prepare the finished slides, record the demo, and complete any peer evaluations. The analysis alone does not complete these course submission steps.

## Additional Method Experiment

The extension adjusts preferences for short betting histories and
balances five behavioral feature groups before applying K-Means.

Run the original analysis first, then the extension:

```bash
python -m pip install -r requirements.txt
python final_analysis.py
python contribution_analysis.py
```

Original results are saved in `outputs/final/`.
Extension results are saved in `outputs/contribution/`.

See `TECHNICAL_CONTRIBUTION.md` for the method and limitations.
The original financial comparisons refer to the original groups.

## Sources Submission Folder

`Sources/` contains a standalone copy of the code, data, and results.
Open that folder and use the same commands above to reproduce the analysis.