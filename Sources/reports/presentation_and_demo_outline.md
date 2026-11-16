# Final presentation and demo outline

This is a speaking outline, not a recorded demo or a finished slide deck. Adjust its length to the course rubric.

## Suggested slides

1. **Question and scope:** Can behavioral profiles be found, and do observed returns differ? State immediately that the data are simulated.
2. **Data and preparation:** 100,000 bets, 5,000 bettors; data checks; 23 initial features and 20 baseline clustering features. Explain why outcomes are excluded.
3. **Methods:** K-Means and Ward, k=2–8, shared standardized feature space; silhouette, ARI, sensitivity checks. Define the measures in plain language.
4. **Baseline profiles:** Lower versus higher average stake; 1,871 versus 3,129 bettors; weak silhouette 0.0829. Use `stake_profiles.png`.
5. **Alternative models and sensitivity:** Ward separates 291 cycling bettors; ARI 0.0019 versus K-Means. Log transforms matter. Use `method_comparison.png` and explain why the higher score alone is not sufficient.
6. **Observed performance:** Both groups lose on average; similar win rates/ROI; larger monetary losses for the higher-stake group. Use `performance_distributions.png`. Clarify gross payout versus profit.
7. **Answer and limitations:** Interpretable descriptive groups, limited evidence of distinct underlying types; simulated data, short histories, no timeline, no prediction or causal conclusions.
8. **Reproducibility:** Repository, exact run command, generated outputs, and contribution summary with actual group member roles added by the team.

## Demo sequence

1. Open README and show the research question and reproduction instructions.
2. Open `final_analysis.ipynb` in VS Code. Select the Python environment in which requirements were installed.
3. Restart the kernel and run all cells; show that all data checks passed.
4. Explain the model comparison and sensitivity tables.
5. Show the profile and performance tables, and walk through `net_profit = gain - stake` and `roi = net_profit / total_stake`.
6. Show the saved figures and explain one limitation of each result.
7. End with the qualified answer to the research question.

## Before submitting

- Check the instructor's final deliverable list, required length, file formats, and due date.
- Review the final report draft together; add names, actual contributions, and required citations/formatting.
- Create the final slide deck using the outline above.
- Record and watch the required demo video; verify audio, access permissions, and the submitted video link.
- Complete any individual peer evaluations.
- Ensure the submitted repository points to the final-analysis work and contains the correct artifacts.
