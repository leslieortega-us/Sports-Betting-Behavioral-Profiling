# Sports Betting Behavioral Profiling

## Research Question
Can simulated sports betting records reveal distinct behavioral profiles,
and how does observed betting performance vary across those profiles?

## Dataset
Source: Sports Betting Profiling Dataset by Emilien Coicaud on Kaggle.
https://www.kaggle.com/datasets/emiliencoicaud/sports-betting-profiling-dataset

The dataset contains 100,000 simulated bets from 5,000 bettors.
The semicolon-separated CSV is stored in `data/bets.csv`.

## Phase 2 Progress
- Inspected data types, missing values, duplicates, stakes, and odds.
- Explored stake, odds, sport, and bet-type distributions.
- Created 23 behavioral features per bettor.
- Prepared and standardized 20 features for clustering.
- Compared K-means models with 2–8 clusters.
- Checked initialization stability and visualized clusters using PCA.

Financial outcomes were excluded from clustering.

## Preliminary Findings
Two clusters had the highest sampled silhouette score, approximately
0.0829, indicating weak separation. The groups primarily differ in
average stake and relative stake variability.

With 50 initializations, pairwise adjusted Rand indices across five
random seeds ranged from 0.997 to 1.000. The first two PCA components
explain 19.9% of total variance.

These results describe simulated data and do not establish bettor skill
or future profitability.

## How to Run
1. Install Python and the VS Code Python and Jupyter extensions.
2. Install the required packages in your selected Python environment:

   ```bash
   python -m pip install pandas numpy matplotlib scikit-learn ipykernel
   ```

3. Open the project folder in VS Code.
4. Open `phase2_analysis.ipynb` and select that Python environment.
5. Ensure the notebook is beside `README.md` and the `data` folder.
6. Restart the kernel and run all cells in order.

The notebook creates an `outputs` folder containing feature tables,
cluster assignments, evaluation results, cluster summaries, and a PCA plot.

## Remaining Work
- Compare hierarchical clustering with K-means.
- Assess sensitivity to feature selection and scaling.
- Verify payout definitions and compare financial performance.
- Complete the final report and presentation.