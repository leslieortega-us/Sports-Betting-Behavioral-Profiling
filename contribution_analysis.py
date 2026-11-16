"""Compare history-adjusted preferences and balanced feature groups."""

from pathlib import Path
from dataclasses import dataclass
import json
import platform

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent

GROUPS = {
    "activity_and_diversity": ["total_bets", "sport_diversity"],
    "stake": ["average_stake", "stake_cv"],
    "odds": ["average_odds", "odds_variance"],
    "sport_preferences": [],
    "bet_type_preferences": ["bet_type_share_multiple"],
}


def behavior_features(bets, sports=None):
    """Create behavioral features without financial outcomes."""
    required = {
        "user_id", "bet_id", "stake", "odds", "sport", "bet_type"
    }

    if not required.issubset(bets):
        raise ValueError("Missing behavioral columns")

    if bets[list(required)].isna().any().any():
        raise ValueError("Missing behavioral values")

    if (bets.stake <= 0).any() or (bets.odds <= 1).any():
        raise ValueError(
            "Stakes must be positive and decimal odds above one"
        )

    features = bets.groupby("user_id").agg(
        total_bets=("bet_id", "size"),
        average_stake=("stake", "mean"),
        stake_std=("stake", "std"),
        average_odds=("odds", "mean"),
        odds_variance=("odds", "var"),
        sport_diversity=("sport", "nunique"),
    )

    if (features.total_bets < 2).any():
        raise ValueError(
            "At least two recorded bets per bettor are required"
        )

    features["stake_cv"] = (
        features.pop("stake_std") / features.average_stake
    )

    sport_shares = pd.crosstab(
        bets.user_id, bets.sport, normalize="index"
    )

    if sports is not None:
        if not set(sport_shares.columns).issubset(sports):
            raise ValueError("Unseen sport outside the fitted schema")

        sport_shares = sport_shares.reindex(
            columns=sports, fill_value=0
        )

    features = features.join(
        sport_shares.add_prefix("sport_share_")
    )

    features["bet_type_share_multiple"] = (
        bets.bet_type.eq("multiple")
        .groupby(bets.user_id)
        .mean()
    )

    return features


@dataclass
class Representation:
    strength: float = 10.0
    balance_groups: bool = True

    def fit(self, features):
        if self.strength < 0:
            raise ValueError("Shrinkage strength cannot be negative")

        self.columns_ = list(features.columns)

        self.sport_columns_ = [
            column for column in self.columns_
            if column.startswith("sport_share_")
        ]

        if not self.sport_columns_:
            raise ValueError("Sport preference features are required")

        self.preference_columns_ = (
            self.sport_columns_ + ["bet_type_share_multiple"]
        )

        counts = features.total_bets

        # Aggregate preferences, weighted by recorded bet counts.
        self.prior_ = (
            features[self.preference_columns_]
            .mul(counts, axis=0)
            .sum()
            / counts.sum()
        )

        self.scaler_ = StandardScaler().fit(
            self._adjust(features)
        )

        groups = dict(GROUPS)
        groups["sport_preferences"] = self.sport_columns_

        assigned_columns = [
            column
            for columns in groups.values()
            for column in columns
        ]

        if (
            len(assigned_columns) != len(set(assigned_columns))
            or set(assigned_columns) != set(self.columns_)
        ):
            raise ValueError(
                "Feature groups must partition the model features"
            )

        self.weights_ = pd.Series(
            1.0, index=self.columns_
        )

        if self.balance_groups:
            for columns in groups.values():
                self.weights_.loc[columns] = (
                    1 / np.sqrt(len(columns))
                )

        self.groups_ = groups
        return self

    def _adjust(self, features):
        if set(features.columns) != set(self.columns_):
            raise ValueError(
                "Feature schema differs from fitted representation"
            )

        adjusted = features[self.columns_].copy()
        counts = adjusted.total_bets

        # Shorter histories receive more pull toward the prior.
        adjusted[self.preference_columns_] = (
            adjusted[self.preference_columns_]
            .mul(counts, axis=0)
            + self.strength * self.prior_
        ).div(counts + self.strength, axis=0)

        log_columns = ["average_stake", "odds_variance"]
        adjusted[log_columns] = np.log1p(
            adjusted[log_columns]
        )

        if not np.isfinite(adjusted.to_numpy()).all():
            raise ValueError("Non-finite transformed features")

        return adjusted

    def transform(self, features):
        standardized = self.scaler_.transform(
            self._adjust(features)
        )

        return standardized * self.weights_.to_numpy()


def order_labels(labels, features):
    """Order group IDs by increasing mean stake."""
    ordering = (
        features.assign(cluster=labels)
        .groupby("cluster")
        .average_stake.mean()
        .sort_values()
        .index
    )

    mapping = {
        cluster: position
        for position, cluster in enumerate(ordering)
    }

    return np.array([mapping[label] for label in labels])


def bootstrap_histories(bets, rng):
    """Resample each bettor's records, preserving history length."""
    ordered = (
        bets.sort_values("user_id", kind="stable")
        .reset_index(drop=True)
    )

    sizes = (
        ordered.groupby("user_id", sort=True)
        .size()
        .to_numpy()
    )

    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    lower_bounds = np.repeat(starts, sizes)
    widths = np.repeat(sizes, sizes)

    positions = (
        lower_bounds
        + (rng.random(len(ordered)) * widths).astype(int)
    )

    return ordered.iloc[positions].copy()


def main():
    output_dir = ROOT / "outputs" / "contribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    bets = pd.read_csv(
        ROOT / "data" / "bets.csv", sep=";"
    )

    # Outcomes are excluded before fitting or resampling.
    behavioral = bets[
        ["user_id", "bet_id", "stake", "odds", "sport", "bet_type"]
    ].copy()

    features = behavior_features(behavioral)
    sports = sorted(behavioral.sport.unique())

    specifications = {
        "baseline": (0, False),
        "shrinkage_only": (10, False),
        "group_balance_only": (0, True),
        "combined_strength_10": (10, True),
        "combined_strength_5": (5, True),
        "combined_strength_20": (20, True),
    }

    representations = {}
    assignments = {}
    summaries = []
    seed_results = []

    reference_representation = Representation(
        0, False
    ).fit(features)

    reference_space = (
        reference_representation.transform(features)
    )

    for name, (strength, balance) in specifications.items():
        representation = Representation(
            strength, balance
        ).fit(features)

        transformed = representation.transform(features)

        model = KMeans(
            n_clusters=2,
            n_init=50,
            random_state=42,
        ).fit(transformed)

        labels = order_labels(model.labels_, features)

        representations[name] = representation
        assignments[name] = labels

        for seed in [0, 1, 2, 3, 42]:
            repeated_labels = KMeans(
                n_clusters=2,
                n_init=50,
                random_state=seed,
            ).fit_predict(transformed)

            seed_results.append({
                "variant": name,
                "seed": seed,
                "ari_to_seed42": adjusted_rand_score(
                    labels, repeated_labels
                ),
            })

        summaries.append({
            "variant": name,
            "strength": strength,
            "balanced_groups": balance,
            "silhouette_own_space": silhouette_score(
                transformed,
                labels,
                sample_size=2000,
                random_state=42,
            ),
            "silhouette_reference_space": silhouette_score(
                reference_space,
                labels,
                sample_size=2000,
                random_state=42,
            ),
            "ari_to_baseline": adjusted_rand_score(
                assignments["baseline"], labels
            ),
            "lower_stake_group": int((labels == 0).sum()),
            "higher_stake_group": int((labels == 1).sum()),
        })

        print("Fitted", name, flush=True)

    # Check agreement with the original final analysis.
    existing = pd.read_csv(
        ROOT / "outputs" / "final"
        / "bettor_features_and_clusters.csv",
        index_col="user_id",
    )

    baseline_agreement = adjusted_rand_score(
        existing.reindex(features.index).cluster,
        assignments["baseline"],
    )

    if baseline_agreement < 0.999:
        raise ValueError(
            "Baseline did not reproduce existing memberships; "
            "investigate before reporting"
        )

    history_results = []
    rng = np.random.default_rng(42)

    for repeat in range(1, 6):
        resampled = bootstrap_histories(behavioral, rng)

        resampled_features = (
            behavior_features(resampled, sports)
            .reindex(features.index)
        )

        for name, (strength, balance) in specifications.items():
            representation = Representation(
                strength, balance
            ).fit(resampled_features)

            transformed = representation.transform(
                resampled_features
            )

            labels = KMeans(
                n_clusters=2,
                n_init=50,
                random_state=42,
            ).fit_predict(transformed)

            history_results.append({
                "variant": name,
                "repeat": repeat,
                "history_bootstrap_ari": adjusted_rand_score(
                    assignments[name], labels
                ),
            })

        print(
            "Completed history bootstrap",
            repeat,
            "of 5",
            flush=True,
        )

    history_results = pd.DataFrame(history_results)
    summary = pd.DataFrame(summaries).set_index("variant")

    history_summary = (
        history_results.groupby("variant")
        .history_bootstrap_ari.agg(
            history_ari_mean="mean",
            history_ari_min="min",
            history_ari_max="max",
        )
    )

    summary = summary.join(history_summary)

    summary.to_csv(output_dir / "method_ablation.csv")

    history_results.to_csv(
        output_dir / "history_stability.csv",
        index=False,
    )

    pd.DataFrame(seed_results).to_csv(
        output_dir / "seed_stability.csv",
        index=False,
    )

    pd.DataFrame(
        assignments, index=features.index
    ).rename_axis("user_id").to_csv(
        output_dir / "bettor_assignments.csv"
    )

    for name in ["baseline", "combined_strength_10"]:
        profile = (
            features.assign(cluster=assignments[name])
            .groupby("cluster")
            .mean()
        )

        profile.insert(
            0, "bettors", np.bincount(assignments[name])
        )

        profile.to_csv(
            output_dir / f"{name}_profiles.csv"
        )

    representation = representations["combined_strength_10"]

    weight_rows = [
        {
            "group": group,
            "feature": column,
            "coordinate_multiplier": (
                representation.weights_[column]
            ),
        }
        for group, columns in representation.groups_.items()
        for column in columns
    ]

    pd.DataFrame(weight_rows).to_csv(
        output_dir / "feature_weights.csv",
        index=False,
    )

    pd.DataFrame({
        "prior_share": representation.prior_
    }).to_csv(
        output_dir / "fitted_preference_priors.csv"
    )

    metadata = {
        "python": platform.python_version(),
        "k": 2,
        "n_init": 50,
        "random_seed": 42,
        "history_bootstrap_repeats": 5,
        "principal_strength": 10,
        "strength_sensitivity": [5, 20],
        "history_stability_interpretation": (
            "Descriptive perturbation diagnostic; "
            "not predictive validation or a confidence interval."
        ),
        "claim": (
            "Project-specific adaptation of established shrinkage "
            "and feature weighting; no field-level novelty claimed."
        ),
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(summary.round(4).to_string(), flush=True)
    print(
        "Saved contribution results to",
        output_dir,
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()