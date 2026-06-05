"""Generate the discovery EDA notebook."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/eda_discovery.ipynb")


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    nb["cells"] = [
        md(
            """
            # Stellar Class Discovery EDA

            This notebook explores the Kaggle stellar-classification dataset with one purpose:
            find new signal that can move balanced accuracy beyond the current public-best
            spatial candidates.

            The flow starts broad, then narrows to known hard regions: class imbalance,
            redshift overlap, low-redshift GALAXY/STAR ambiguity, train/test shift, and
            sky-position neighbourhood structure.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.metrics import balanced_accuracy_score
            from sklearn.model_selection import StratifiedKFold
            from sklearn.neighbors import NearestNeighbors

            PROJECT_ROOT = Path.cwd()
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.data import CLASS_LABELS, build_features, load_raw  # noqa: E402
            from src.features import PHOTOMETRIC_COLUMNS  # noqa: E402
            from src.spatial import radec_to_xyz  # noqa: E402

            RANDOM_STATE = 42
            PLOT_SAMPLE_ROWS = 80_000
            SPATIAL_SAMPLE_ROWS = 120_000
            NN_SAMPLE_ROWS = 30_000

            pd.set_option("display.max_columns", 120)
            plt.style.use("default")
            """
        ),
        code(
            """
            def sample_frame(df: pd.DataFrame, n: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
                if len(df) <= n:
                    return df.copy()
                return df.sample(n=n, random_state=random_state).copy()


            def describe_train_test_shift(train_s: pd.Series, test_s: pd.Series) -> pd.Series:
                return pd.Series(
                    {
                        "train_mean": train_s.mean(),
                        "test_mean": test_s.mean(),
                        "mean_delta": test_s.mean() - train_s.mean(),
                        "train_std": train_s.std(),
                        "test_std": test_s.std(),
                        "train_p01": train_s.quantile(0.01),
                        "test_p01": test_s.quantile(0.01),
                        "train_p50": train_s.quantile(0.50),
                        "test_p50": test_s.quantile(0.50),
                        "train_p99": train_s.quantile(0.99),
                        "test_p99": test_s.quantile(0.99),
                    }
                )


            def plot_hist_by_class(
                df: pd.DataFrame,
                column: str,
                bins: int = 80,
                clip_quantiles: tuple[float, float] | None = (0.005, 0.995),
            ) -> None:
                fig, ax = plt.subplots(figsize=(9, 4))
                values = df[column]
                plot_df = df
                if clip_quantiles is not None:
                    lo, hi = values.quantile(list(clip_quantiles))
                    plot_df = df[(values >= lo) & (values <= hi)]
                for label in CLASS_LABELS:
                    subset = plot_df.loc[plot_df["class"] == label, column]
                    ax.hist(subset, bins=bins, alpha=0.45, density=True, label=label)
                ax.set_title(f"{column} distribution by class")
                ax.set_xlabel(column)
                ax.set_ylabel("density")
                ax.legend()
                plt.show()
            """
        ),
        md(
            """
            ## 1. Load Data And Validate Schema

            The first cell loads the three Kaggle CSVs and checks the submission contract:
            train has labels, test does not, and sample-submission ids align exactly with test ids.
            """
        ),
        code(
            """
            train, test, sample_submission = load_raw()
            X_train, y_train, categorical_columns, encoder = build_features(train)
            X_test, _, _, _ = build_features(test, label_encoder=encoder)

            print(f"train: {train.shape}")
            print(f"test: {test.shape}")
            print(f"sample_submission: {sample_submission.shape}")
            print(f"feature matrix: train={X_train.shape}, test={X_test.shape}")
            print(f"categorical columns: {categorical_columns}")

            required_train = {"id", "alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"}
            required_test = required_train - {"class"}
            assert required_train.issubset(train.columns)
            assert required_test.issubset(test.columns)
            assert sample_submission["id"].equals(test["id"])
            assert set(train["class"]) == set(CLASS_LABELS)
            assert not train.isna().any().any()
            assert not test.isna().any().any()
            """
        ),
        code(
            """
            schema = pd.DataFrame(
                {
                    "train_dtype": train.dtypes.astype(str),
                    "test_dtype": test.dtypes.astype(str),
                    "train_missing": train.isna().sum(),
                    "test_missing": test.isna().sum(),
                    "train_unique": train.nunique(dropna=False),
                    "test_unique": test.nunique(dropna=False),
                }
            )
            schema
            """
        ),
        md(
            """
            ## 2. Target Balance And Metric Implications

            The competition metric is balanced accuracy, so minority-class recall matters as much
            as majority-class recall. This section makes the class mix explicit before looking at
            feature relationships.
            """
        ),
        code(
            """
            class_counts = train["class"].value_counts().reindex(CLASS_LABELS)
            class_summary = pd.DataFrame(
                {
                    "count": class_counts,
                    "share": class_counts / len(train),
                    "balanced_accuracy_weight": 1 / len(CLASS_LABELS),
                }
            )
            class_summary
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            class_counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
            ax.set_title("Training target distribution")
            ax.set_ylabel("rows")
            ax.set_xlabel("class")
            plt.xticks(rotation=0)
            plt.show()
            """
        ),
        md(
            """
            ## 3. Numeric Feature Distributions By Class

            These plots show which raw numeric features have separable class structure and where
            the distributions overlap. Overlap regions are candidate targets for more local or
            interaction-heavy features.
            """
        ),
        code(
            """
            numeric_columns = ["alpha", "delta", *PHOTOMETRIC_COLUMNS, "redshift"]
            train_plot = sample_frame(train, PLOT_SAMPLE_ROWS)

            summary_by_class = train.groupby("class")[numeric_columns].agg(["mean", "std", "median"])
            summary_by_class
            """
        ),
        code(
            """
            for column in numeric_columns:
                plot_hist_by_class(train_plot, column)
            """
        ),
        md(
            """
            ## 4. Redshift Overlap And Low-Redshift Ambiguity

            Redshift is the dominant single feature, but the hard cases live where class ranges
            overlap. The low-redshift region is especially important because STAR and nearby
            GALAXY examples can become hard to separate from tabular features alone.
            """
        ),
        code(
            """
            redshift_bins = pd.IntervalIndex.from_tuples(
                [(-0.01, 0.02), (0.02, 0.08), (0.08, 0.15), (0.15, 0.35), (0.35, 0.75), (0.75, 1.5), (1.5, 8.0)]
            )
            redshift_table = (
                train.assign(redshift_bin=pd.cut(train["redshift"], bins=redshift_bins))
                .pivot_table(index="redshift_bin", columns="class", values="id", aggfunc="count", fill_value=0)
                .reindex(columns=CLASS_LABELS)
            )
            redshift_share = redshift_table.div(redshift_table.sum(axis=1), axis=0)
            redshift_table.join(redshift_share, rsuffix="_share")
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(9, 5))
            redshift_share.plot(kind="bar", stacked=True, ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
            ax.set_title("Class mix by redshift bin")
            ax.set_xlabel("redshift bin")
            ax.set_ylabel("class share")
            ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            low_z = train[train["redshift"].between(-0.01, 0.15)].copy()
            print(f"low-redshift rows: {len(low_z):,} ({len(low_z) / len(train):.1%} of train)")
            low_z["class"].value_counts(normalize=True).reindex(CLASS_LABELS).to_frame("share")
            """
        ),
        md(
            """
            ## 5. Photometric Color And Magnitude Relationships

            Color indices often expose shape that raw magnitudes hide. These cells inspect both
            existing engineered color features and two-dimensional views around the low-redshift
            ambiguity zone.
            """
        ),
        code(
            """
            color_columns = ["u_g", "g_r", "r_i", "i_z", "u_r", "u_z", "g_z", "r_z"]
            color_df = train[["id", "class", "redshift"]].join(X_train[color_columns])
            color_df.groupby("class")[color_columns].agg(["mean", "std", "median"])
            """
        ),
        code(
            """
            color_plot = sample_frame(color_df, PLOT_SAMPLE_ROWS)
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            pairs = [("u_g", "g_r"), ("g_r", "r_i"), ("u_r", "r_z"), ("u_z", "redshift")]
            colors = {"GALAXY": "#4c78a8", "QSO": "#f58518", "STAR": "#54a24b"}
            for ax, (x_col, y_col) in zip(axes.ravel(), pairs, strict=True):
                for label in CLASS_LABELS:
                    subset = color_plot[color_plot["class"] == label]
                    ax.scatter(subset[x_col], subset[y_col], s=2, alpha=0.18, label=label, c=colors[label])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col}")
            axes[0, 0].legend(markerscale=4)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            low_z_color = sample_frame(color_df[color_df["redshift"].between(-0.01, 0.15)], PLOT_SAMPLE_ROWS)
            fig, ax = plt.subplots(figsize=(7, 6))
            for label in CLASS_LABELS:
                subset = low_z_color[low_z_color["class"] == label]
                ax.scatter(subset["u_g"], subset["g_r"], s=3, alpha=0.2, label=label, c=colors[label])
            ax.set_title("Low-redshift color overlap")
            ax.set_xlabel("u_g")
            ax.set_ylabel("g_r")
            ax.legend(markerscale=4)
            plt.show()
            """
        ),
        md(
            """
            ## 6. Categorical Signal

            The categorical fields are useful but not label leaks. This section inspects class mix
            by category and by the engineered `spectral_population` interaction.
            """
        ),
        code(
            """
            categorical_source = train[["class", "spectral_type", "galaxy_population"]].join(
                X_train[["spectral_population"]]
            )
            for column in ["spectral_type", "galaxy_population", "spectral_population"]:
                counts = pd.crosstab(categorical_source[column], categorical_source["class"]).reindex(columns=CLASS_LABELS)
                display(counts)
                display(counts.div(counts.sum(axis=1), axis=0))
            """
        ),
        md(
            """
            ## 7. Train/Test Shift Checks

            Public score gaps often come from distribution mismatch. These checks compare train
            and test numerics, categorical levels, and sky coverage before proposing new features.
            """
        ),
        code(
            """
            shift_table = pd.DataFrame(
                {
                    column: describe_train_test_shift(train[column], test[column])
                    for column in numeric_columns
                }
            ).T
            shift_table["abs_mean_delta_over_train_std"] = (
                shift_table["mean_delta"].abs() / shift_table["train_std"].replace(0, np.nan)
            )
            shift_table.sort_values("abs_mean_delta_over_train_std", ascending=False)
            """
        ),
        code(
            """
            for column in ["spectral_type", "galaxy_population"]:
                train_share = train[column].value_counts(normalize=True).rename("train_share")
                test_share = test[column].value_counts(normalize=True).rename("test_share")
                display(pd.concat([train_share, test_share], axis=1).fillna(0).assign(delta=lambda x: x["test_share"] - x["train_share"]))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
            sample_frame(train, SPATIAL_SAMPLE_ROWS).plot.scatter(
                x="alpha", y="delta", s=1, alpha=0.15, ax=axes[0], title="Train sky coverage"
            )
            sample_frame(test, SPATIAL_SAMPLE_ROWS).plot.scatter(
                x="alpha", y="delta", s=1, alpha=0.15, ax=axes[1], title="Test sky coverage", color="#f58518"
            )
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 8. Spatial Structure Discovery

            The strongest known lift came from spatial-neighbour features. These cells make the
            structure visible and provide a cheap nearest-neighbour class-agreement diagnostic.
            """
        ),
        code(
            """
            spatial_plot = sample_frame(train, SPATIAL_SAMPLE_ROWS)
            fig, ax = plt.subplots(figsize=(9, 5))
            for label in CLASS_LABELS:
                subset = spatial_plot[spatial_plot["class"] == label]
                ax.scatter(subset["alpha"], subset["delta"], s=1, alpha=0.18, label=label, c=colors[label])
            ax.set_title("Training labels in sky position")
            ax.set_xlabel("alpha")
            ax.set_ylabel("delta")
            ax.legend(markerscale=5)
            plt.show()
            """
        ),
        code(
            """
            nn_sample = sample_frame(train, NN_SAMPLE_ROWS)
            xyz = radec_to_xyz(nn_sample["alpha"].to_numpy(), nn_sample["delta"].to_numpy())
            labels = encoder.transform(nn_sample["class"])
            nn = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(xyz)
            distances, indices = nn.kneighbors(xyz)
            neighbour_labels = labels[indices[:, 1:]]

            rows = []
            for k in [1, 3, 5, 10]:
                same_class = (neighbour_labels[:, :k] == labels[:, None]).mean()
                majority_vote = np.apply_along_axis(
                    lambda row: np.bincount(row, minlength=len(CLASS_LABELS)).argmax(),
                    1,
                    neighbour_labels[:, :k],
                )
                rows.append(
                    {
                        "k": k,
                        "same_class_pair_rate": same_class,
                        "majority_vote_balanced_accuracy": balanced_accuracy_score(labels, majority_vote),
                        "median_distance": np.median(distances[:, k]),
                    }
                )

            pd.DataFrame(rows)
            """
        ),
        code(
            """
            fold_ids = np.zeros(len(nn_sample), dtype=int)
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            for fold, (_, valid_idx) in enumerate(splitter.split(nn_sample, labels)):
                fold_ids[valid_idx] = fold
            pd.Series(fold_ids).value_counts().sort_index().to_frame("rows_per_fold")
            """
        ),
        md(
            """
            ## 9. Residual Analysis Hooks

            This section reads existing experiment artifacts when they are present. It keeps the
            notebook useful as an EDA artifact without requiring every generated probability cache.
            """
        ),
        code(
            """
            experiment_paths = [
                Path("experiments/16_spatial_xgb.json"),
                Path("experiments/19_loo_spatial_final.json"),
                Path("experiments/23_loo_spatial_star_tilt.json"),
                Path("experiments/26_loo_spatial_xgb_calibrated.json"),
            ]
            experiment_records = []
            for path in experiment_paths:
                if not path.exists():
                    continue
                with path.open() as handle:
                    payload = json.load(handle)
                experiment_records.append(
                    {
                        "path": str(path),
                        "oof_balanced_accuracy": payload.get("oof_balanced_accuracy")
                        or payload.get("tuned_oof_balanced_accuracy"),
                        "public_score": payload.get("public_score"),
                        "submission_path": payload.get("submission_path"),
                    }
                )
            pd.DataFrame(experiment_records)
            """
        ),
        code(
            """
            submission_paths = [
                Path("submissions/16_spatial_blend.csv"),
                Path("submissions/19_loo_spatial_final.csv"),
                Path("submissions/23_loo_spatial_star_tilt.csv"),
                Path("submissions/26_loo_spatial_xgb_calibrated.csv"),
            ]
            submission_counts = []
            for path in submission_paths:
                if path.exists():
                    counts = pd.read_csv(path)["class"].value_counts().reindex(CLASS_LABELS, fill_value=0)
                    submission_counts.append(counts.rename(path.name))
            if submission_counts:
                submission_count_table = pd.concat(submission_counts, axis=1).T
                display(submission_count_table)
            """
        ),
        code(
            """
            if Path("experiments/16_spatial_xgb.json").exists():
                with Path("experiments/16_spatial_xgb.json").open() as handle:
                    spatial_payload = json.load(handle)
                keys_of_interest = [
                    "tuned_oof_balanced_accuracy",
                    "oof_balanced_accuracy",
                    "multipliers",
                    "blend_weights",
                    "per_class_recall",
                ]
                {key: spatial_payload.get(key) for key in keys_of_interest if key in spatial_payload}
            """
        ),
        md(
            """
            ## 10. Discovery Hypotheses And Next Experiments

            This final section translates the EDA into experiment ideas. Keep this list updated
            after inspecting notebook outputs, especially the spatial and low-redshift cells.

            1. Build local photometric-neighbour features: nearest neighbours in sky position
               plus color/magnitude similarity, not raw redshift distance.
            2. Try logit-space blending for spatial model probabilities before multiplier tuning.
            3. Inspect low-redshift GALAXY/STAR regions for class-conditional color pockets.
            4. Compare final-only LOO spatial variants against honest OOF spatial features to
               isolate train/test spatial-density mismatch.
            5. Add a small residual report for rows where spatial models disagree with the
               strongest non-spatial model family.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
