"""
generate_figures.py

Utility script to generate synthetic figures for the
crypto-mining traffic detection case study.

All data in these plots is synthetic and does NOT represent
real client data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def ensure_figures_dir(path: str = "figures") -> str:
    """Ensure that the figures folder exists and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def create_synthetic_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset with four numerical features and a binary label."""
    np.random.seed(seed)

    # Synthetic features
    feature_1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    feature_2 = 0.6 * feature_1 + np.random.normal(loc=0.0, scale=0.7, size=n_samples)
    feature_3 = np.random.normal(loc=2.0, scale=1.5, size=n_samples)
    feature_4 = 0.3 * (feature_1 ** 2) + np.random.normal(loc=0.0, scale=0.5, size=n_samples)

    df = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": feature_4,
    })

    # Synthetic binary label (e.g. benign vs. mining)
    threshold = 0.5
    df["label"] = (df["feature_1"] + df["feature_2"] > threshold).astype(int)

    return df


def plot_performance_bars(figures_dir: str) -> None:
    """Create a bar plot with illustrative ROC-AUC values and save it as PNG."""
    # Synthetic ROC-AUC values for illustration
    datasets = ["Trainset", "Testset 1", "Testset 2"]
    roc_auc = [0.98, 0.94, 0.90]  # Example values for illustration only

    x = np.arange(len(datasets))

    plt.figure(figsize=(6, 4))

    plt.bar(x, roc_auc)

    plt.xticks(x, datasets, rotation=15)
    plt.ylim(0.8, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title("Illustrative model performance across datasets")

    # Add value labels above bars
    for i, value in enumerate(roc_auc):
        plt.text(
            i,
            value + 0.005,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    out_path = os.path.join(figures_dir, "performance_bar_roc_auc.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved performance bar plot to {out_path}")


def plot_scatter_matrix(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Create a scatter matrix for the numerical features and save it as PNG.
    The diagonal shows overlapping histograms for the two label classes.
    """
    features = ["feature_1", "feature_2", "feature_3", "feature_4"]
    labels = df["label"]

    # Map labels to colors (0 = benign, 1 = mining)
    color_map = labels.map({0: "tab:blue", 1: "tab:orange"})

    # Create scatter matrix (no extra plt.figure() here!)
    axes = scatter_matrix(
        df[features],
        figsize=(8, 8),
        diagonal="hist",
        alpha=0.6,
        c=color_map,
    )

    fig = axes[0, 0].get_figure()

    # Custom diagonal: two transparent histograms per feature
    for i, feature in enumerate(features):
        ax = axes[i, i]
        ax.clear()  # remove default diagonal histogram

        data_0 = df.loc[labels == 0, feature]
        data_1 = df.loc[labels == 1, feature]

        ax.hist(
            data_0,
            bins=20,
            alpha=0.5,
            label="Benign flows",
        )
        ax.hist(
            data_1,
            bins=20,
            alpha=0.5,
            label="Mining flows",
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

        # Legend only once
        if i == 0:
            ax.legend()

    fig.suptitle("Pairwise feature relationships (synthetic data)", fontsize=14)
    fig.subplots_adjust(top=0.92)

    out_path = os.path.join(figures_dir, "feature_scatter_matrix.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved scatter matrix to {out_path}")


def main() -> None:
    """Entry point to generate figures for the case study."""
    figures_dir = ensure_figures_dir()
    df = create_synthetic_dataset()

    plot_performance_bars(figures_dir)
    plot_scatter_matrix(df, figures_dir)


if __name__ == "__main__":
    main()
