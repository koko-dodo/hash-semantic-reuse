from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_parameter_tradeoff():
    df = pd.read_csv(RESULTS / "parameter_study.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for k, group in df.groupby("k"):
        group = group.sort_values("L")
        ax.plot(
            group["candidate_ratio"] * 100,
            group["mean_recall"] * 100,
            marker="o",
            label=f"k={k}"
        )

        for _, row in group.iterrows():
            ax.annotate(
                f"L={int(row['L'])}",
                (
                    row["candidate_ratio"] * 100,
                    row["mean_recall"] * 100
                ),
                xytext=(6, 8 if k == 12 else -12),
                textcoords="offset points",
                fontsize=8
            )

    ax.set_xlabel("Candidate Ratio (%)")
    ax.set_ylabel("Mean Recall (%)")
    ax.set_title("LSH: Recall vs. Candidate Ratio")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "parameter_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_scalability():
    df = pd.read_csv(RESULTS / "scalability.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        df["database_size"],
        df["avg_candidates"],
        marker="o",
        label="All candidates"
    )

    ax.plot(
        df["database_size"],
        df["avg_false_candidates"],
        marker="s",
        label="False candidates"
    )

    ax.set_xlabel("Database Size (N)")
    ax.set_ylabel("Average Candidate Count")
    ax.set_title("LSH Scalability (k=16, L=80)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "scalability.png", dpi=200)
    plt.close(fig)


def plot_baselines():
    df = pd.read_csv(RESULTS / "baseline_comparison.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(df["method"], df["mean_time_ms"])

    ax.set_ylabel("Mean Query Time (ms)")
    ax.set_title("Retrieval Latency Comparison")
    ax.tick_params(axis="x", labelrotation=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_latency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_parameter_tradeoff()
    plot_scalability()
    plot_baselines()
    print("All figures generated successfully.")