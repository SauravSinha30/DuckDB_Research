import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"


def plot_layout_latency(df: pd.DataFrame) -> None:
    layouts = ["random", "date_sorted", "city_sorted", "lex_date_city", "zorder_date_city"]
    queries = ["date_filter", "city_filter", "joint_filter", "full_aggregation"]
    labels = ["Random", "Date Sorted", "City Sorted", "Lex (Date, City)", "Z-Order (Date, City)"]

    x = np.arange(len(queries))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (layout, label) in enumerate(zip(layouts, labels)):
        sub = df[df["layout"] == layout].set_index("query").loc[queries]
        med_ms = sub["median_time_sec"].values * 1000.0
        std_ms = sub["std_dev_sec"].values * 1000.0
        ax.bar(
            x + (i - 2) * width,
            med_ms,
            width,
            yerr=std_ms,
            capsize=3,
            label=label,
        )

    ax.set_ylabel("Median Query Latency (ms)")
    ax.set_title("DuckDB Query Latency by Parquet Physical Layout (10M Rows, RG=50K)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Date Filter (~4.1%)", "City Filter (10%)", "Joint Filter (~0.41%)", "Full Aggregation"])
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/layout_query_latency.png", dpi=180)
    plt.close(fig)


def plot_row_group_sweep(df: pd.DataFrame) -> None:
    date_df = df[df["query"] == "date_filter"].sort_values("row_group_size")
    agg_df = df[df["query"] == "full_aggregation"].sort_values("row_group_size")

    x_labels = [f"{int(v):,}" for v in date_df["row_group_size"]]
    x = np.arange(len(x_labels))

    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    ax1.plot(x, date_df["median_time_sec"] * 1000.0, marker="o", linewidth=2, label="Date Filter Latency (ms)")
    ax1.plot(x, agg_df["median_time_sec"] * 1000.0, marker="s", linewidth=2, label="Full Aggregation Latency (ms)")
    ax1.set_xlabel("Parquet ROW_GROUP_SIZE (rows)")
    ax1.set_ylabel("Median Query Latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, date_df["file_size_MB"], marker="^", linestyle="--", color="tab:green", linewidth=2, label="File Size (MB)")
    ax2.set_ylabel("Parquet File Size (MB)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", frameon=True)
    ax1.set_title("Row-Group Size Trade-off: Query Latency vs. Compression (10M Rows, date_sorted)")

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/row_group_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_roi_breakeven(df: pd.DataFrame) -> None:
    sub = df[df["query"] == "joint_filter"].copy()
    sub = sub[sub["layout"] != "random"]

    fig, ax = plt.subplots(figsize=(9, 5))
    q_counts = np.linspace(0, 1500, 200)

    random_row = df[(df["layout"] == "random") & (df["query"] == "joint_filter")].iloc[0]
    ax.plot(
        q_counts,
        random_row["build_time_sec"] + q_counts * random_row["median_time_sec"],
        linestyle="--",
        color="black",
        linewidth=2,
        label="Random (No Sort Overhead)",
    )

    for _, row in sub.iterrows():
        total_time = row["build_time_sec"] + q_counts * row["median_time_sec"]
        ax.plot(q_counts, total_time, linewidth=2, label=row["layout"])

    ax.set_xlabel("Number of Joint Filter Queries Executed")
    ax.set_ylabel("Cumulative Time: Build + Queries (seconds)")
    ax.set_title("Sorting Cost Amortization (ROI) on 10M Rows — Joint Filter Workload")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/roi_break_even.png", dpi=180)
    plt.close(fig)


def plot_skew_hilbert(df: pd.DataFrame) -> None:
    sub = df[df["cardinality"] == 1024].copy()
    queries = ["date_filter", "hot_entity", "cold_entity", "joint_hot"]
    layouts = ["date_sorted", "lex_date_entity", "zorder", "hilbert"]
    labels = ["Date Sorted", "Lex (Date, Entity)", "Z-Order", "Hilbert"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, skew in zip(axes, ["uniform", "heavy_skew"]):
        s_df = sub[sub["skew"] == skew]
        x = np.arange(len(queries))
        width = 0.18
        for i, (layout, label) in enumerate(zip(layouts, labels)):
            vals = (
                s_df[s_df["layout"] == layout]
                .set_index("query")
                .loc[queries]["median_ms"]
                .values
            )
            ax.bar(x + (i - 1.5) * width, vals, width, label=label)
        title_skew = "Uniform (0.1% hot)" if skew == "uniform" else "Heavy Skew (25% hot)"
        ax.set_title(f"C=1024, {title_skew}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Date", "Hot Entity", "Cold Entity", "Joint Hot"])
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Median Query Latency (ms)")
    axes[1].legend(frameon=True)
    fig.suptitle("Skew & Space-Filling Curves: Z-Order vs. Hilbert vs. Lex (10M Rows, C=1024)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/skew_hilbert_comparison.png", dpi=180)
    plt.close(fig)


def main():
    df1 = pd.read_csv(f"{RESULTS_DIR}/experiment1_layouts_10m.csv")
    df2 = pd.read_csv(f"{RESULTS_DIR}/experiment2_row_group_sweep_10m.csv")
    plot_layout_latency(df1)
    plot_row_group_sweep(df2)
    plot_roi_breakeven(df1)
    if os.path.exists(f"{RESULTS_DIR}/experiment3_skew_hilbert_10m.csv"):
        df3 = pd.read_csv(f"{RESULTS_DIR}/experiment3_skew_hilbert_10m.csv")
        plot_skew_hilbert(df3)
    print("Generated plots in results/.")


if __name__ == "__main__":
    main()
