import os
import random
import time
import duckdb
import numpy as np
import pandas as pd

N = 10_000_000
DEFAULT_ROW_GROUP_SIZE = 50_000
WARMUP_RUNS = 3
BENCHMARK_ROUNDS = 15
TMP_DIR = "data/tmp_bench"
RESULTS_DIR = "results"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_synthetic_table(con: duckdb.DuckDBPyConnection, n: int) -> float:
    """Generate N synthetic sales rows directly in DuckDB SQL."""
    start = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TABLE sales AS
        WITH base AS (
            SELECT
                i,
                CAST(abs(hash(i * 31 + 1)) % 730 AS INTEGER) AS date_offset,
                CAST(abs(hash(i * 37 + 2)) % 10 AS INTEGER) AS city_id,
                CAST(abs(hash(i * 41 + 3)) % 4 AS INTEGER) AS prod_id,
                CAST(500 + (abs(hash(i * 43 + 4)) % 99500) AS INTEGER) AS amount
            FROM range({n}) AS t(i)
        )
        SELECT
            DATE '2024-01-01' + date_offset AS sale_date,
            date_offset,
            city_id,
            CASE city_id
                WHEN 0 THEN 'Patna'
                WHEN 1 THEN 'Delhi'
                WHEN 2 THEN 'Mumbai'
                WHEN 3 THEN 'Kolkata'
                WHEN 4 THEN 'Bangalore'
                WHEN 5 THEN 'Chennai'
                WHEN 6 THEN 'Pune'
                WHEN 7 THEN 'Hyderabad'
                WHEN 8 THEN 'Jaipur'
                ELSE 'Lucknow'
            END AS city,
            CASE prod_id
                WHEN 0 THEN 'Laptop'
                WHEN 1 THEN 'Phone'
                WHEN 2 THEN 'Tablet'
                ELSE 'Monitor'
            END AS product,
            amount
        FROM base
    """)
    return time.perf_counter() - start


def spread_bits_sql(expr: str) -> str:
    """Spread 10-bit integer into even bit positions for 2D Morton (Z-order) encoding."""
    x = f"CAST({expr} AS UINTEGER)"
    s1 = f"(({x} | ({x} << 8)) & 16711935)"     # 0x00FF00FF
    s2 = f"(({s1} | ({s1} << 4)) & 252645135)"  # 0x0F0F0F0F
    s3 = f"(({s2} | ({s2} << 2)) & 858993459)"  # 0x33333333
    s4 = f"(({s3} | ({s3} << 1)) & 1431655765)" # 0x55555555
    return s4


def count_pruned_row_groups(
    con: duckdb.DuckDBPyConnection, parquet_path: str, query_name: str
) -> tuple[int, int]:
    """
    Inspect Parquet row-group min/max zone maps to determine how many
    row groups overlap the query predicate vs. total row groups.
    """
    meta = con.execute(f"""
        SELECT row_group_id, path_in_schema, stats_min_value, stats_max_value
        FROM parquet_metadata('{parquet_path}')
        WHERE path_in_schema IN ('sale_date', 'city')
    """).fetchdf()

    total_rgs = int(meta["row_group_id"].nunique())

    if query_name == "full_aggregation":
        return total_rgs, total_rgs

    date_meta = meta[meta["path_in_schema"] == "sale_date"].set_index("row_group_id")
    city_meta = meta[meta["path_in_schema"] == "city"].set_index("row_group_id")

    scanned = 0
    for rg in range(total_rgs):
        date_match = True
        city_match = True

        if query_name in ("date_filter", "joint_filter"):
            d_min = str(date_meta.loc[rg, "stats_min_value"])
            d_max = str(date_meta.loc[rg, "stats_max_value"])
            # Overlap with ['2024-06-01', '2024-06-30']
            date_match = not (d_max < "2024-06-01" or d_min >= "2024-07-01")

        if query_name in ("city_filter", "joint_filter"):
            c_min = str(city_meta.loc[rg, "stats_min_value"])
            c_max = str(city_meta.loc[rg, "stats_max_value"])
            city_match = (c_min <= "Patna" <= c_max)

        if date_match and city_match:
            scanned += 1

    return scanned, total_rgs


def run_layout_experiment(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Experiment 1: Compare 5 physical layouts at 10M rows with ROI & pruning stats."""
    print("\n=== Experiment 1: Layout Comparison (N=10M, ROW_GROUP_SIZE=50,000) ===")

    date_norm = "(date_offset * 1023) // 729"
    city_norm = "(city_id * 1023) // 9"
    z_expr = f"({spread_bits_sql(date_norm)} | ({spread_bits_sql(city_norm)} << 1))"

    layouts = {
        "random": "SELECT sale_date, city, product, amount FROM sales",
        "date_sorted": "SELECT sale_date, city, product, amount FROM sales ORDER BY sale_date",
        "city_sorted": "SELECT sale_date, city, product, amount FROM sales ORDER BY city",
        "lex_date_city": "SELECT sale_date, city, product, amount FROM sales ORDER BY sale_date, city",
        "zorder_date_city": f"SELECT sale_date, city, product, amount FROM sales ORDER BY {z_expr}",
    }

    queries = {
        "date_filter": """
            SELECT COUNT(*), SUM(amount)
            FROM read_parquet('{file}')
            WHERE sale_date >= DATE '2024-06-01'
              AND sale_date < DATE '2024-07-01'
        """,
        "city_filter": """
            SELECT COUNT(*), SUM(amount)
            FROM read_parquet('{file}')
            WHERE city = 'Patna'
        """,
        "joint_filter": """
            SELECT COUNT(*), SUM(amount)
            FROM read_parquet('{file}')
            WHERE sale_date >= DATE '2024-06-01'
              AND sale_date < DATE '2024-07-01'
              AND city = 'Patna'
        """,
        "full_aggregation": """
            SELECT COUNT(*), SUM(amount), AVG(amount)
            FROM read_parquet('{file}')
        """,
    }

    files = {}
    build_times = {}
    file_sizes = {}

    for name, sql in layouts.items():
        path = f"{TMP_DIR}/{name}.parquet"
        t0 = time.perf_counter()
        con.execute(f"""
            COPY ({sql})
            TO '{path}'
            (FORMAT PARQUET, ROW_GROUP_SIZE {DEFAULT_ROW_GROUP_SIZE})
        """)
        build_times[name] = time.perf_counter() - t0
        files[name] = path
        file_sizes[name] = os.path.getsize(path) / (1024 * 1024)
        print(f"  Built {name:16s}: {build_times[name]:.3f}s | {file_sizes[name]:.2f} MB")

    tasks = [(layout, q) for layout in files for q in queries]
    outputs = {}

    # Warm-up & equivalence checks
    for layout, q in tasks:
        sql = queries[q].format(file=files[layout])
        for _ in range(WARMUP_RUNS):
            res = con.execute(sql).fetchone()
        outputs[(layout, q)] = res

    for q in queries:
        ref = outputs[("random", q)]
        for layout in files:
            assert outputs[(layout, q)] == ref, f"Mismatch on {layout} / {q}"
    print("  All query outputs verified identical across layouts.")

    # Randomized benchmark rounds
    rng = random.Random(42)
    timings = {t: [] for t in tasks}
    for _ in range(BENCHMARK_ROUNDS):
        round_tasks = tasks.copy()
        rng.shuffle(round_tasks)
        for layout, q in round_tasks:
            sql = queries[q].format(file=files[layout])
            t0 = time.perf_counter()
            con.execute(sql).fetchone()
            timings[(layout, q)].append(time.perf_counter() - t0)

    # Collect stats + pruning + ROI
    rows = []
    random_medians = {
        q: float(np.median(timings[("random", q)])) for q in queries
    }
    random_build = build_times["random"]

    for layout, q in tasks:
        vals = np.array(timings[(layout, q)])
        med = float(np.median(vals))
        scanned, total_rgs = count_pruned_row_groups(con, files[layout], q)

        build_overhead = build_times[layout] - random_build
        per_query_saving = random_medians[q] - med
        if layout == "random":
            breakeven = 0.0
        elif per_query_saving > 1e-6 and build_overhead > 0:
            breakeven = build_overhead / per_query_saving
        else:
            breakeven = float("inf")

        rows.append({
            "layout": layout,
            "query": q,
            "median_time_sec": med,
            "mean_time_sec": float(np.mean(vals)),
            "std_dev_sec": float(np.std(vals, ddof=1)),
            "p95_time_sec": float(np.percentile(vals, 95)),
            "build_time_sec": build_times[layout],
            "file_size_MB": file_sizes[layout],
            "row_groups_scanned": scanned,
            "row_groups_total": total_rgs,
            "pruning_pct": 100.0 * (1.0 - scanned / total_rgs),
            "breakeven_queries": breakeven,
        })

    # Clean up temporary parquet files to save disk space
    for path in files.values():
        os.remove(path)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/experiment1_layouts_10m.csv", index=False)
    print("  Saved results/experiment1_layouts_10m.csv")
    return df


def run_row_group_sweep(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Experiment 2: Sweep ROW_GROUP_SIZE on date_sorted (N=10M)."""
    print("\n=== Experiment 2: Row-Group Size Sweep (N=10M, date_sorted) ===")
    rg_sizes = [10_000, 50_000, 122_880, 500_000, 1_000_000]

    queries = {
        "date_filter": """
            SELECT COUNT(*), SUM(amount)
            FROM read_parquet('{file}')
            WHERE sale_date >= DATE '2024-06-01'
              AND sale_date < DATE '2024-07-01'
        """,
        "full_aggregation": """
            SELECT COUNT(*), SUM(amount), AVG(amount)
            FROM read_parquet('{file}')
        """,
    }

    rows = []
    for rg_size in rg_sizes:
        path = f"{TMP_DIR}/date_sorted_rg_{rg_size}.parquet"
        t0 = time.perf_counter()
        con.execute(f"""
            COPY (SELECT sale_date, city, product, amount FROM sales ORDER BY sale_date)
            TO '{path}'
            (FORMAT PARQUET, ROW_GROUP_SIZE {rg_size})
        """)
        build_time = time.perf_counter() - t0
        size_mb = os.path.getsize(path) / (1024 * 1024)

        for q_name, q_sql in queries.items():
            sql = q_sql.format(file=path)
            for _ in range(WARMUP_RUNS):
                con.execute(sql).fetchone()

            vals = []
            for _ in range(BENCHMARK_ROUNDS):
                t_q = time.perf_counter()
                con.execute(sql).fetchone()
                vals.append(time.perf_counter() - t_q)

            scanned, total_rgs = count_pruned_row_groups(con, path, q_name)
            rows.append({
                "row_group_size": rg_size,
                "query": q_name,
                "median_time_sec": float(np.median(vals)),
                "std_dev_sec": float(np.std(vals, ddof=1)),
                "build_time_sec": build_time,
                "file_size_MB": size_mb,
                "row_groups_scanned": scanned,
                "row_groups_total": total_rgs,
                "pruning_pct": 100.0 * (1.0 - scanned / total_rgs),
            })

        print(
            f"  RG={rg_size:7d} | size={size_mb:5.2f} MB | build={build_time:.2f}s"
        )
        os.remove(path)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/experiment2_row_group_sweep_10m.csv", index=False)
    print("  Saved results/experiment2_row_group_sweep_10m.csv")
    return df


def main():
    con = duckdb.connect()
    print(f"Generating {N:,} synthetic sales rows in DuckDB...")
    gen_time = create_synthetic_table(con, N)
    print(f"Table created in {gen_time:.2f}s.")

    df1 = run_layout_experiment(con)
    df2 = run_row_group_sweep(con)

    con.close()
    if os.path.exists(TMP_DIR) and not os.listdir(TMP_DIR):
        os.rmdir(TMP_DIR)

    print("\n--- Summary: Experiment 1 ---")
    print(df1[["layout", "query", "median_time_sec", "file_size_MB", "pruning_pct", "breakeven_queries"]].to_string(index=False))
    print("\n--- Summary: Experiment 2 ---")
    print(df2[["row_group_size", "query", "median_time_sec", "file_size_MB", "row_groups_scanned", "row_groups_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
