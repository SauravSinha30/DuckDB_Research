
import duckdb
import pandas as pd
import numpy as np
import time
import os
import random

# Configuration
N = 1_000_000
ROW_GROUP_SIZE = 50_000
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

rng = np.random.default_rng(42)

print("Generating dataset...")

# Generate synthetic sales data
df = pd.DataFrame({
    "sale_date": pd.Timestamp("2024-01-01") +
                 pd.to_timedelta(
                     rng.integers(0, 730, N), unit="D"
                 ),
    "city": rng.choice(
        ["Patna", "Delhi", "Mumbai", "Kolkata",
         "Bangalore", "Chennai", "Pune", "Hyderabad",
         "Jaipur", "Lucknow"], N
    ),
    "product": rng.choice(
        ["Laptop", "Phone", "Tablet", "Monitor"], N
    ),
    "amount": rng.integers(500, 100000, N)
})

con = duckdb.connect()
con.register("sales_df", df)

# Create three Parquet layouts
layouts = {
    "random": "SELECT * FROM sales_df",
    "date_sorted": "SELECT * FROM sales_df ORDER BY sale_date",
    "city_sorted": "SELECT * FROM sales_df ORDER BY city"
}

files = {}

for name, query in layouts.items():
    path = f"data/{name}.parquet"

    start = time.perf_counter()

    con.execute(f"""
        COPY ({query})
        TO '{path}'
        (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    build_time = time.perf_counter() - start
    files[name] = path

    print(
        f"{name}: created in {build_time:.3f} seconds"
    )



# Define benchmark queries
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

    "full_aggregation": """
        SELECT COUNT(*), SUM(amount), AVG(amount)
        FROM read_parquet('{file}')
    """
}

# Prepare all nine layout-query combinations
tasks = [
    (layout, query_name)
    for layout in files
    for query_name in queries
]

timings = {
    (layout, query_name): []
    for layout, query_name in tasks
}

query_outputs = {}

print("\nPerforming warm-up...")

# Warm up every combination
for layout, query_name in tasks:
    sql = queries[query_name].format(file=files[layout])

    for _ in range(3):
        output = con.execute(sql).fetchone()

    query_outputs[(layout, query_name)] = output

# Check that equivalent queries return identical results
for query_name in queries:
    outputs = [
        query_outputs[(layout, query_name)]
        for layout in files
    ]

    assert outputs[0] == outputs[1] == outputs[2], (
        f"Result mismatch in {query_name}"
    )

print("All query results match.")

print("\nRunning randomized benchmark...")

# 21 rounds, with randomized execution order
rng = random.Random(42)

for round_number in range(21):
    round_tasks = tasks.copy()
    rng.shuffle(round_tasks)

    for layout, query_name in round_tasks:
        sql = queries[query_name].format(file=files[layout])

        start = time.perf_counter()
        con.execute(sql).fetchone()
        elapsed = time.perf_counter() - start

        timings[(layout, query_name)].append(elapsed)

# Generate statistical results
results = []

for layout, query_name in tasks:
    values = np.array(timings[(layout, query_name)])

    results.append({
        "layout": layout,
        "query": query_name,
        "median_time_sec": np.median(values),
        "mean_time_sec": np.mean(values),
        "std_dev_sec": np.std(values, ddof=1),
        "min_time_sec": np.min(values),
        "max_time_sec": np.max(values),
        "file_size_MB": (
            os.path.getsize(files[layout]) / (1024 * 1024)
        )
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    "results/benchmark_results_v3.csv",
    index=False
)

print("\nValidated Benchmark Results:")
print(result_df.to_string(index=False))

con.close()

print("\nValidation experiment completed!")


# # Save benchmark results
# result_df = pd.DataFrame(results)
# result_df.to_csv(
#     "results/benchmark_results_v2.csv",
#     index=False
# )

# print("\nFinal Results:")
# print(result_df.to_string(index=False))

# con.close()
# print("\nExperiment completed!")
