import os
import random
import time
import duckdb
import numpy as np
import pandas as pd

N = 10_000_000
ROW_GROUP_SIZE = 50_000
WARMUP_RUNS = 2
BENCHMARK_ROUNDS = 11
TMP_DIR = "data/tmp_exp3"
RESULTS_DIR = "results"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def spread_bits_u32(v: np.ndarray) -> np.ndarray:
    """Spread lower 10 bits of uint32 array into even bit positions."""
    x = v.astype(np.uint32, copy=True)
    x = (x | (x << 8)) & np.uint32(0x00FF00FF)
    x = (x | (x << 4)) & np.uint32(0x0F0F0F0F)
    x = (x | (x << 2)) & np.uint32(0x33333333)
    x = (x | (x << 1)) & np.uint32(0x55555555)
    return x


def morton_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return spread_bits_u32(x) | (spread_bits_u32(y) << np.uint32(1))


def hilbert_2d(n_bits: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized 2D Hilbert curve index on a (2^n_bits x 2^n_bits) grid."""
    x = x.astype(np.int64, copy=True)
    y = y.astype(np.int64, copy=True)
    d = np.zeros_like(x, dtype=np.int64)
    s = 1 << (n_bits - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += (s * s) * ((3 * rx) ^ ry)
        mask = ry == 0
        swap_mask = mask & (rx == 1)
        x[swap_mask] ^= s - 1
        y[swap_mask] ^= s - 1
        x_tmp = x[mask].copy()
        x[mask] = y[mask]
        y[mask] = x_tmp
        s >>= 1
    return d


def count_pruned(con: duckdb.DuckDBPyConnection, path: str, q_name: str, cold_id: int) -> tuple[int, int]:
    meta = con.execute(f"""
        SELECT row_group_id, path_in_schema,
               CAST(stats_min_value AS INTEGER) AS vmin,
               CAST(stats_max_value AS INTEGER) AS vmax
        FROM parquet_metadata('{path}')
        WHERE path_in_schema IN ('date_offset', 'entity_id')
    """).fetchdf()

    total_rgs = int(meta["row_group_id"].nunique())
    d_meta = meta[meta["path_in_schema"] == "date_offset"].set_index("row_group_id")
    e_meta = meta[meta["path_in_schema"] == "entity_id"].set_index("row_group_id")

    scanned = 0
    for rg in range(total_rgs):
        d_ok = True
        e_ok = True
        if q_name in ("date_filter", "joint_hot", "joint_cold"):
            d_min, d_max = int(d_meta.loc[rg, "vmin"]), int(d_meta.loc[rg, "vmax"])
            d_ok = not (d_max < 152 or d_min > 181)
        if q_name in ("hot_entity", "joint_hot"):
            e_min, e_max = int(e_meta.loc[rg, "vmin"]), int(e_meta.loc[rg, "vmax"])
            e_ok = e_min <= 0 <= e_max
        if q_name in ("cold_entity", "joint_cold"):
            e_min, e_max = int(e_meta.loc[rg, "vmin"]), int(e_meta.loc[rg, "vmax"])
            e_ok = e_min <= cold_id <= e_max
        if d_ok and e_ok:
            scanned += 1
    return scanned, total_rgs


def main():
    rng_np = np.random.default_rng(42)
    con = duckdb.connect()

    # Pre-generate base uniform columns once
    date_offset = rng_np.integers(0, 730, size=N, dtype=np.int32)
    u_entity = rng_np.random(size=N)
    amount = rng_np.integers(500, 100000, size=N, dtype=np.int32)

    scenarios = [
        # (cardinality, skew_name, power_exponent)
        (10, "uniform", 1.0),
        (10, "heavy_skew", 5.0),
        (1024, "uniform", 1.0),
        (1024, "heavy_skew", 5.0),
    ]

    all_rows = []

    for card, skew_name, p_exp in scenarios:
        entity_id = np.floor(card * (u_entity ** p_exp)).astype(np.int32)
        entity_id = np.clip(entity_id, 0, card - 1)
        hot_share = 100.0 * np.mean(entity_id == 0)
        cold_id = card - 1
        cold_share = 100.0 * np.mean(entity_id == cold_id)

        print(f"\n=== Scenario: C={card}, {skew_name} (hot_id=0: {hot_share:.2f}%, cold_id={cold_id}: {cold_share:.3f}%) ===")

        # Normalize to 10-bit [0, 1023]
        d_norm = ((date_offset.astype(np.int64) * 1023) // 729).astype(np.uint32)
        e_norm = ((entity_id.astype(np.int64) * 1023) // (card - 1)).astype(np.uint32)

        t_z0 = time.perf_counter()
        z_key = morton_2d(d_norm, e_norm)
        t_z = time.perf_counter() - t_z0

        t_h0 = time.perf_counter()
        h_key = hilbert_2d(10, d_norm, e_norm)
        t_h = time.perf_counter() - t_h0

        df_in = pd.DataFrame({
            "date_offset": date_offset,
            "entity_id": entity_id,
            "amount": amount,
            "z_key": z_key,
            "h_key": h_key,
        })
        con.register("tbl_in", df_in)

        layouts = {
            "date_sorted": "SELECT date_offset, entity_id, amount FROM tbl_in ORDER BY date_offset",
            "lex_date_entity": "SELECT date_offset, entity_id, amount FROM tbl_in ORDER BY date_offset, entity_id",
            "zorder": "SELECT date_offset, entity_id, amount FROM tbl_in ORDER BY z_key",
            "hilbert": "SELECT date_offset, entity_id, amount FROM tbl_in ORDER BY h_key",
        }

        queries = {
            "date_filter": "SELECT COUNT(*), SUM(amount) FROM read_parquet('{file}') WHERE date_offset BETWEEN 152 AND 181",
            "hot_entity": "SELECT COUNT(*), SUM(amount) FROM read_parquet('{file}') WHERE entity_id = 0",
            "cold_entity": f"SELECT COUNT(*), SUM(amount) FROM read_parquet('{{file}}') WHERE entity_id = {cold_id}",
            "joint_hot": "SELECT COUNT(*), SUM(amount) FROM read_parquet('{file}') WHERE date_offset BETWEEN 152 AND 181 AND entity_id = 0",
            "joint_cold": f"SELECT COUNT(*), SUM(amount) FROM read_parquet('{{file}}') WHERE date_offset BETWEEN 152 AND 181 AND entity_id = {cold_id}",
        }

        files = {}
        build_times = {}
        for l_name, l_sql in layouts.items():
            p = f"{TMP_DIR}/{l_name}.parquet"
            t0 = time.perf_counter()
            con.execute(f"COPY ({l_sql}) TO '{p}' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})")
            extra = t_z if l_name == "zorder" else (t_h if l_name == "hilbert" else 0.0)
            build_times[l_name] = (time.perf_counter() - t0) + extra
            files[l_name] = p

        tasks = [(l, q) for l in files for q in queries]
        for l, q in tasks:
            sql = queries[q].format(file=files[l])
            for _ in range(WARMUP_RUNS):
                con.execute(sql).fetchone()

        rng_py = random.Random(42)
        timings = {t: [] for t in tasks}
        for _ in range(BENCHMARK_ROUNDS):
            rt = tasks.copy()
            rng_py.shuffle(rt)
            for l, q in rt:
                sql = queries[q].format(file=files[l])
                t0 = time.perf_counter()
                con.execute(sql).fetchone()
                timings[(l, q)].append(time.perf_counter() - t0)

        for l, q in tasks:
            scanned, total_rgs = count_pruned(con, files[l], q, cold_id)
            med_ms = 1000.0 * float(np.median(timings[(l, q)]))
            all_rows.append({
                "cardinality": card,
                "skew": skew_name,
                "hot_share_pct": hot_share,
                "layout": l,
                "query": q,
                "median_ms": med_ms,
                "build_time_sec": build_times[l],
                "file_size_MB": os.path.getsize(files[l]) / (1024 * 1024),
                "rg_scanned": scanned,
                "rg_total": total_rgs,
                "pruning_pct": 100.0 * (1.0 - scanned / total_rgs),
            })

        for p in files.values():
            os.remove(p)
        con.unregister("tbl_in")
        del df_in, z_key, h_key, entity_id

    con.close()
    if os.path.exists(TMP_DIR) and not os.listdir(TMP_DIR):
        os.rmdir(TMP_DIR)

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(f"{RESULTS_DIR}/experiment3_skew_hilbert_10m.csv", index=False)
    print("\nSaved results/experiment3_skew_hilbert_10m.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
