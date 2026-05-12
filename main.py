"""Forecast-driven allocation model (v4).

Input tables (passed as DataFrames in `inputs` dict):
    DemandStatement  : (Customer ID, Product, Forecast, Gross Profit MT)
    Customer         : (Customer ID, Customer Name, Segment)
    Product          : (Product ID, Product Name, Supply)
    Segmentation     : OPTIONAL one-row table with weights for each segment

Parameters:
    w_gp  : weight on gross profit term  (default 0.5)
    w_seg : weight on segment term       (default 0.5)

Joining rules:
    - DemandStatement.Customer ID must exist in Customer table → else DROP row + warn
    - DemandStatement.Product   must exist in Product table   → else supply treated as 0
      (so the LP allocates 0 across all rows for that product)
    - Segmentation weights are rescaled so PARTNER = 1.0 (others scale relative)
    - Default segment weights if table not provided:
        PARTNER=100, DISTRIBUTOR=75, STRATEGIC=60, STANDARD=45,
        LIGHT TOUCH=25, UNASSIGNED=10

Objective (maximize):

    sum over rows (i,k):
        w_gp  * (Allocated_ik * GP_per_MT_i) / GP_max_possible
      + w_seg * (Allocated_ik * SegWeight_i) / Seg_max_possible

    where:
        GP_max_possible  = sum_ik (Forecast_ik * GP_per_MT_i)
        Seg_max_possible = sum_ik (Forecast_ik * SegWeight_i)

    Both terms read as "fraction of theoretical max captured." With w_gp=1
    the objective IS the GP capture rate (a number in [0,1]). Same logic
    for w_seg=1 and the segment-weighted volume capture rate. Mixed weights
    blend the two capture rates linearly.

Constraints:
    sum_i Allocated_ik <= Supply_k         for each product k
    0 <= Allocated_ik <= Forecast_ik       per row
"""

from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-9

DEFAULT_SEG_WEIGHTS = {
    "PARTNER":     100,
    "DISTRIBUTOR":  75,
    "STRATEGIC":    60,
    "STANDARD":     45,
    "LIGHT TOUCH":  25,
    "UNASSIGNED":   10,
}


def _resolve_segment_weights(seg_df: pd.DataFrame | None) -> Dict[str, float]:
    """Return segment → weight dict, rescaled so PARTNER = 1.0."""
    if seg_df is None or seg_df.empty:
        raw = DEFAULT_SEG_WEIGHTS.copy()
    else:
        raw = {col: float(seg_df.iloc[0][col]) for col in seg_df.columns}

    partner = raw.get("PARTNER", 0.0)
    if partner <= 0:
        raise ValueError("PARTNER segment weight must be > 0 (used as normalization anchor)")
    return {k: v / partner for k, v in raw.items()}


def run(
    inputs: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    configs: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    # ------------------------------------------------------------------
    # 1.  Load inputs
    # ------------------------------------------------------------------
    demand    = inputs["DemandStatement"].copy()
    customers = inputs["Customer"].copy()
    products  = inputs["Product"].copy()
    seg_df    = inputs.get("Segmentation")

    w_gp  = float(parameters.get("w_gp",  0.5))
    w_seg = float(parameters.get("w_seg", 0.5))
    w_total = w_gp + w_seg
    if w_total <= 0:
        raise ValueError("At least one weight must be > 0")
    w_gp, w_seg = w_gp / w_total, w_seg / w_total

    seg_weights = _resolve_segment_weights(seg_df)

    # ------------------------------------------------------------------
    # 2.  Resolve joins; drop unmatched customers, zero-out unmatched products
    # ------------------------------------------------------------------
    n_before = len(demand)
    demand = demand.merge(
        customers[["Customer ID", "Segment"]],
        on="Customer ID",
        how="left",
    )
    unmatched_cust = demand["Segment"].isna()
    if unmatched_cust.any():
        bad = demand.loc[unmatched_cust, "Customer ID"].unique().tolist()
        warnings.warn(f"Dropping {unmatched_cust.sum()} demand row(s); "
                      f"unknown Customer IDs: {bad}")
        demand = demand[~unmatched_cust].copy()

    # Validate segment values
    unknown_seg = ~demand["Segment"].isin(seg_weights.keys())
    if unknown_seg.any():
        bad = demand.loc[unknown_seg, "Segment"].unique().tolist()
        warnings.warn(f"Dropping {unknown_seg.sum()} demand row(s); "
                      f"segments not in weight table: {bad}")
        demand = demand[~unknown_seg].copy()

    # Product join: missing product → supply = 0 (NOT a drop)
    demand = demand.merge(
        products[["Product ID", "Supply"]].rename(columns={"Product ID": "Product"}),
        on="Product",
        how="left",
    )
    demand["Supply"] = demand["Supply"].fillna(0.0)

    if len(demand) == 0:
        raise RuntimeError("All demand rows dropped after validation; nothing to allocate")

    demand = demand.reset_index(drop=True)
    n = len(demand)
    print(f"[info] {n_before} demand rows in → {n} rows after validation")

    # ------------------------------------------------------------------
    # 3.  Build vectors
    # ------------------------------------------------------------------
    cust_ids    = demand["Customer ID"].tolist()
    prod_ids    = demand["Product"].tolist()
    forecast    = demand["Forecast"].astype(float).values
    gp_per_mt   = demand["Gross Profit MT"].astype(float).values
    seg_w_vec   = demand["Segment"].map(seg_weights).astype(float).values

    # Theoretical maxima (if every customer got their full forecast)
    GP_TOT  = float((forecast * gp_per_mt).sum())
    SEG_TOT = float((forecast * seg_w_vec).sum())

    if GP_TOT <= EPS:
        raise ValueError("Total GP-possible is zero; check Gross Profit MT values")
    if SEG_TOT <= EPS:
        raise ValueError("Total segment-weighted demand is zero; check segment weights")

    # ------------------------------------------------------------------
    # 4.  Per-MT objective coefficients
    #
    # contribution_ik = x_ik * (gp_per_mt_i / GP_TOT)   * w_gp
    #                 + x_ik * (seg_w_i     / SEG_TOT)  * w_seg
    #
    # Both terms divide by their own theoretical maximum, so each is the
    # row's contribution to a capture-rate in [0,1]. Weights blend the
    # two rates linearly and are interpretable on the same scale.
    # ------------------------------------------------------------------
    c_gp_per_mt  = gp_per_mt / GP_TOT
    c_seg_per_mt = seg_w_vec / SEG_TOT
    c_per_mt     = w_gp * c_gp_per_mt + w_seg * c_seg_per_mt
    c            = -c_per_mt  # linprog minimises

    priority_rank = (-c_per_mt).argsort().argsort() + 1

    # ------------------------------------------------------------------
    # 5.  Per-product capacity constraints
    # ------------------------------------------------------------------
    unique_products_in_demand = list(dict.fromkeys(prod_ids))
    supply_map = dict(zip(products["Product ID"], products["Supply"].astype(float)))
    # Missing products default to 0 (per design decision)
    for p in unique_products_in_demand:
        supply_map.setdefault(p, 0.0)

    A_ub = np.zeros((len(unique_products_in_demand), n))
    b_ub = np.zeros(len(unique_products_in_demand))
    for row_idx, prod in enumerate(unique_products_in_demand):
        A_ub[row_idx] = np.array([p == prod for p in prod_ids], dtype=float)
        b_ub[row_idx] = supply_map[prod]

    # ------------------------------------------------------------------
    # 6.  Solve
    # ------------------------------------------------------------------
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, f) for f in forecast],
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"LP solver did not converge: {result.message}")
    allocated = result.x

    # ------------------------------------------------------------------
    # 7.  Outputs
    # ------------------------------------------------------------------
    forecast_fill = np.where(forecast > 0, allocated / forecast, 0.0)
    gp_allocated  = allocated * gp_per_mt
    gp_max_row    = forecast  * gp_per_mt
    seg_score_row = allocated * seg_w_vec

    # Customer Name lookup
    name_map = dict(zip(customers["Customer ID"], customers["Customer Name"]))
    pname_map = dict(zip(products["Product ID"], products["Product Name"]))

    allocation_results = pd.DataFrame({
        "Customer ID":         cust_ids,
        "Customer Name":       [name_map.get(c, "") for c in cust_ids],
        "Segment":             demand["Segment"].values,
        "Segment_Weight":      np.round(seg_w_vec, 4),
        "Product":             prod_ids,
        "Product Name":        [pname_map.get(p, "") for p in prod_ids],
        "Priority_Rank":       priority_rank,
        "Forecast":            np.round(forecast, 2),
        "Allocated":           np.round(allocated, 2),
        "Forecast_Fill_Rate":  np.round(forecast_fill, 4),
        "GP_per_MT":           np.round(gp_per_mt, 2),
        "GP_Allocated":        np.round(gp_allocated, 2),
        "GP_Max_at_Forecast":  np.round(gp_max_row, 2),
        "Score_GP":            np.round(w_gp  * c_gp_per_mt  * allocated, 6),
        "Score_Segment":       np.round(w_seg * c_seg_per_mt * allocated, 6),
        "Score_Total":         np.round(c_per_mt * allocated, 6),
    }).sort_values(["Product", "Priority_Rank"]).reset_index(drop=True)

    per_product = allocation_results.groupby(["Product", "Product Name"]).agg(
        Total_Forecast=("Forecast",     "sum"),
        Total_Allocated=("Allocated",   "sum"),
        Total_GP_Allocated=("GP_Allocated","sum"),
    ).reset_index()
    per_product["Supply"] = per_product["Product"].map(supply_map)
    per_product["Supply_Util_Pct"] = np.round(
        per_product["Total_Allocated"] / per_product["Supply"].clip(lower=EPS) * 100, 2
    )
    per_product = per_product[[
        "Product", "Product Name", "Supply", "Total_Forecast",
        "Total_Allocated", "Supply_Util_Pct", "Total_GP_Allocated",
    ]]

    gp_capture_rate  = gp_allocated.sum() / max(GP_TOT, EPS)
    seg_capture_rate = seg_score_row.sum() / max(SEG_TOT, EPS)

    model_summary = pd.DataFrame([{
        "N_Demand_Rows":         n,
        "N_Customers":           len(set(cust_ids)),
        "N_Products":            len(unique_products_in_demand),
        "Total_Supply":          round(sum(supply_map[p] for p in unique_products_in_demand), 2),
        "Total_Forecast":        round(forecast.sum(), 2),
        "Total_Allocated":       round(allocated.sum(), 2),
        "Total_GP_Allocated":    round(gp_allocated.sum(), 2),
        "Total_GP_at_Forecast":  round(GP_TOT, 2),
        "GP_Capture_Rate":       round(gp_capture_rate, 4),
        "Segment_Capture_Rate":  round(seg_capture_rate, 4),
        "Objective_Score":       round(-result.fun, 6),
        "w_gp":                  round(w_gp,  4),
        "w_seg":                 round(w_seg, 4),
        "Solver_Status":         result.message,
    }])

    return {
        "allocation_results":  allocation_results,
        "per_product_summary": per_product,
        "model_summary":       model_summary,
    }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    inputs = {
        "DemandStatement": pd.read_csv("demand_statement.csv"),
        "Customer":        pd.read_csv("customer.csv"),
        "Product":         pd.read_csv("product.csv"),
        "Segmentation":    pd.read_csv("segmentation.csv"),
    }
    params = {"w_gp": 0.5, "w_seg": 0.5}

    out = run(inputs=inputs, parameters=params, configs={})

    print("\n=== Allocation Results ===")
    print(out["allocation_results"].to_string(index=False))
    print("\n=== Per-Product Summary ===")
    print(out["per_product_summary"].to_string(index=False))
    print("\n=== Model Summary ===")
    print(out["model_summary"].to_string(index=False))

    out["allocation_results"].to_csv("allocation_results.csv", index=False)
    out["per_product_summary"].to_csv("per_product_summary.csv", index=False)
    out["model_summary"].to_csv("model_summary.csv", index=False)
