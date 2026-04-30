"""Weighted-sum LP allocation model (v3).

Changes vs v2:
    - Schema is now per (Company, Product) row. A company can appear on multiple
      products; a product can be supplied to multiple companies.
    - Removed columns: Alt?, Filled, Strategic? — no longer part of the model.
    - Contract_MT retained for reporting only; not used as a constraint.
    - Allocation cap is Forecast_MT (not Contract_MT). Forecast > Contract is
      permitted: the model allocates up to forecasted true demand.
    - Supply is per-product. Each product has its own capacity constraint;
      products do not share supply.
    - Objective reduces to GP vs Importance (two terms instead of four).

Coefficient structure (per-MT):
    c_gp_ik   = (GP_i / max_GP) / total_forecast
    c_imp_ik  = (Importance_i / 4) / total_forecast
    c_ik      = w_gp * c_gp_ik + w_imp * c_imp_ik

LP:
    max   sum_(i,k) c_ik * x_ik
    s.t.  sum_i x_ik <= S_k         for each product k
          0 <= x_ik <= forecast_ik
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-9


def _build_coefficients(gp_per_unit, importance, total_forecast):
    max_gp = max(float(gp_per_unit.max()), EPS)
    tf     = max(float(total_forecast),    EPS)
    gp_coeff  = (gp_per_unit / max_gp) / tf
    imp_coeff = (importance  / 4.0)    / tf
    return gp_coeff, imp_coeff


def run(
    inputs: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    configs: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    # ------------------------------------------------------------------
    # 1.  Load inputs & weights
    # ------------------------------------------------------------------
    df       = inputs["customer_data"].copy()
    supply_df = inputs["supply"].copy()  # columns: Product, Supply_MT

    w_gp  = float(parameters.get("w_gp",  0.50))
    w_imp = float(parameters.get("w_imp", 0.50))

    w_total = w_gp + w_imp
    if w_total <= 0:
        raise ValueError("At least one weight must be > 0")
    w_gp, w_imp = w_gp / w_total, w_imp / w_total

    # ------------------------------------------------------------------
    # 2.  Parse data
    # ------------------------------------------------------------------
    companies   = df["Company"].tolist()
    products    = df["Product"].tolist()
    contract    = df["Contract_MT"].astype(float).values    # informational only
    forecast    = df["Forecast_MT"].astype(float).values
    gp_per_unit = df["GP_per_MT"].astype(float).values
    importance  = df["Importance"].astype(float).values

    n               = len(df)
    total_forecast  = forecast.sum()

    # Per-product supply lookup
    supply_map = dict(zip(supply_df["Product"], supply_df["Supply_MT"].astype(float)))
    unique_products = list(dict.fromkeys(products))  # preserve order, unique
    missing = [p for p in unique_products if p not in supply_map]
    if missing:
        raise ValueError(f"No supply provided for products: {missing}")

    # ------------------------------------------------------------------
    # 3.  Build objective
    # ------------------------------------------------------------------
    gp_coeff, imp_coeff = _build_coefficients(gp_per_unit, importance, total_forecast)
    c_per_unit = w_gp * gp_coeff + w_imp * imp_coeff
    c = -c_per_unit  # linprog minimises

    priority_rank = (-c_per_unit).argsort().argsort() + 1  # 1 = highest priority

    # ------------------------------------------------------------------
    # 4.  Build per-product capacity constraints
    # ------------------------------------------------------------------
    # A_ub: one row per product, one column per (company, product) decision var.
    # Entry is 1 if that decision var contributes to that product's supply.
    A_ub = np.zeros((len(unique_products), n))
    b_ub = np.zeros(len(unique_products))
    for row_idx, prod in enumerate(unique_products):
        col_mask = np.array([p == prod for p in products], dtype=float)
        A_ub[row_idx] = col_mask
        b_ub[row_idx] = supply_map[prod]

    # ------------------------------------------------------------------
    # 5.  Solve
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
    # 6.  Outputs
    # ------------------------------------------------------------------
    forecast_fill_rate = np.where(forecast > 0, allocated / forecast, 0.0)
    contract_fill_rate = np.where(contract > 0, allocated / contract, 0.0)
    gp_allocated       = allocated * gp_per_unit
    gp_max_possible    = forecast  * gp_per_unit  # ceiling driven by forecast, not contract

    score_gp  = w_gp  * gp_coeff  * allocated
    score_imp = w_imp * imp_coeff * allocated

    allocation_results = pd.DataFrame({
        "Company":              companies,
        "Product":              products,
        "Priority_Rank":        priority_rank,
        "Contract_MT":          contract.astype(int),
        "Forecast_MT":          forecast.astype(int),
        "Allocated_MT":         np.round(allocated, 2),
        "Forecast_Fill_Rate":   np.round(forecast_fill_rate, 4),
        "Contract_Fill_Rate":   np.round(contract_fill_rate, 4),
        "GP_per_MT":            gp_per_unit,
        "GP_Allocated":         np.round(gp_allocated, 2),
        "GP_Max_at_Forecast":   np.round(gp_max_possible, 2),
        "Importance":           importance.astype(int),
        "Score_GP":             np.round(score_gp,  6),
        "Score_Importance":     np.round(score_imp, 6),
        "Score_Total":          np.round(score_gp + score_imp, 6),
    }).sort_values(["Product", "Priority_Rank"]).reset_index(drop=True)

    # Per-product summary
    per_product = allocation_results.groupby("Product").agg(
        Total_Forecast_MT=("Forecast_MT",  "sum"),
        Total_Allocated_MT=("Allocated_MT","sum"),
        Total_GP_Allocated=("GP_Allocated","sum"),
    ).reset_index()
    per_product["Supply_MT"] = per_product["Product"].map(supply_map).astype(int)
    per_product["Supply_Util_Pct"] = np.round(
        per_product["Total_Allocated_MT"] / per_product["Supply_MT"].clip(lower=EPS) * 100, 2
    )
    per_product = per_product[[
        "Product", "Supply_MT", "Total_Forecast_MT",
        "Total_Allocated_MT", "Supply_Util_Pct", "Total_GP_Allocated",
    ]]

    model_summary = pd.DataFrame([{
        "N_Rows":               n,
        "N_Companies":          df["Company"].nunique(),
        "N_Products":           len(unique_products),
        "Total_Supply":         int(sum(supply_map[p] for p in unique_products)),
        "Total_Contract":       int(contract.sum()),
        "Total_Forecast":       int(forecast.sum()),
        "Total_Allocated":      round(allocated.sum(), 2),
        "Total_GP_Allocated":   round(gp_allocated.sum(), 2),
        "Total_GP_at_Forecast": round(gp_max_possible.sum(), 2),
        "GP_Capture_Pct":       round(gp_allocated.sum() / max(gp_max_possible.sum(), EPS) * 100, 2),
        "Objective_Score":      round(-result.fun, 6),
        "w_gp":                 round(w_gp,  4),
        "w_imp":                round(w_imp, 4),
        "Solver_Status":        result.message,
    }])

    return {
        "allocation_results":   allocation_results,
        "per_product_summary":  per_product,
        "model_summary":        model_summary,
    }


# ----------------------------------------------------------------------
# Local testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = {
        "customer_data": pd.read_csv("customer_data_v3.csv"),
        "supply":        pd.read_csv("supply_v3.csv"),
    }
    params = {"w_gp": 0.50, "w_imp": 0.50}

    outputs = run(inputs=sample_data, parameters=params, configs={})

    print("=== Allocation Results ===")
    print(outputs["allocation_results"].to_string(index=False))
    print("\n=== Per-Product Summary ===")
    print(outputs["per_product_summary"].to_string(index=False))
    print("\n=== Model Summary ===")
    print(outputs["model_summary"].to_string(index=False))

    outputs["allocation_results"].to_csv("allocation_results_v3.csv", index=False)
    outputs["per_product_summary"].to_csv("per_product_summary_v3.csv", index=False)
    outputs["model_summary"].to_csv("model_summary_v3.csv", index=False)
