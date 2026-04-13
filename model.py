from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def run(
    inputs: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    configs: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:

    # ------------------------------------------------------------------
    # 1.  Load inputs & parameters
    # ------------------------------------------------------------------
    df = inputs["customer_data"].copy()

    w_gp  = float(parameters.get("w_gp",  0.40))
    w_imp = float(parameters.get("w_imp", 0.30))
    w_str = float(parameters.get("w_str", 0.20))
    w_alt = float(parameters.get("w_alt", 0.10))

    # Re-normalise weights so they always sum to 1 even if user edits them
    w_total = w_gp + w_imp + w_str + w_alt
    w_gp, w_imp, w_str, w_alt = (
        w_gp / w_total,
        w_imp / w_total,
        w_str / w_total,
        w_alt / w_total,
    )

    # ------------------------------------------------------------------
    # 2.  Parse data
    # ------------------------------------------------------------------
    customers    = df["Customer"].tolist()
    demand       = df["Quantity"].astype(float).values        # MT requested per customer
    gp_per_unit  = df["$/MT"].astype(float).values            # gross profit per MT
    alt          = df["Alt?"].astype(float).values            # 1=needs us, 4=can go elsewhere
    strategic    = df["Strategic?"].astype(float).values      # 4=most strategic
    importance   = df["Importance"].astype(float).values      # 4=most important

    # Supply is stored in the first row's Supply column
    supply = float(df["Supply"].dropna().iloc[0])

    n            = len(customers)
    total_demand = demand.sum()

    # ------------------------------------------------------------------
    # 3.  Build per-customer objective coefficients
    #
    #     Each term is normalized so its max possible contribution = 1,
    #     then scaled by the user weight.  scipy.linprog *minimises*,
    #     so we negate the whole objective.
    #
    #     GP term    : higher $/MT and more units allocated → higher score
    #                  normaliser: max_gp_per_unit × total_demand
    #
    #     Qualitative terms (imp, str, alt):
    #                  measured as fill-rate weighted by score
    #                  normaliser: n  (so perfect fill of all top-scored
    #                  customers gives a term value of 1)
    #                  divided by demand_i inside the coefficient because
    #                  x_i/demand_i is the fill-rate for customer i
    # ------------------------------------------------------------------
    max_gp = gp_per_unit.max()

    gp_coeff  = (gp_per_unit / max_gp) / total_demand          # shape (n,)
    imp_coeff = (importance / 4.0)      / (n * demand)          # shape (n,)
    str_coeff = (strategic  / 4.0)      / (n * demand)          # shape (n,)
    alt_coeff = ((5 - alt)  / 4.0)      / (n * demand)          # shape (n,)

    c = -(                                                       # negate to maximise
        w_gp  * gp_coeff
        + w_imp * imp_coeff
        + w_str * str_coeff
        + w_alt * alt_coeff
    )

    # ------------------------------------------------------------------
    # 4.  Constraints & bounds
    # ------------------------------------------------------------------
    # Inequality: Σ x_i <= supply
    A_ub = np.ones((1, n))
    b_ub = np.array([supply])

    # Bounds: 0 <= x_i <= demand_i
    bounds = [(0.0, d) for d in demand]

    # ------------------------------------------------------------------
    # 5.  Solve LP
    # ------------------------------------------------------------------
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.status != 0:
        raise RuntimeError(f"LP solver did not converge: {result.message}")

    allocated = result.x

    # ------------------------------------------------------------------
    # 6.  Build output tables
    # ------------------------------------------------------------------

    # --- Output 1: per-customer allocation results ---
    fill_rate        = allocated / demand
    gp_allocated     = allocated * gp_per_unit
    gp_max_possible  = demand    * gp_per_unit

    # Decompose each customer's contribution to the objective score
    # (un-negated, pre-weight) for transparency
    score_gp  = w_gp  * gp_coeff  * allocated
    score_imp = w_imp * imp_coeff * allocated
    score_str = w_str * str_coeff * allocated
    score_alt = w_alt * alt_coeff * allocated
    score_total = score_gp + score_imp + score_str + score_alt

    allocation_results = pd.DataFrame({
        "Customer":         customers,
        "Demand_MT":        demand.astype(int),
        "Allocated_MT":     np.round(allocated, 2),
        "Fill_Rate":        np.round(fill_rate, 4),
        "GP_Per_MT":        gp_per_unit,
        "GP_Allocated":     np.round(gp_allocated, 2),
        "GP_Max_Possible":  np.round(gp_max_possible, 2),
        "Importance":       importance.astype(int),
        "Strategic":        strategic.astype(int),
        "Alt":              alt.astype(int),
        "Score_GP":         np.round(score_gp,  6),
        "Score_Importance": np.round(score_imp, 6),
        "Score_Strategic":  np.round(score_str, 6),
        "Score_Alt":        np.round(score_alt, 6),
        "Score_Total":      np.round(score_total, 6),
    })

    # Sort by descending total score so highest-priority allocations appear first
    allocation_results = allocation_results.sort_values(
        "Score_Total", ascending=False
    ).reset_index(drop=True)

    # --- Output 2: model summary ---
    model_summary = pd.DataFrame([{
        "Total_Supply":          int(supply),
        "Total_Demand":          int(total_demand),
        "Total_Allocated":       round(allocated.sum(), 2),
        "Supply_Utilisation_Pct":round(allocated.sum() / supply * 100, 2),
        "Total_GP_Allocated":    round(gp_allocated.sum(), 2),
        "Total_GP_Max_Possible": round(gp_max_possible.sum(), 2),
        "GP_Capture_Pct":        round(gp_allocated.sum() / gp_max_possible.sum() * 100, 2),
        "Objective_Score":       round(-result.fun, 6),
        "w_gp":                  round(w_gp,  4),
        "w_imp":                 round(w_imp, 4),
        "w_str":                 round(w_str, 4),
        "w_alt":                 round(w_alt, 4),
        "Solver_Status":         result.message,
    }])

    return {
        "allocation_results": allocation_results,
        "model_summary":      model_summary,
    }


# ----------------------------------------------------------------------
# Local testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        "Customer":   list("ABCDEFGHIJK"),
        "Quantity":   [500, 600, 400, 450, 430, 250, 280, 1000, 250, 750, 700],
        "$/MT":       [100,  90, 130, 120,  90,  85, 105,  100,  95, 100, 115],
        "GP":         [0] * 11,        # derived column, not used directly
        "Alt?":       [3, 2, 1, 4, 1, 1, 3, 2, 1, 2, 1],
        "Strategic?": [3, 1, 2, 1, 4, 4, 2, 1, 3, 3, 4],
        "Importance": [3, 1, 3, 1, 4, 4, 2, 2, 2, 3, 3],
        "Supply":     [4000] + [None] * 10,
    })

    results = run(
        inputs={"customer_data": sample},
        parameters={"w_gp": 0.40, "w_imp": 0.30, "w_str": 0.20, "w_alt": 0.10},
        configs={},
    )

    print("=== Allocation Results ===")
    print(results["allocation_results"].to_string(index=False))
    print("\n=== Model Summary ===")
    print(results["model_summary"].to_string(index=False))
