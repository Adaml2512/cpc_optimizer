"""Weighted-sum LP allocation model (v2).

Changes vs v1 (see inline FIX tags):
    FIX #1 : Qualitative coefficients (imp/str/alt) no longer divide by per-customer
             demand. The old (x_i / d_i) ratio biased the LP toward small-demand
             customers: a customer with Importance=4, demand=250 got ~4x the
             marginal signal of an Importance=4, demand=1000 customer. All four
             terms now share a single global normalizer (total_remaining) so the
             weights w_gp / w_imp / w_str / w_alt are genuinely comparable.

    FIX #2 : Added Priority_Rank column. Because the objective is linear and
             constraints are box + single sum, the LP is mathematically equivalent
             to ranking customers by per-unit coefficient c_i and filling greedily.
             Priority_Rank surfaces this ordering pre-solve so the mechanism is
             transparent. LP form is retained so the module extends cleanly to
             MILP (top-off bonus, shutdown floors) without restructuring.

    FIX #5 : Div-by-zero guards on max_gp (zero-price edge case), total_remaining
             (fully-filled book), demand_i (zero-demand row), and GP_Max_Possible
             (all-zero price column). EPS floor used for denominators.

Scale conventions unchanged:
    Importance / Strategic : 4 = highest, 1 = lowest
    Alt                    : 1 = sole supplier (high retention), 4 = many alternatives
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-9  # FIX #5: denominator floor


def _build_coefficients(gp_per_unit, importance, strategic, alt, total_remaining):
    """Per-MT objective coefficients, all normalized by the same global scalar.

    Note: total_remaining is a constant scalar, so it has no effect on the LP's
    argmax. It exists only so reported Score_* columns are interpretable as
    fractional contributions to an aggregate objective.
    """
    # FIX #5: guards
    max_gp = max(float(gp_per_unit.max()), EPS)
    tr     = max(float(total_remaining),   EPS)

    # FIX #1: common divisor for all four terms (previously per-customer for imp/str/alt)
    gp_coeff  = (gp_per_unit / max_gp) / tr
    imp_coeff = (importance  / 4.0)    / tr
    str_coeff = (strategic   / 4.0)    / tr
    alt_coeff = ((5 - alt)   / 4.0)    / tr

    return gp_coeff, imp_coeff, str_coeff, alt_coeff


def run(
    inputs: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    configs: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    # ------------------------------------------------------------------
    # 1.  Load inputs & weights
    # ------------------------------------------------------------------
    df = inputs["customer_data"].copy()

    w_gp  = float(parameters.get("w_gp",  0.40))
    w_imp = float(parameters.get("w_imp", 0.30))
    w_str = float(parameters.get("w_str", 0.20))
    w_alt = float(parameters.get("w_alt", 0.10))

    w_total = w_gp + w_imp + w_str + w_alt
    if w_total <= 0:
        raise ValueError("At least one weight must be > 0")
    w_gp, w_imp, w_str, w_alt = (w / w_total for w in (w_gp, w_imp, w_str, w_alt))

    # ------------------------------------------------------------------
    # 2.  Parse data
    # ------------------------------------------------------------------
    customers   = df["Customer"].tolist()
    demand      = df["Quantity"].astype(float).values
    filled      = df["Filled"].astype(float).values if "Filled" in df.columns else np.zeros(len(df))
    remaining   = np.maximum(demand - filled, 0.0)
    gp_per_unit = df["$/MT"].astype(float).values
    alt         = df["Alt?"].astype(float).values
    strategic   = df["Strategic?"].astype(float).values
    importance  = df["Importance"].astype(float).values
    supply      = float(parameters.get("supply", 4000))

    n               = len(customers)
    total_remaining = remaining.sum()

    # ------------------------------------------------------------------
    # 3.  Build objective (negated — linprog minimises)
    # ------------------------------------------------------------------
    gp_coeff, imp_coeff, str_coeff, alt_coeff = _build_coefficients(
        gp_per_unit, importance, strategic, alt, total_remaining
    )

    c_per_unit = (
        w_gp  * gp_coeff
        + w_imp * imp_coeff
        + w_str * str_coeff
        + w_alt * alt_coeff
    )
    c = -c_per_unit

    # FIX #2: expose the implicit greedy ordering
    priority_rank = (-c_per_unit).argsort().argsort() + 1  # 1 = highest priority

    # ------------------------------------------------------------------
    # 4.  Solve
    # ------------------------------------------------------------------
    result = linprog(
        c,
        A_ub=np.ones((1, n)),
        b_ub=np.array([supply]),
        bounds=[(0.0, r) for r in remaining],
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"LP solver did not converge: {result.message}")

    allocated = result.x

    # ------------------------------------------------------------------
    # 5.  Outputs
    # ------------------------------------------------------------------
    cumulative_filled = filled + allocated
    # FIX #5: guard zero-demand rows
    total_fill_rate   = np.where(demand > 0, cumulative_filled / demand, 0.0)
    gp_allocated      = allocated * gp_per_unit
    gp_max_possible   = demand    * gp_per_unit

    score_gp  = w_gp  * gp_coeff  * allocated
    score_imp = w_imp * imp_coeff * allocated
    score_str = w_str * str_coeff * allocated
    score_alt = w_alt * alt_coeff * allocated

    allocation_results = pd.DataFrame({
        "Customer":         customers,
        "Priority_Rank":    priority_rank,                   # FIX #2
        "Demand_MT":        demand.astype(int),
        "Filled_MT":        filled.astype(int),
        "Remaining_MT":     remaining.astype(int),
        "Allocated_MT":     np.round(allocated, 2),
        "Total_Fill_Rate":  np.round(total_fill_rate, 4),
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
        "Score_Total":      np.round(score_gp + score_imp + score_str + score_alt, 6),
    }).sort_values("Priority_Rank").reset_index(drop=True)   # sort by priority, not achieved score

    # FIX #5: guards on summary denominators
    supply_util_pct = round(allocated.sum() / supply * 100, 2) if supply > EPS else 0.0
    gp_capture_pct  = round(gp_allocated.sum() / max(gp_max_possible.sum(), EPS) * 100, 2)

    model_summary = pd.DataFrame([{
        "Total_Supply":           int(supply),
        "Total_Demand":           int(demand.sum()),
        "Total_Filled_Before":    int(filled.sum()),
        "Total_Remaining":        int(remaining.sum()),
        "Total_Allocated":        round(allocated.sum(), 2),
        "Supply_Utilisation_Pct": supply_util_pct,
        "Total_GP_Allocated":     round(gp_allocated.sum(), 2),
        "Total_GP_Max_Possible":  round(gp_max_possible.sum(), 2),
        "GP_Capture_Pct":         gp_capture_pct,
        "Objective_Score":        round(-result.fun, 6),
        "w_gp":                   round(w_gp,  4),
        "w_imp":                  round(w_imp, 4),
        "w_str":                  round(w_str, 4),
        "w_alt":                  round(w_alt, 4),
        "Solver_Status":          result.message,
    }])

    return {
        "allocation_results": allocation_results,
        "model_summary":      model_summary,
    }


# ----------------------------------------------------------------------
# Local testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = {"customer_data": pd.read_csv("customer_data.csv")}
    params      = {"w_gp": 0.40, "w_imp": 0.30, "w_str": 0.20, "w_alt": 0.10, "supply": 7927}

    outputs = run(inputs=sample_data, parameters=params, configs={})

    print("=== Allocation Results ===")
    print(outputs["allocation_results"].to_string(index=False))
    print("\n=== Model Summary ===")
    print(outputs["model_summary"].to_string(index=False))

    outputs["allocation_results"].to_csv("allocation_results.csv", index=False)
    outputs["model_summary"].to_csv("model_summary.csv", index=False)
