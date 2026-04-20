# Simple Model — Allocation Results

## Run Parameters

| Parameter | Value |
|---|---|
| Supply available | 7,927 MT |
| Total customer demand | 18,794 MT |
| Already filled (pre-run) | 7,469 MT |
| Remaining unfilled demand | 11,325 MT |
| Supply vs remaining | 7,927 / 11,325 = **70% of remaining demand can be met** |
| Weights | GP 40%, Importance 30%, Strategic 20%, Alt 10% |

---

## Summary Stats

| Metric | Value |
|---|---|
| Total allocated | 7,927 MT (100% supply utilised) |
| Total GP generated | $909,349 |
| GP capture vs theoretical max | 43.4% |
| Solver status | Optimal |

GP capture is 43.4% of the theoretical maximum (all demand fully filled at full GP). This reflects the supply constraint — we can only fill 70% of remaining demand, and the model is directing supply to the highest-value customers first.

---

## Allocation Outcomes

### Fully filled (100% total fill rate)
22 of 30 customers reach 100% total fill rate after this allocation. Most of these had significant prior fills already, so the remaining volumes were small enough to cover fully.

| Customer | Demand | Pre-filled | Allocated | Note |
|---|---|---|---|---|
| C04 | 1,052 | 967 | 85 | Only 85 MT remaining — gets all of it |
| C18 | 278 | 250 | 28 | Only 28 MT remaining |
| C28 | 703 | 572 | 131 | High importance + contract firm |
| C13 | 570 | 147 | 423 | Top priority rank — Importance 4, Strategic 4 |
| C23 | 1,281 | 0 | 1,281 | Largest single allocation — high GP + Strategic 4 |

### Partially filled
| Customer | Demand | Pre-filled | Allocated | Total Fill Rate | Note |
|---|---|---|---|---|---|
| C14 | 1,199 | 788 | 9 | 66.5% | Supply nearly exhausted by this point in priority order |

### Zero allocated (supply exhausted)
| Customer | Remaining | Priority Rank | Reason |
|---|---|---|---|
| C16 | 258 | 24 | Importance 1, Strategic 2 |
| C17 | 467 | 25 | Importance 2, Strategic 2, moderate GP |
| C26 | 187 | 26 | Importance 1, Strategic 1 — lowest qualitative scores |
| C11 | 932 | 27 | Alt=4 (can go elsewhere), low importance |
| C29 | 703 | 28 | Low across all dimensions |
| C25 | 217 | 29 | Importance 1, Strategic 1 |
| C09 | 232 | 30 | Lowest priority — Importance 1, Strategic 1, Alt 3 |

---

## Priority Logic

The model scores each customer on a per-MT basis using a weighted combination of four normalised signals:

- **GP score**: how much gross profit per MT relative to the best customer
- **Importance score**: customer importance (4=highest)
- **Strategic score**: strategic value (4=highest)
- **Alt score**: 5 minus Alt rating — customers with fewer alternatives score higher (Alt 1 = sole supplier = highest retention risk)

All four signals share the same normaliser (total remaining supply), making the weights directly comparable. The LP then allocates greedily in priority order — highest-scoring customers are filled to their remaining demand cap first.

---

## Key Observations

**C23 received the largest single allocation (1,281 MT)** despite having zero prior fill. It ranked #2 in priority because it combines the highest GP per MT ($131), Importance 3, Strategic 4, and Alt 1 (no alternatives) — a strong signal across all dimensions.

**C11 (932 MT remaining) gets nothing** despite being a large customer. Its score is hurt by Alt=4 (can easily go elsewhere) and low Importance/Strategic ratings. With 7 customers ranked above it and supply limited, it falls below the cutoff.

**The Alt score had a decisive swing effect on C26.** C26 has decent GP ($131/MT) but Importance 1, Strategic 1, and Alt 1. The Alt=1 gives it a goodretention signal, but the qualitative scores are too weak to compete. It ranks 26th.

**Customers with small remaining demand (C04, C18, C28) are efficiently cleared** — their remaining volumes are small enough that filling them costs little supply while delivering 100% fill rate and goodwill.
