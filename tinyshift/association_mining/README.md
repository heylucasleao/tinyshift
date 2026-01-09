# TransactionAnalyzer

`TransactionAnalyzer` provides association rule mining and transaction pattern analysis for market-basket-style data.

It encodes transactions to a one-hot matrix (via `TransactionEncoder`) and computes item-to-item association measures.

## Features

- One-hot encoding for transactions via `TransactionEncoder`
- Association metrics implemented:
  - `lift(antecedent, consequent)` — Lift
  - `confidence(antecedent, consequent)` — Confidence
  - `kulczynski(antecedent, consequent)` — Kulczynski measure
  - `zhang_metric(antecedent, consequent)` — Zhang's metric
  - `yules_q(antecedent, consequent)` — Yule's Q coefficient
  - `hypergeom(antecedent, consequent)` — Hypergeometric p-value
  - `mutual_information(antecedent, consequent)` — Mutual information
- Correlation matrix generation via `correlation_matrix(row_items, column_items, metric="...")`
- Model persistence: `save(filename)` / `load(filename)` (pickle)

## Usage

1. Fit the analyzer on a dataset of transactions:

```python
from tinyshift.association_mining import TransactionAnalyzer

transactions = [
    ["milk", "bread", "butter"],
    ["beer", "diapers", "bread"],
    ["milk", "beer", "diapers"],
    ["bread", "butter"]
]

analyzer = TransactionAnalyzer().fit(transactions)
```

2. Calculate associations between items:

```python
print(analyzer.confidence("milk", "bread"))
print(analyzer.zhang_metric("diapers", "beer"))
print(analyzer.yules_q("diapers", "beer"))
print(analyzer.mutual_information("milk", "bread"))
```

3. Create a correlation matrix (choose metric: `lift`, `confidence`, `kulczynski`, `zhang`, `yules_q`, `hypergeom`, `mutual_information`):

```python
matrix = analyzer.correlation_matrix(
    ["milk", "bread"],
    ["butter", "jam"],
    metric="hypergeom"
)
print(matrix)
```

## Association Metrics

Each metric measures different aspects of the relationship between items
in transactions. Use the metric that best matches the question you're asking.

| Metric                 | Range   | Short Interpretation                          | Recommended Use                                                    |
| ---------------------- | ------- | --------------------------------------------- | ------------------------------------------------------------------ |
| **Lift**               | 0 → ∞   | How much more often A and B occur together    | Overall correlation strength                                       |
| **Confidence**         | 0 → 1   | `P(B\|A)` — probability of consequent given A  | Simple directional rule strength                                   |
| **Kulczynski**         | 0 → 1   | Average of `P(B\|A)` and `P(A\|B)`               | Helpful when symmetry is desired                                   |
| **Zhang's Metric**     | -1 → 1  | Deviation from independence                   | Balanced measure less biased by marginal frequencies               |
| **Yule's Q**           | -1 → 1  | Odds-ratio-based direction and strength       | Interpreting direction (reinforcement vs opposition)               |
| **Hypergeom p-value**  | 0 → 1   | Statistical significance of co-occurrence     | Hypothesis testing for unlikely co-occurrence                      |
| **Mutual Information** | 0 → ∞   | Shared information between the two variables  | Measure of dependency that is model-agnostic                      |

