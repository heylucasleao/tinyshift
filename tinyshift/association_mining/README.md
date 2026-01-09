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

| Metric                  | Range  | Interpretation                                | Question You Want to Answer                                    | Recommended Usage                                             |
| ----------------------- | ------ | --------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **Lift**                | 0 → ∞  | Correlation between antecedent and consequent | “How much more often do `A` and `B` occur together than expected?” | Good for overall correlation strength                         |
| **Confidence**          | 0 → 1  | `P(B\|A)` — probability of consequent given `A`    | “If `A` occurs, how likely is `B`?”                                | Simple baseline, but may be misleading with rare items        |
| **Kulczynski**         | 0 → 1  | Average of `P(B\|A)` and `P(A\|B)`            | “Are `A` and `B` mutually predictive?”                             | Balances asymmetric confidence; useful with imbalanced supports |
| **Zhang’s Metric**      | -1 → 1 | Deviation from statistical independence       | “How far is the `A → B` relation from being independent?”          | Balanced measure less biased by item frequency                |
| **Yule’s Q**            | -1 → 1 | Odds ratio-based association                  | “Do `A` and `B` strongly reinforce or oppose each other?”          | Best when interpreting direction and strength of association  |
| **Hypergeom p-value**   | 0 → 1  | Statistical significance of co-occurrence     | “Is the co-occurrence of `A` and `B` statistically significant?”   | Use when testing whether an association is unlikely by chance |
| **Mutual Information** | 0 → ∞  | Shared information between items              | “How much information does one item give about the other?”     | Captures non-linear associations; scale depends on distributions |



