# Detailed Comparison: HBOS vs. SPAD vs. SPAD+ vs. PCAReconstructionError

## **Overview**
| Metric               | HBOS                          | SPAD                          | SPAD+                         | PCAReconstructionError         |
|----------------------|-------------------------------|-------------------------------|-------------------------------|--------------------------------|
| **Approach**         | Histogram-based               | Probabilistic (log-P)         | Probabilistic + PCA           | PCA Reconstruction             |
| **Independence**     | Assumes independence          | Assumes independence          | Captures correlations         | Captures correlations          |
| **Efficiency**       | Very fast                     | Very fast                          | Moderate                      | Moderate-fast                  |

---

## **Pros and Cons**

### **HBOS**
✅ **Pros**:
- Extremely fast, ideal for high-dimensional data
- Simple implementation
- Effective for univariate anomalies

❌ **Cons**:
- Misses correlated anomalies
- Requires careful bin selection
- Poor with multimodal distributions

### **SPAD**
✅ **Pros**:
- Extremely fast, ideal for high-dimensional data
- Probabilistic foundation
- Handles zero probabilities via smoothing
- Good interpretability

❌ **Cons**:
- Still assumes independence
- Discretization affects performance

### **SPAD+**
✅ **Pros**:
- Accounts for correlations via PCA
- Better multivariate detection
- Retains probabilistic interpretation

❌ **Cons**:
- PCA adds computation
- Still has discretization issues

### **PCAReconstructionError**
✅ **Pros**:
- Naturally handles correlations
- No independence assumptions
- Simple squared reconstruction-error score

❌ **Cons**:
- Requires sufficient samples for PCA
- Sensitive to feature scale; standardize heterogeneous features first
- Loses feature interpretability

---

## **Score and Probability**

| Metric               | HBOS                          | SPAD                          | SPAD+                         | PCAReconstructionError         |
|----------------------|-------------------------------|-------------------------------|-------------------------------|--------------------------------|
| **Score Formula**    | Sum of `-log(bin probability)` | Sum of `-log(bin probability)` | Same score over original and PCA-extended features | Squared reconstruction error |
| **Score Direction**  | ↑ (high = anomalous)          | ↑ (high = anomalous)          | ↑ (high = anomalous)          | ↑ (high = anomalous)          |
| **Probability**      | Product of per-feature bin probabilities | Product of per-feature bin probabilities | Product over original and PCA-extended features | Not probabilistic |

---

## **When to Use Each?**

- **HBOS**: 
  - Need speed with high-dim data 
  - Preliminary anomaly screening
  - When independence assumption holds

- **SPAD**:
  - Need speed with high-dim data 
  - Probabilistic interpretation needed
  - When independence assumption holds

- **SPAD+**:
  - Correlated features present
  - Willing to trade speed for accuracy
  - Need probabilistic interpretation

- **PCAReconstructionError**:
  - Linear correlations expected
  - Continuous feature space
  - When reconstruction makes sense

---
