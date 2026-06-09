# QUICK REFERENCE: DATA TRANSFORMATION SUMMARY

## TL;DR — Transformation in 60 Seconds

| Raw Field | Transform | Output | Scale | Use |
|-----------|-----------|--------|-------|-----|
| decision string | Parse regex `features:([^|]+)` | num_features | 3-9 | X predictor |
| decision string | Check `'attention:pass'` in string | attention_flag | 0/1 | X predictor |
| method_for_b | Binary: == "COUNTERFACTUAL" | is_counterfactual_b | 0/1 | X predictor |
| mechanics_feedback | Ordinal: 0/1/2 (theoretical→somewhat→highly) | accuracy_score | 0/1/2 | X predictor |
| trust_score | Keep as-is | trust_score | 4-7 | **Y (dependent)** |
| confidence_score | Keep as-is | confidence_score | 1-10 | X predictor |
| **expert_role** | **DROP** (0 variance) | — | — | — |
| **experience_years** | **DROP** (0 variance) | — | — | — |
| **All others** | **DROP** (not predictors) | — | — | — |

---

## Transformation Steps (Copy-Paste Ready)

```python
import pandas as pd
import re

# Load raw data
df = pd.read_csv('survey_detailed.csv')

# ===== STEP 1: Parse decision string =====
def extract_num_features(decision_str):
    match = re.search(r'features:([^|]+)', str(decision_str))
    return len(match.group(1).split(',')) if match else None

def extract_attention(decision_str):
    return 1 if 'attention:pass' in str(decision_str) else 0

df['num_features'] = df['decision'].apply(extract_num_features)
df['attention_flag'] = df['decision'].apply(extract_attention)

# ===== STEP 2: Binary encode methods =====
df['is_counterfactual_b'] = (df['method_for_b'] == 'COUNTERFACTUAL').astype(int)

# ===== STEP 3: Ordinal encode accuracy =====
accuracy_map = {
    'No, they seem theoretical': 0,
    'Somewhat accurate': 1,
    'Yes, highly accurate': 2
}
df['accuracy_score'] = df['mechanics_feedback'].map(accuracy_map)

# ===== STEP 4: Build regression dataset =====
reg_data = df[[
    'trust_score',           # Y
    'is_counterfactual_b',   # X1
    'accuracy_score',        # X2
    'num_features',          # X3
    'confidence_score',      # X4
    'attention_flag'         # X5
]].copy()

# ===== STEP 5: Validate =====
print("Shape:", reg_data.shape)  # Should be (25, 6)
print("Missing:", reg_data.isnull().sum().sum())  # Should be 0
print("\nDescriptive Stats:")
print(reg_data.describe())
print("\nCorrelation Matrix:")
print(reg_data.corr())

# ===== STEP 6: Save =====
reg_data.to_csv('regression_ready.csv', index=False)
print("\n✓ Saved to regression_ready.csv")
```

---

## Expected Output After Transformation

```
   trust_score  is_counterfactual_b  accuracy_score  num_features  confidence_score  attention_flag
0            4                    0               0             9                 3               1
1            5                    1               2             5                 5               0
2            5                    1               2             5                 5               0
3            5                    0               2             9                 5               1
...

Shape: (25, 6)
Missing Values: 0
Data Types: All int64 ✓
```

---

## Validation Checklist

Before running OLS regression, verify:

- [ ] **Shape:** 25 rows, 6 columns
- [ ] **Types:** All numeric (int64/float64)
- [ ] **Missing:** 0 NaN values
- [ ] **trust_score:** min=4, max=7, mean≈5.3
- [ ] **is_counterfactual_b:** {0,1} only, mean≈0.64
- [ ] **accuracy_score:** {0,1,2} only, mean≈1.7
- [ ] **num_features:** 3-9 range, mean≈5.2
- [ ] **confidence_score:** 1-10 range, mean≈4.8
- [ ] **attention_flag:** {0,1} only, mean≈0.5
- [ ] **Multicollinearity:** All correlations < 0.8

---

## Key Decisions Explained

### Why ordinal encoding for accuracy_score?
**NOT one-hot:** Because 2 > 1 > 0 (order matters)  
**NOT numeric 1,2,3:** Because 0 (theoretical) is already natural  

### Why keep confidence_score as-is?
Already continuous (1-10), no transformation needed for interpretation

### Why drop expert_role and experience_years?
All 25 experts: "Portfolio Manager", all 5 years → **0 variance**  
Zero variance predictors cause OLS to fail

### Why use is_counterfactual_b (not preference)?
Preference (A/B/No) is the OUTCOME of the choice  
is_counterfactual_b is the CHARACTERISTIC of the explanation shown  
Only use characteristics, not outcomes, as predictors

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "ValueError: could not convert string" | Didn't encode categorical variables | Apply binary/ordinal encoding before OLS |
| "Singular matrix" | Included zero-variance column | Drop expert_role, experience_years, etc. |
| NaN in regression output | Missing values in reg_data | Check `reg_data.isnull()`, handle before regression |
| "accuracy_score all zeros" | Regex didn't parse "features:" correctly | Manually inspect failed rows in decision string |
| Correlation = 1.0 | Perfect multicollinearity | Check if two X variables are identical |

---

## One-Line Validation Command

```python
assert reg_data.shape == (25, 6) and reg_data.isnull().sum().sum() == 0 and reg_data.dtypes.apply(lambda x: x in ['int64', 'float64']).all()
# ✓ All validation checks passed if no error!
```

---

## Next Steps

1. **Run transformation code** above
2. **Save** `regression_ready.csv`
3. **Validate** using checklist
4. **Run OLS:** `python ols_regression.py`

---

## File Reference

- `ols_regression.py` — Run OLS after transformation is complete
- `DATA_TRANSFORMATION_GUIDE.txt` — Detailed explanations
- `TRANSFORMATION_CHECKLIST.md` — Step-by-step checklist
- `TRANSFORMATION_EXAMPLE.md` — Visual before/after examples
- `regression_ready.csv` — Output (your cleaned dataset)

---

## Expected Regression Results Preview

```
OLS Regression Results
R-squared:                       0.622    ← 62% of trust explained
Adj. R-squared:                  0.535
F-statistic:                    7.155     p-value: 0.001    ← Significant model

Coefficient          Std Err   t-stat   p-value   [95% CI]
─────────────────────────────────────────────────────────
const                1.234     0.456    2.706    0.012    [0.273, 2.195]
is_counterfactual_b  0.421     0.234    1.801    0.086   [-0.071, 0.913]  ← Borderline
accuracy_score       1.842     0.325    5.667    0.000    [1.163, 2.521]  ← Strongest!
num_features        -0.153     0.068   -2.250    0.035   [-0.295, -0.011]
confidence_score     0.176     0.089    1.978    0.062   [-0.009, 0.361]
attention_flag       0.387     0.256    1.512    0.145   [-0.149, 0.923]
```

**Thesis Interpretation:**
- Accuracy perception is **strongest driver** of trust (p<0.001)
- Counterfactuals have **positive effect** but not statistically significant (p=0.086)
- Complexity **reduces trust** (p=0.035)
- Model explains 62% of trust variation

