# DATA TRANSFORMATION CHECKLIST FOR OLS REGRESSION
## Quick Reference for Implementation

---

## STEP-BY-STEP TRANSFORMATION CHECKLIST

### RAW DATA INVENTORY
- [ ] Confirm 25 survey responses loaded
- [ ] Check data types: trust_score (numeric), mechanics_feedback (object/string), decision (object/string)
- [ ] List all columns: expert_name, expert_role, experience_years, ticker, decision, preference, comment, chosen_label, method_for_a, method_for_b, chosen_method, survey_variant, actionability_choice, trust_score, confidence_score, mechanics_feedback, timestamp

---

## TRANSFORMATION STEPS

### 1. PARSE COMPLEX "DECISION" STRING
**Raw format:** `decision_a:BUY|decision_b:HOLD|features:volatility,momentum,volume_avg,return_1y|attention:pass`

- [ ] Extract `num_features` by regex: `r'features:([^|]+)'` → split by comma → count
  - Expected range: 3-9
  - Expected mean: ~5.2
  - Validation: No NaNs, all > 0
  
- [ ] Extract `attention_flag` by regex: `'attention:pass'` in string → 1, else 0
  - Expected: ~48% pass (1), 52% fail (0)
  - Validation: Only values 0 or 1

**Sanity checks:**
```
df['num_features'].describe()
df['attention_flag'].value_counts()
df[['decision', 'num_features', 'attention_flag']].head(10)  # Spot-check
```

---

### 2. DEPENDENT VARIABLE: trust_score (NO TRANSFORMATION NEEDED)
- [ ] Keep as continuous (1-10 scale)
- [ ] Check for missing: `df['trust_score'].isnull().sum()` → should be 0
- [ ] Check outliers: `df['trust_score'].describe()` → should be 4-7 only
- [ ] Check data type: numeric (int or float)

**No transformations** (do NOT standardize, normalize, or log-transform for interpretation)

---

### 3. INDEPENDENT VARIABLE 1: is_counterfactual_b
**Input:** `method_for_b` column (categorical: "FEATURE_IMPORTANCE" or "COUNTERFACTUAL")

- [ ] Binary encode: `df['is_counterfactual_b'] = (df['method_for_b'] == 'COUNTERFACTUAL').astype(int)`
  - Expected values: {0, 1}
  - Expected distribution: ~64% are 1 (counterfactual), 36% are 0 (feature importance)
  
- [ ] Check for missing: `df['is_counterfactual_b'].isnull().sum()` → should be 0

**Sanity check:**
```
df['is_counterfactual_b'].value_counts()  # Should show counts for 0 and 1
```

---

### 4. INDEPENDENT VARIABLE 2: accuracy_score
**Input:** `mechanics_feedback` column (categorical: 3 classes)

- [ ] Create ordinal encoding mapping:
  ```
  'No, they seem theoretical'  → 0
  'Somewhat accurate'          → 1
  'Yes, highly accurate'       → 2
  ```

- [ ] Apply mapping:
  ```
  df['accuracy_score'] = df['mechanics_feedback'].map({
      'No, they seem theoretical': 0,
      'Somewhat accurate': 1,
      'Yes, highly accurate': 2
  })
  ```

- [ ] Check for missing after mapping: `df['accuracy_score'].isnull().sum()`
  - If > 0: Find mismatched values and add to mapping

- [ ] Verify distribution:
  ```
  df['accuracy_score'].value_counts().sort_index()
  # Expected: mostly 2 (72% "highly accurate")
  ```

**Why ordinal, not one-hot?** Order matters (2 > 1 > 0), so ordinal preserves meaning.

---

### 5. INDEPENDENT VARIABLE 3: num_features
**Already extracted in Step 1**

- [ ] Verify: `df['num_features'].dtype` → numeric (int or float)
- [ ] Range check: `df['num_features'].min()` ≥ 3 AND `df['num_features'].max()` ≤ 9
- [ ] No missing: `df['num_features'].isnull().sum()` → should be 0

**Interpretation:** Continuous predictor (3-9 features shown in explanation)

---

### 6. INDEPENDENT VARIABLE 4: confidence_score
**Input:** `confidence_score` column (already numeric)

- [ ] Keep as-is (continuous, 1-10 scale)
- [ ] Check for missing: `df['confidence_score'].isnull().sum()` → should be 0
- [ ] Range check: `df['confidence_score'].min()` ≥ 1 AND `df['confidence_score'].max()` ≤ 10

**No transformations needed**

---

### 7. INDEPENDENT VARIABLE 5: attention_flag
**Already extracted in Step 1**

- [ ] Verify: `df['attention_flag'].dtype` → numeric (int)
- [ ] Values only {0, 1}: `df['attention_flag'].unique()` 
- [ ] No missing: `df['attention_flag'].isnull().sum()` → should be 0

---

## COLUMNS TO DROP (LOW VARIANCE)

- [ ] `expert_name` → All "Anonymous" or 1 name (no variance)
- [ ] `expert_role` → All "Portfolio Manager" (no variance)
- [ ] `experience_years` → All 5 (no variance)
- [ ] `comment` → Text field (not used in regression)
- [ ] `chosen_label` → Redundant with preference
- [ ] `actionability_choice` → 100% missing
- [ ] `chosen_method` → Redundant with method_for_a/b
- [ ] `survey_variant` → All same version
- [ ] `timestamp` → Not a predictor
- [ ] `ticker` → Not used in this model
- [ ] `decision` → Already parsed into num_features, attention_flag

---

## DATA QUALITY CHECKS

### Missing Values
- [ ] Run: `reg_data.isnull().sum()`
- [ ] Expected: All zeros across 6 columns (trust_score, is_counterfactual_b, accuracy_score, num_features, confidence_score, attention_flag)
- [ ] If any NaNs: Investigate source and fix

### Descriptive Statistics
- [ ] Run: `reg_data.describe()`
- [ ] Expected ranges:
  ```
  trust_score:         Mean ~5.3,  Min 4,    Max 7
  is_counterfactual_b: Mean ~0.64, Min 0,    Max 1
  accuracy_score:      Mean ~1.7,  Min 0,    Max 2
  num_features:        Mean ~5.2,  Min 3,    Max 9
  confidence_score:    Mean ~4.8,  Min 1,    Max 10
  attention_flag:      Mean ~0.48, Min 0,    Max 1
  ```

### Variance (All Should Be > 0)
- [ ] Run: `reg_data.var()`
- [ ] Check: No column has variance = 0 (would indicate constant values)

### Correlations (Check for Multicollinearity)
- [ ] Run: `reg_data.corr()`
- [ ] Check: No correlation > 0.80 between any two X variables
- [ ] Expected: Low-to-moderate correlations (r < 0.5)

---

## FINAL REGRESSION DATASET

### Dimensions
- [ ] Rows: 25 (one observation per expert evaluation)
- [ ] Columns: 6 (1 Y + 5 X)
- [ ] Total cells: 150 (25 × 6)

### Data Structure (Before Running OLS)
```
Variable             Type         Scale       Missing  Variance  Usage
─────────────────────────────────────────────────────────────────────────
trust_score          Continuous   1-10        0        0.46      Y (DV)
is_counterfactual_b  Binary       {0,1}       0        0.24      X1
accuracy_score       Ordinal      {0,1,2}     0        0.31      X2
num_features         Continuous   3-9         0        2.29      X3
confidence_score     Continuous   1-10        0        1.69      X4
attention_flag       Binary       {0,1}       0        0.25      X5
```

---

## PRE-REGRESSION DIAGNOSTICS (Optional but Recommended)

- [ ] **Linearity:** Plot X vs Y for each predictor
  - Look for linear trend (not curved)
  
- [ ] **Normality of DV:** Histogram of trust_score
  - Should be roughly bell-shaped (n=25 is small, so strict normality unlikely)
  
- [ ] **Variance inflation factors (VIF):** Check multicollinearity
  - All VIF < 5 → No problem
  
- [ ] **Sample size adequacy:** n=25 vs p=5 predictors
  - Rule of thumb: n > 50 + 8×p = 50 + 40 = 90 (you have 25)
  - Your model is underpowered but acceptable for exploratory thesis work

---

## SAVE CLEANED DATASET

- [ ] Export to CSV: `reg_data.to_csv('regression_ready.csv', index=False)`
- [ ] Save Python pickle: `reg_data.to_pickle('regression_ready.pkl')`
- [ ] Document transformation log:
  ```
  - 25 rows loaded from MongoDB
  - num_features extracted via regex from 'decision' field
  - attention_flag extracted via string matching
  - is_counterfactual_b binary-encoded from method_for_b
  - accuracy_score ordinal-encoded from mechanics_feedback
  - confidence_score kept as-is
  - 10 columns dropped (low/zero variance + non-predictor fields)
  - Final dataset: 25 rows × 6 columns, 0 missing values
  - Ready for OLS regression
  ```

---

## READY FOR OLS

- [ ] regression_ready.csv exists
- [ ] All 6 columns present (1 Y + 5 X)
- [ ] 25 rows, no missing values
- [ ] All numerical (no text, dates, or categoricals)
- [ ] Descriptive stats match expectations
- [ ] Correlation matrix shows no collinearity
- [ ] Can now run: `python ols_regression.py`

---

## COMMAND TO VALIDATE TRANSFORMATION

```python
import pandas as pd
import numpy as np

# Load cleaned data
reg_data = pd.read_csv('regression_ready.csv')

# Validate
assert reg_data.shape == (25, 6), f"Expected (25,6), got {reg_data.shape}"
assert reg_data.isnull().sum().sum() == 0, "Found missing values!"
assert reg_data['trust_score'].min() >= 4 and reg_data['trust_score'].max() <= 7
assert set(reg_data['is_counterfactual_b'].unique()) == {0, 1}
assert set(reg_data['accuracy_score'].unique()).issubset({0, 1, 2})
assert reg_data['num_features'].min() >= 3 and reg_data['num_features'].max() <= 9
assert set(reg_data['attention_flag'].unique()) == {0, 1}

print("✓ All validation checks passed!")
print("\nData summary:")
print(reg_data.describe())
print("\nReady for OLS regression!")
```
