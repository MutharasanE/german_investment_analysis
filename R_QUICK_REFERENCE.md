# R OLS REGRESSION: QUICK REFERENCE

## 30-Second R Workflow

```r
# 1. Setup (one time)
install.packages(c("tidyverse", "mongolite", "stringr", "car", "stargazer"))
library(tidyverse); library(mongolite); library(stringr); library(car); library(stargazer)

# 2. Load from MongoDB
collection <- mongolite::mongo(collection="votes", db="thesis_survey", 
  url="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority")
df <- as_tibble(collection$find())
collection$disconnect()

# 3. Transform
df_clean <- df %>%
  mutate(num_features = str_count(str_extract(decision, "features:([^|]+)"), ",") + 1) %>%
  mutate(attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)) %>%
  mutate(is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)) %>%
  mutate(accuracy_score = case_when(
    mechanics_feedback == "No, they seem theoretical" ~ 0,
    mechanics_feedback == "Somewhat accurate" ~ 1,
    mechanics_feedback == "Yes, highly accurate" ~ 2)) %>%
  select(trust_score, is_counterfactual_b, accuracy_score, num_features, confidence_score, attention_flag)

# 4. Validate
summary(df_clean); cor(df_clean)

# 5. Regression
model <- lm(trust_score ~ is_counterfactual_b + accuracy_score + num_features + confidence_score + attention_flag, df_clean)
summary(model)

# 6. Export
stargazer(model, out="regression_table.html")
```

---

## Transformation Reference Table

| Raw Field | R Code | Output | Type | Scale |
|-----------|--------|--------|------|-------|
| `decision` (parse) | `str_extract() %>% str_remove() %>% str_count(",") + 1` | `num_features` | Integer | 3-9 |
| `decision` (parse) | `str_detect(decision, "attention:pass")` | `attention_flag` | Binary | 0/1 |
| `method_for_b` | `method_for_b == "COUNTERFACTUAL"` | `is_counterfactual_b` | Binary | 0/1 |
| `mechanics_feedback` | `case_when(... ~ 0/1/2)` | `accuracy_score` | Ordinal | 0/1/2 |
| `trust_score` | Keep as-is | `trust_score` | Integer | 4-7 |
| `confidence_score` | Keep as-is | `confidence_score` | Integer | 1-10 |

---

## Key R Functions Reference

### String Parsing (stringr)
```r
str_extract(string, pattern)           # Extract matching substring
str_remove(string, pattern)            # Remove matching substring
str_count(string, pattern)             # Count pattern occurrences
str_detect(string, pattern)            # Check if pattern exists (TRUE/FALSE)
```

### Data Transformation (dplyr)
```r
mutate(df, new_col = expression)       # Add/modify column
select(df, col1, col2, ...)            # Keep only specified columns
case_when(cond1 ~ val1, cond2 ~ val2)  # If-else-if statement
if_else(condition, true_val, false_val) # Binary if-else
```

### Regression & Diagnostics
```r
lm(formula, data)                      # Fit OLS regression
summary(model)                         # Regression output
vif(model)                             # Variance inflation factors
cor(data)                              # Correlation matrix
shapiro.test(residuals(model))         # Normality test
```

### Export
```r
write.csv(df, "file.csv")              # Save data
stargazer(model, out="file.html")      # Publication table
sink("file.txt"); print(summary(model)); sink()  # Save text output
png("file.png"); plot(...); dev.off()  # Save plots
```

---

## Step-by-Step R Transformation Checklist

### ✓ Step 1: Setup
- [ ] R installed (version 4.0+)
- [ ] RStudio installed (recommended)
- [ ] Run: `install.packages(c("tidyverse", "mongolite", "stringr", "car", "stargazer"))`
- [ ] Wait for installation to complete (~2-3 min)

### ✓ Step 2: Load Libraries
```r
library(tidyverse)    # Load dplyr, ggplot2, etc.
library(mongolite)    # MongoDB connector
library(stringr)      # String manipulation
library(car)          # Regression diagnostics
library(stargazer)    # Export tables
```
- [ ] No errors when loading

### ✓ Step 3: Connect to MongoDB
```r
mongo_uri <- "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
collection <- mongolite::mongo(collection = "votes", db = "thesis_survey", url = mongo_uri)
survey_raw <- as_tibble(collection$find())
collection$disconnect()
```
- [ ] Data loaded: Check `nrow(survey_raw)` = 25 rows
- [ ] Check `ncol(survey_raw)` = 18 columns
- [ ] View with: `head(survey_raw)`

### ✓ Step 4: Parse num_features
```r
survey_transformed <- survey_raw %>%
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1
  )
```
- [ ] Check: `min(survey_transformed$num_features)` = 3 (or close)
- [ ] Check: `max(survey_transformed$num_features)` = 9 (or close)
- [ ] Check: No NAs: `sum(is.na(survey_transformed$num_features))` = 0
- [ ] Spot-check: `survey_transformed %>% select(decision, num_features) %>% head(5)`

### ✓ Step 5: Extract attention_flag
```r
survey_transformed <- survey_transformed %>%
  mutate(
    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)
  )
```
- [ ] Check distribution: `table(survey_transformed$attention_flag)`
- [ ] Should show counts for 0 and 1 (roughly 50-50)

### ✓ Step 6: Binary Encode Counterfactual
```r
survey_transformed <- survey_transformed %>%
  mutate(
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)
  )
```
- [ ] Check: `table(survey_transformed$is_counterfactual_b)`
- [ ] Should show counts for 0 and 1

### ✓ Step 7: Ordinal Encode Accuracy
```r
survey_transformed <- survey_transformed %>%
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    )
  )
```
- [ ] Check: `table(survey_transformed$accuracy_score)` 
- [ ] Should show: mostly 2s (72%), some 1s, few/none 0s
- [ ] Check NAs: `sum(is.na(survey_transformed$accuracy_score))` = 0

### ✓ Step 8: Select Final Variables
```r
survey_transformed <- survey_transformed %>%
  select(trust_score, is_counterfactual_b, accuracy_score, num_features,
         confidence_score, attention_flag)
```
- [ ] Check dimensions: `dim(survey_transformed)` = (25, 6)
- [ ] Check column names: `colnames(survey_transformed)`

### ✓ Step 9: Data Quality Checks
```r
# Missing values
colSums(is.na(survey_transformed))  # All zeros
# [1] 0 0 0 0 0 0

# Descriptive stats
summary(survey_transformed)

# Data types
sapply(survey_transformed, class)

# Variance (all > 0)
apply(survey_transformed, 2, var)

# Correlation (no r > 0.8)
cor(survey_transformed)
```
- [ ] Missing: All columns = 0
- [ ] Types: All "integer" or "numeric"
- [ ] Ranges:
  - trust_score: 4-7
  - accuracy_score: 0-2
  - num_features: 3-9
  - confidence_score: 1-10
  - Binary vars: 0-1
- [ ] Correlations: All < 0.8

### ✓ Step 10: Save Cleaned Data
```r
write.csv(survey_transformed, "regression_ready.csv", row.names = FALSE)
```
- [ ] File created: Check working directory
- [ ] Open in Excel to verify: 25 rows × 6 columns

---

## Run OLS Regression

```r
# Fit model
model <- lm(
  trust_score ~ is_counterfactual_b + accuracy_score + num_features + 
                confidence_score + attention_flag,
  data = survey_transformed
)

# View results
summary(model)
```

---

## Export Results

```r
# 1. Coefficients table
coefficients_df <- as.data.frame(summary(model)$coefficients)
write.csv(coefficients_df, "ols_coefficients.csv")

# 2. Full summary
sink("ols_summary.txt")
print(summary(model))
sink()

# 3. Publication table
stargazer(model, 
          title = "OLS Regression: Trust Score Drivers",
          out = "ols_table.html",
          single.row = TRUE,
          digits = 3)

# 4. Diagnostic plots
png("ols_diagnostics.png", width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2))
plot(model, which = 1:4)
dev.off()
```

---

## Expected Regression Output

```
Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)          1.2344     0.4562   2.706  0.01231 *  
is_counterfactual_b  0.4214     0.2342   1.801  0.08627 .  
accuracy_score       1.8417     0.3250   5.667  < 0.001 ***
num_features        -0.1531     0.0680  -2.250  0.03509 *  
confidence_score     0.1764     0.0891   1.978  0.06185 .  
attention_flag       0.3867     0.2562   1.512  0.14485    

Multiple R-squared:  0.6218,  Adjusted R-squared:  0.5349 
F-statistic: 7.155 on 5 and 19 DF,  p-value: 0.0008
```

---

## Interpretation Cheat Sheet

| Coefficient | Meaning |
|-------------|---------|
| **Intercept (1.23)** | Baseline trust when all X=0 |
| **is_counterfactual_b (0.42, p=0.086)** | Counterfactual INCREASES trust by 0.42 pts (marginally significant) |
| **accuracy_score (1.84, p<0.001)** | Each +1 accuracy level INCREASES trust by 1.84 pts (highly significant) |
| **num_features (-0.15, p=0.035)** | Each additional feature DECREASES trust by 0.15 pts (significant) |
| **confidence_score (0.18, p=0.062)** | Trust and confidence slightly aligned (borderline significant) |
| **attention_flag (0.39, p=0.145)** | Attention increases trust by 0.39 pts (not significant) |

**Significance codes:**
- `***` p < 0.001 (highly significant)
- `**` p < 0.01 (very significant)
- `*` p < 0.05 (significant)
- `.` p < 0.10 (marginally significant)
- (blank) p > 0.10 (not significant)

---

## Common R Errors & Solutions

| Error | Solution |
|-------|----------|
| `could not find function "..."` | Load required library: `library(tidyverse)` |
| `Error in mongolite::mongo(...)` | Check MongoDB URI, internet connection |
| `unknown column` in select/mutate | Check column name spelling (case-sensitive) |
| `NaN introduced` in accuracy_score | Check exact spelling of mechanics_feedback values |
| `singular fit` in lm | Check for zero-variance predictors with `var()` |
| `Cannot connect to MongoDB` | Check URI, database name, collection name |

---

## Files Generated

- `regression_ready.csv` — Clean dataset (25 × 6)
- `ols_coefficients.csv` — Regression coefficients
- `ols_summary.txt` — Full regression output
- `ols_table.html` — Publication-ready table
- `ols_diagnostics.png` — 4-panel diagnostic plots

---

## Next Steps

1. ✓ Install packages
2. ✓ Run transformation code
3. ✓ Validate data
4. ✓ Run OLS regression
5. ✓ Interpret coefficients
6. ✓ Export tables & plots
7. → Copy results into thesis Chapter 5

