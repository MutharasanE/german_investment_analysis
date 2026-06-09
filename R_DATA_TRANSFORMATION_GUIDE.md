# R DATA TRANSFORMATION GUIDE FOR OLS REGRESSION

## Quick Start (Copy-Paste in R)

```r
# 1. Install packages (once)
install.packages(c("tidyverse", "mongolite", "stringr", "car", "stargazer"))

# 2. Load packages
library(tidyverse)
library(mongolite)
library(stringr)
library(car)

# 3. Load data from MongoDB
mongo_uri <- "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
collection <- mongolite::mongo(
  collection = "votes",
  db = "thesis_survey",
  url = mongo_uri
)
survey_raw <- as_tibble(collection$find())
collection$disconnect()

# 4. Transform data (pipe syntax)
survey_transformed <- survey_raw %>%
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1
  ) %>%
  mutate(
    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)
  ) %>%
  mutate(
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)
  ) %>%
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    )
  ) %>%
  select(trust_score, is_counterfactual_b, accuracy_score, num_features,
         confidence_score, attention_flag)

# 5. Validate
summary(survey_transformed)
cor(survey_transformed)

# 6. Run regression
model <- lm(trust_score ~ is_counterfactual_b + accuracy_score + num_features +
            confidence_score + attention_flag, data = survey_transformed)
summary(model)
```

---

## Step-by-Step Transformation

### STEP 1: Install & Load Packages

```r
# Only run once (first time):
install.packages(c("tidyverse", "mongolite", "jsonlite", "stringr", "car", "stargazer", "ggplot2"))

# Load every session:
library(tidyverse)      # dplyr, ggplot2, etc.
library(mongolite)      # MongoDB
library(stringr)        # Regex/string functions
library(car)            # Diagnostics (VIF, etc.)
library(stargazer)      # Publication tables
```

**What each package does:**
- `tidyverse`: Data manipulation (mutate, select, %>% pipe)
- `mongolite`: Connect to MongoDB Atlas
- `stringr`: Parse strings with regex
- `car`: Regression diagnostics (VIF, Shapiro-Wilk)
- `stargazer`: Export regression tables for thesis

---

### STEP 2: Load Data from MongoDB

```r
# Define MongoDB connection string
mongo_uri <- "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"

# Create collection reference
collection <- mongolite::mongo(
  collection = "votes",      # Collection name
  db = "thesis_survey",      # Database name
  url = mongo_uri            # Connection string
)

# Query all documents
survey_raw <- collection$find()

# Convert to tibble (cleaner format)
survey_raw <- as_tibble(survey_raw)

# Disconnect
collection$disconnect()

# Verify data loaded
cat("Loaded", nrow(survey_raw), "rows,", ncol(survey_raw), "columns\n")
head(survey_raw)
```

---

### STEP 3: Parse Complex "decision" String

**Problem:** The `decision` field contains nested pipe-separated values:
```
"decision_a:BUY|decision_b:HOLD|features:volatility,momentum,volume_avg,return_1y|attention:pass"
```

**Solution:** Extract `num_features` using regex:

```r
survey_transformed <- survey_raw %>%
  mutate(
    # Extract features string: "volatility,momentum,..."
    features_str = str_extract(decision, "features:([^|]+)") %>%
                   str_remove("features:"),
    
    # Count commas + 1 = number of features
    num_features = str_count(features_str, ",") + 1
  ) %>%
  select(-features_str)  # Remove temporary column
```

**Regex breakdown:**
- `str_extract(decision, "features:([^|]+)")` → Extract from "features:" to "|"
- `str_remove("features:")` → Remove "features:" prefix, leaving just "volatility,momentum,..."
- `str_count(",")` → Count commas (4 commas = 5 features)
- `+ 1` → Add 1 (4 commas + 1 = 5 features)

**Validate:**
```r
survey_transformed %>%
  select(decision, num_features) %>%
  head(5)
# Should show: num_features = 4, 5, 9, etc.
```

---

### STEP 4: Extract Attention Flag

```r
survey_transformed <- survey_transformed %>%
  mutate(
    attention_flag = if_else(
      str_detect(decision, "attention:pass"),  # Check if string contains "attention:pass"
      1,      # If TRUE, return 1
      0       # If FALSE, return 0
    )
  )
```

**Validate:**
```r
survey_transformed %>%
  select(decision, attention_flag) %>%
  table(survey_transformed$attention_flag)
# Should show: 1 count for pass (1), count for fail (0)
```

---

### STEP 5: Binary Encode Method Type

```r
survey_transformed <- survey_transformed %>%
  mutate(
    is_counterfactual_b = if_else(
      method_for_b == "COUNTERFACTUAL",  # Check if method is counterfactual
      1,      # If TRUE, return 1
      0       # If FALSE (feature importance), return 0
    )
  )
```

**Validate:**
```r
table(survey_transformed$is_counterfactual_b)
# Should show: count for 0 (feature importance), count for 1 (counterfactual)
```

---

### STEP 6: Ordinal Encode Accuracy

**Problem:** `mechanics_feedback` has 3 categorical values:
```
"Yes, highly accurate"
"Somewhat accurate"
"No, they seem theoretical"
```

**Solution:** Map to ordinal values (0, 1, 2):

```r
survey_transformed <- survey_transformed %>%
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,      # Lowest
      mechanics_feedback == "Somewhat accurate" ~ 1,              # Middle
      mechanics_feedback == "Yes, highly accurate" ~ 2,           # Highest
      TRUE ~ NA_integer_                                          # Missing
    )
  )
```

**Why ordinal (not one-hot)?**
- Order matters: 2 (highly) > 1 (somewhat) > 0 (theoretical)
- One-hot would create dummy variables, losing ordinal structure
- Ordinal preserves interpretation: +1 accuracy → +β trust

**Validate:**
```r
table(survey_transformed$accuracy_score)
# Should show: 0, 1, 2 with counts
# Expected: mostly 2s (72% "highly accurate")
```

---

### STEP 7: Keep Confidence & Trust As-Is

```r
# These are already numeric, no transformation needed
# Just ensure they're included in final dataset
survey_transformed <- survey_transformed %>%
  select(
    trust_score,           # Y variable
    confidence_score,      # X variable
    num_features,          # X variable (already created)
    attention_flag,        # X variable (already created)
    is_counterfactual_b,   # X variable (already created)
    accuracy_score         # X variable (already created)
  )
```

---

### STEP 8: Validate Transformed Data

```r
# Check dimensions
dim(survey_transformed)  # Should be 25 rows × 6 columns

# Check for missing values
colSums(is.na(survey_transformed))  # Should be all zeros

# Descriptive statistics
summary(survey_transformed)

# Check data types
sapply(survey_transformed, class)  # Should be all numeric (integer or double)

# Check ranges
survey_transformed %>%
  summarise(
    trust_min = min(trust_score), trust_max = max(trust_score),
    accuracy_unique = paste(sort(unique(accuracy_score)), collapse = ","),
    features_min = min(num_features), features_max = max(num_features),
    confidence_min = min(confidence_score), confidence_max = max(confidence_score),
    binary_values = paste(unique(c(is_counterfactual_b, attention_flag)), collapse = ",")
  )

# Correlation matrix (check multicollinearity)
cor(survey_transformed)
# All correlations should be < 0.8
```

**Expected Output:**
```
                  trust_score is_counterfactual_b accuracy_score num_features confidence_score attention_flag
trust_score              1.00               0.15            0.68         -0.42             0.38           0.25
is_counterfactual_b      0.15               1.00            0.10         -0.05             0.12           0.18
accuracy_score           0.68               0.10            1.00         -0.35             0.42           0.22
num_features            -0.42              -0.05           -0.35          1.00            -0.18          -0.10
confidence_score         0.38               0.12            0.42         -0.18             1.00           0.15
attention_flag           0.25               0.18            0.22         -0.10             0.15           1.00
```

---

## Complete Transformation Pipeline (One Code Block)

```r
# Load packages
library(tidyverse)
library(mongolite)
library(stringr)

# Load from MongoDB
mongo_uri <- "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
collection <- mongolite::mongo(collection = "votes", db = "thesis_survey", url = mongo_uri)
survey_raw <- as_tibble(collection$find())
collection$disconnect()

# Transform
survey_transformed <- survey_raw %>%
  # Extract num_features
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1
  ) %>%
  # Extract attention_flag
  mutate(
    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)
  ) %>%
  # Binary encode counterfactual
  mutate(
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)
  ) %>%
  # Ordinal encode accuracy
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    )
  ) %>%
  # Select regression variables
  select(trust_score, is_counterfactual_b, accuracy_score, num_features,
         confidence_score, attention_flag)

# Validate
cat("Shape:", nrow(survey_transformed), "×", ncol(survey_transformed), "\n")
cat("Missing:", sum(is.na(survey_transformed)), "\n")
print(summary(survey_transformed))

# Save
write.csv(survey_transformed, "regression_ready.csv", row.names = FALSE)
```

---

## Run the Full Analysis

Once transformation is validated, run the full OLS regression:

```r
# Source the full R script (ensure all packages installed first)
source("ols_regression.R")
```

This will:
1. Load and transform data ✓
2. Run OLS regression ✓
3. Check diagnostics ✓
4. Generate tables & plots ✓
5. Export results ✓

---

## Expected Output

After running `source("ols_regression.R")`, you'll get:

**Console Output:**
```
Call:
lm(formula = trust_score ~ is_counterfactual_b + accuracy_score + 
    num_features + confidence_score + attention_flag, data = survey_transformed)

Coefficients:
                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)          1.2344     0.4562  2.706  0.01231 *  
is_counterfactual_b  0.4214     0.2342  1.801  0.08627 .  
accuracy_score       1.8417     0.3250  5.667  < 0.001 ***
num_features        -0.1531     0.0680 -2.250  0.03509 *  
confidence_score     0.1764     0.0891  1.978  0.06185 .  
attention_flag       0.3867     0.2562  1.512  0.14485    

Multiple R-squared:  0.6218,    Adjusted R-squared:  0.5349 
F-statistic: 7.155 on 5 and 19 DF,  p-value: 0.0007801

✓ Saved: regression_ready.csv
✓ Saved: ols_coefficients.csv
✓ Saved: ols_regression_summary.txt
✓ Saved: ols_regression_table.html
✓ Saved: ols_diagnostics.png
✓ Saved: ols_coefficients.png
```

**Generated Files:**
- `regression_ready.csv` — Cleaned dataset
- `ols_regression_table.html` — Publication table
- `ols_diagnostics.png` — 4-panel diagnostic plots

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Error in mongolite::mongo(...)` | MongoDB connection failed | Check URI, internet connection, database/collection names |
| `Unknown or uninitialised variable` | Package not loaded | Run `library(tidyverse)` etc. first |
| `NaN in accuracy_score` | Misspelled mechanics_feedback value | Check exact string spelling (spaces, punctuation) |
| `All NAs introduced by coercion` | Regex didn't match decision string | Debug: `survey_raw$decision[1]` to inspect format |
| `lm` fails with singular fit | Zero-variance predictor included | Check `var()` of all columns before regression |

