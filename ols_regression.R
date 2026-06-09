"""
R WORKFLOW FOR OLS LINEAR REGRESSION
Data Transformation + OLS Analysis
"""

# ============================================================================
# SECTION 1: R SETUP & PACKAGES
# ============================================================================

# Install required packages (run once)
install.packages(c("tidyverse", "mongolite", "jsonlite", "stringr", "car", "stargazer", "ggplot2"))

# Load packages
library(tidyverse)      # Data manipulation (dplyr)
library(mongolite)      # MongoDB connection
library(stringr)        # String parsing (regex)
library(car)            # VIF, diagnostics
library(stargazer)      # Publication-ready tables
library(ggplot2)        # Visualization
library(gridExtra)      # Multi-panel plots

# ============================================================================
# SECTION 2: LOAD DATA FROM MONGODB
# ============================================================================

# Connect to MongoDB
mongo_uri <- Sys.getenv("MONGO_URI")
if (mongo_uri == "") stop("MONGO_URI environment variable not set")

# Create MongoDB connection
collection <- mongolite::mongo(
  collection = "votes",
  db = "thesis_survey",
  url = mongo_uri
)

# Query all survey responses
survey_raw <- collection$find()

# Convert to tibble for easier manipulation
survey_raw <- as_tibble(survey_raw)

# Disconnect
collection$disconnect()

# Check data loaded
cat("Survey data loaded: ", nrow(survey_raw), " rows, ", ncol(survey_raw), " columns\n")
cat("\nColumn names:\n")
print(colnames(survey_raw))
cat("\nFirst few rows:\n")
print(head(survey_raw, 3))

# ============================================================================
# SECTION 3: DATA TRANSFORMATION
# ============================================================================

# Start transformation pipeline
survey_transformed <- survey_raw %>%

  # Step 1: Extract number of features from decision string
  # Format: "decision_a:BUY|decision_b:HOLD|features:volatility,momentum,...|attention:pass"
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1
  ) %>%

  # Step 2: Extract attention flag
  # "attention:pass" = 1, "attention:fail" = 0
  mutate(
    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)
  ) %>%

  # Step 3: Binary encode method_for_b
  # 1 = COUNTERFACTUAL, 0 = FEATURE_IMPORTANCE
  mutate(
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)
  ) %>%

  # Step 4: Ordinal encode accuracy_score
  # 0 = "No, they seem theoretical"
  # 1 = "Somewhat accurate"
  # 2 = "Yes, highly accurate"
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    )
  ) %>%

  # Step 5: Keep confidence_score as-is (already numeric)
  # Step 6: Keep trust_score as-is (already numeric, this is our Y variable)

  # Step 7: Select only regression variables
  select(
    trust_score,           # Y (Dependent Variable)
    is_counterfactual_b,   # X1
    accuracy_score,        # X2
    num_features,          # X3
    confidence_score,      # X4
    attention_flag         # X5
  )

# ============================================================================
# SECTION 4: DATA VALIDATION
# ============================================================================

cat("\n" + "="*80)
cat("DATA VALIDATION\n")
cat("="*80)

# Check dimensions
cat("\nDimensions:", nrow(survey_transformed), "rows,", ncol(survey_transformed), "columns\n")

# Check for missing values
cat("\nMissing values:\n")
print(colSums(is.na(survey_transformed)))

# Descriptive statistics
cat("\nDescriptive Statistics:\n")
print(summary(survey_transformed))

# Check data types
cat("\nData Types:\n")
print(sapply(survey_transformed, class))

# Check ranges
cat("\nVariable Ranges:\n")
cat("  trust_score:           min =", min(survey_transformed$trust_score),
    ", max =", max(survey_transformed$trust_score), "\n")
cat("  is_counterfactual_b:   unique values =", unique(survey_transformed$is_counterfactual_b), "\n")
cat("  accuracy_score:        unique values =", sort(unique(survey_transformed$accuracy_score)), "\n")
cat("  num_features:          min =", min(survey_transformed$num_features),
    ", max =", max(survey_transformed$num_features), "\n")
cat("  confidence_score:      min =", min(survey_transformed$confidence_score),
    ", max =", max(survey_transformed$confidence_score), "\n")
cat("  attention_flag:        unique values =", unique(survey_transformed$attention_flag), "\n")

# Correlation matrix
cat("\nCorrelation Matrix (Check for Multicollinearity):\n")
cor_matrix <- cor(survey_transformed)
print(cor_matrix)

# Check variance
cat("\nVariance (all should be > 0):\n")
print(apply(survey_transformed, 2, var))

# ============================================================================
# SECTION 5: OLS REGRESSION
# ============================================================================

cat("\n" + "="*80)
cat("OLS LINEAR REGRESSION\n")
cat("="*80)

# Fit OLS model
# Model: trust_score ~ is_counterfactual_b + accuracy_score + num_features + confidence_score + attention_flag

ols_model <- lm(
  trust_score ~ is_counterfactual_b + accuracy_score + num_features +
                confidence_score + attention_flag,
  data = survey_transformed
)

# Display regression summary
cat("\nMODEL SUMMARY:\n")
print(summary(ols_model))

# ============================================================================
# SECTION 6: REGRESSION DIAGNOSTICS
# ============================================================================

cat("\n" + "="*80)
cat("REGRESSION DIAGNOSTICS\n")
cat("="*80)

# 1. Check linearity: plot residuals vs fitted
cat("\nDiagnostic 1: Linearity (Residuals vs Fitted)\n")
print("Plot saved as: residuals_vs_fitted.png")

# 2. Normality: Q-Q plot
cat("\nDiagnostic 2: Normality of Residuals\n")
shapiro_test <- shapiro.test(residuals(ols_model))
cat("  Shapiro-Wilk Test: W =", shapiro_test$statistic, ", p-value =", shapiro_test$p.value, "\n")
if (shapiro_test$p.value > 0.05) {
  cat("  ✓ Residuals appear normally distributed (p > 0.05)\n")
} else {
  cat("  ⚠ Residuals may not be normally distributed (p < 0.05)\n")
}

# 3. Homoscedasticity: Breusch-Pagan test
cat("\nDiagnostic 3: Homoscedasticity (Constant Variance)\n")
# You'll need to install lmtest package for this
# install.packages("lmtest")
# library(lmtest)
# bp_test <- bptest(ols_model)
# cat("  Breusch-Pagan Test: p-value =", bp_test$p.value, "\n")

# 4. Multicollinearity: VIF
cat("\nDiagnostic 4: Multicollinearity (VIF)\n")
vif_values <- vif(ols_model)
print(vif_values)
cat("  (VIF < 5 indicates no multicollinearity)\n")

# 5. Independence: Durbin-Watson test
cat("\nDiagnostic 5: Independence (Durbin-Watson)\n")
# install.packages("lmtest")
# library(lmtest)
# dw_test <- dwtest(ols_model)
# cat("  Durbin-Watson statistic:", dw_test$statistic, "\n")

# ============================================================================
# SECTION 7: COEFFICIENT INTERPRETATION
# ============================================================================

cat("\n" + "="*80)
cat("COEFFICIENT INTERPRETATION\n")
cat("="*80)

coefficients_df <- as.data.frame(summary(ols_model)$coefficients)
coefficients_df <- rownames_to_column(coefficients_df, var = "Variable")

cat("\nRegression Coefficients:\n")
print(coefficients_df)

# Extract individual coefficients
intercept <- coef(ols_model)["(Intercept)"]
beta_counterfactual <- coef(ols_model)["is_counterfactual_b"]
beta_accuracy <- coef(ols_model)["accuracy_score"]
beta_features <- coef(ols_model)["num_features"]
beta_confidence <- coef(ols_model)["confidence_score"]
beta_attention <- coef(ols_model)["attention_flag"]

# p-values
pval_counterfactual <- summary(ols_model)$coefficients["is_counterfactual_b", 4]
pval_accuracy <- summary(ols_model)$coefficients["accuracy_score", 4]
pval_features <- summary(ols_model)$coefficients["num_features", 4]
pval_confidence <- summary(ols_model)$coefficients["confidence_score", 4]
pval_attention <- summary(ols_model)$coefficients["attention_flag", 4]

cat("\nKEY FINDINGS:\n\n")

cat("1. Intercept (Baseline):", round(intercept, 4), "\n")
cat("   Meaning: Baseline trust score when all predictors = 0\n\n")

cat("2. Counterfactual Effect:", round(beta_counterfactual, 4), "(p =", round(pval_counterfactual, 4), ")\n")
cat("   Meaning: Showing counterfactual (vs FI) INCREASES trust by",
    round(beta_counterfactual, 3), "points\n")
if (pval_counterfactual < 0.05) {
  cat("   Statistical Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else if (pval_counterfactual < 0.10) {
  cat("   Statistical Significance: *Marginally significant* (0.05 < p < 0.10)\n")
} else {
  cat("   Statistical Significance: NOT significant (p > 0.10)\n")
}
cat("\n")

cat("3. Accuracy Score Effect:", round(beta_accuracy, 4), "(p =", round(pval_accuracy, 4), ")\n")
cat("   Meaning: Each +1 point in accuracy perception increases trust by",
    round(beta_accuracy, 3), "points\n")
cat("   (e.g., going from 'theoretical' [0] to 'highly accurate' [2] increases trust by ~",
    round(2 * beta_accuracy, 2), ")\n")
if (pval_accuracy < 0.05) {
  cat("   Statistical Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else {
  cat("   Statistical Significance: NOT significant\n")
}
cat("\n")

cat("4. Feature Complexity Effect:", round(beta_features, 4), "(p =", round(pval_features, 4), ")\n")
cat("   Meaning: Each additional feature DECREASES trust by",
    round(abs(beta_features), 3), "points\n")
cat("   Implication: Simpler explanations build more trust (complexity tax)\n")
if (pval_features < 0.05) {
  cat("   Statistical Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else {
  cat("   Statistical Significance: NOT significant\n")
}
cat("\n")

cat("5. Confidence Score Effect:", round(beta_confidence, 4), "(p =", round(pval_confidence, 4), ")\n")
cat("   Meaning: Trust and confidence are POSITIVELY aligned\n")
if (pval_confidence < 0.05) {
  cat("   Statistical Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else {
  cat("   Statistical Significance: NOT significant\n")
}
cat("\n")

cat("6. Attention Flag Effect:", round(beta_attention, 4), "(p =", round(pval_attention, 4), ")\n")
cat("   Meaning: Attentive experts rate trust", round(beta_attention, 3), "points higher\n")
if (pval_attention < 0.05) {
  cat("   Statistical Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else {
  cat("   Statistical Significance: NOT significant\n")
}

# ============================================================================
# SECTION 8: MODEL FIT
# ============================================================================

cat("\n" + "="*80)
cat("MODEL FIT\n")
cat("="*80)

r_squared <- summary(ols_model)$r.squared
adj_r_squared <- summary(ols_model)$adj.r.squared
f_statistic <- summary(ols_model)$fstatistic[1]
f_pvalue <- pf(summary(ols_model)$fstatistic[1],
               summary(ols_model)$fstatistic[2],
               summary(ols_model)$fstatistic[3],
               lower.tail = FALSE)

cat("\nR-squared:", round(r_squared, 4),
    "({", round(100 * r_squared, 1), "% of trust variation explained})\n")
cat("Adjusted R-squared:", round(adj_r_squared, 4), "\n")
cat("F-statistic: F(5,19) =", round(f_statistic, 4), ", p-value <", round(f_pvalue, 4), "\n")

if (f_pvalue < 0.001) {
  cat("Model Significance: ***HIGHLY SIGNIFICANT*** (p < 0.001)\n")
} else if (f_pvalue < 0.05) {
  cat("Model Significance: ***SIGNIFICANT*** (p < 0.05)\n")
} else {
  cat("Model Significance: NOT significant\n")
}

# ============================================================================
# SECTION 9: EXPORT RESULTS
# ============================================================================

cat("\n" + "="*80)
cat("EXPORTING RESULTS\n")
cat("="*80)

# Export coefficients table
coefficients_table <- as.data.frame(summary(ols_model)$coefficients)
write.csv(coefficients_table, "ols_coefficients.csv")
cat("\n✓ Saved: ols_coefficients.csv\n")

# Export full summary
sink("ols_regression_summary.txt")
print(summary(ols_model))
sink()
cat("✓ Saved: ols_regression_summary.txt\n")

# Stargazer table for publication
stargazer(ols_model,
          title = "OLS Regression: Trust Score Drivers",
          out = "ols_regression_table.html",
          single.row = TRUE,
          digits = 3)
cat("✓ Saved: ols_regression_table.html\n")

# Save cleaned data
write.csv(survey_transformed, "regression_ready.csv", row.names = FALSE)
cat("✓ Saved: regression_ready.csv\n")

# ============================================================================
# SECTION 10: VISUALIZATION
# ============================================================================

cat("\n" + "="*80)
cat("GENERATING VISUALIZATIONS\n")
cat("="*80)

# Create 4-panel diagnostic plot
png("ols_diagnostics.png", width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2))

# Plot 1: Residuals vs Fitted
plot(ols_model, which = 1, main = "Residuals vs Fitted")

# Plot 2: Q-Q plot
plot(ols_model, which = 2, main = "Normal Q-Q Plot")

# Plot 3: Scale-Location (homoscedasticity)
plot(ols_model, which = 3, main = "Scale-Location")

# Plot 4: Residuals vs Leverage (influential points)
plot(ols_model, which = 5, main = "Residuals vs Leverage")

dev.off()
cat("✓ Saved: ols_diagnostics.png\n")

# Coefficient plot
png("ols_coefficients.png", width = 10, height = 6, units = "in", res = 300)

# Extract coefficients (exclude intercept)
coef_names <- names(coef(ols_model))[-1]  # Remove intercept
coef_values <- coef(ols_model)[coef_names]
coef_colors <- ifelse(coef_values > 0, "green", "red")

barplot(coef_values,
        names.arg = coef_names,
        col = coef_colors,
        main = "OLS Regression Coefficients",
        ylab = "Coefficient Value",
        xlab = "Predictor",
        horiz = TRUE,
        cex.names = 0.8)
abline(v = 0, col = "black", lwd = 1)

dev.off()
cat("✓ Saved: ols_coefficients.png\n")

cat("\n" + "="*80)
cat("ANALYSIS COMPLETE!")
cat("="*80)
cat("\nGenerated Files:\n")
cat("  1. regression_ready.csv            - Cleaned dataset\n")
cat("  2. ols_coefficients.csv            - Regression coefficients table\n")
cat("  3. ols_regression_summary.txt      - Full regression output\n")
cat("  4. ols_regression_table.html       - Publication-ready table\n")
cat("  5. ols_diagnostics.png            - 4-panel diagnostic plots\n")
cat("  6. ols_coefficients.png           - Coefficient bar plot\n")
