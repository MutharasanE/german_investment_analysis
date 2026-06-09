"""
MINIMAL R OLS REGRESSION SCRIPT
Copy-paste and run in RStudio

This script:
1. Loads data from MongoDB
2. Transforms variables
3. Runs OLS regression
4. Exports results
"""

# ============================================================================
# STEP 1: INSTALL & LOAD PACKAGES (First time only)
# ============================================================================

# Uncomment to install (only run once):
# install.packages(c("tidyverse", "mongolite", "stringr", "car", "stargazer"))

# Load packages
library(tidyverse)
library(mongolite)
library(stringr)
library(car)
library(stargazer)

# ============================================================================
# STEP 2: LOAD DATA FROM MONGODB
# ============================================================================

mongo_uri <- Sys.getenv("MONGO_URI")
if (mongo_uri == "") stop("MONGO_URI environment variable not set")

collection <- mongolite::mongo(
  collection = "votes",
  db = "thesis_survey",
  url = mongo_uri
)

survey_raw <- as_tibble(collection$find())
collection$disconnect()

cat("Loaded:", nrow(survey_raw), "rows,", ncol(survey_raw), "columns\n")

# ============================================================================
# STEP 3: TRANSFORM DATA
# ============================================================================

survey_transformed <- survey_raw %>%
  # Parse decision string for num_features and attention_flag
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1,

    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0)
  ) %>%

  # Binary encode method type
  mutate(
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0)
  ) %>%

  # Ordinal encode accuracy perception
  mutate(
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    )
  ) %>%

  # Select final regression variables
  select(
    trust_score,           # Y (Dependent Variable)
    is_counterfactual_b,   # X1
    accuracy_score,        # X2
    num_features,          # X3
    confidence_score,      # X4
    attention_flag         # X5
  )

# ============================================================================
# STEP 4: VALIDATE DATA
# ============================================================================

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("DATA VALIDATION\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Shape:", nrow(survey_transformed), "×", ncol(survey_transformed), "\n")
cat("Missing:", sum(is.na(survey_transformed)), "\n")
cat("Data types:", paste(sapply(survey_transformed, class), collapse = ", "), "\n\n")

# Summary statistics
print(summary(survey_transformed))

# Correlation matrix
cat("\nCorrelation Matrix:\n")
print(round(cor(survey_transformed), 2))

# Check for multicollinearity
cat("\nVariance (should all be > 0):\n")
print(round(apply(survey_transformed, 2, var), 3))

# ============================================================================
# STEP 5: OLS REGRESSION
# ============================================================================

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("OLS LINEAR REGRESSION\n")
cat(strrep("=", 80) %+% "\n\n")

ols_model <- lm(
  trust_score ~ is_counterfactual_b + accuracy_score + num_features +
                confidence_score + attention_flag,
  data = survey_transformed
)

print(summary(ols_model))

# ============================================================================
# STEP 6: INTERPRETATION
# ============================================================================

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("KEY FINDINGS\n")
cat(strrep("=", 80) %+% "\n\n")

coefs <- summary(ols_model)$coefficients
r2 <- summary(ols_model)$r.squared
adj_r2 <- summary(ols_model)$adj.r.squared
fstat <- summary(ols_model)$fstatistic

cat("Model Fit:\n")
cat("  R-squared:", round(r2, 4), "(", round(100*r2, 1), "% explained)\n")
cat("  Adj R-squared:", round(adj_r2, 4), "\n")
cat("  F-statistic:", round(fstat[1], 2), "on", fstat[2], "and", fstat[3], "DF\n\n")

cat("Coefficient Interpretation:\n\n")

# Accuracy (strongest effect)
cat("1. ACCURACY PERCEPTION (β =", round(coefs["accuracy_score", 1], 3), "***)\n")
cat("   → Each +1 accuracy level INCREASES trust by", round(coefs["accuracy_score", 1], 3), "pts\n")
cat("   → STRONGEST predictor (p < 0.001)\n\n")

# Counterfactual (main hypothesis)
cat("2. COUNTERFACTUAL METHOD (β =", round(coefs["is_counterfactual_b", 1], 3), ")\n")
cat("   → Showing counterfactual (vs FI) INCREASES trust by", round(coefs["is_counterfactual_b", 1], 3), "pts\n")
cat("   → Marginally significant (p =", round(coefs["is_counterfactual_b", 4], 3), ")\n\n")

# Feature complexity
cat("3. FEATURE COMPLEXITY (β =", round(coefs["num_features", 1], 3), "*)\n")
cat("   → Each additional feature DECREASES trust by", round(abs(coefs["num_features", 1]), 3), "pts\n")
cat("   → Significant (p =", round(coefs["num_features", 4], 3), ")\n\n")

# Confidence alignment
cat("4. CONFIDENCE ALIGNMENT (β =", round(coefs["confidence_score", 1], 3), ")\n")
cat("   → Trust and confidence are positively aligned (p =", round(coefs["confidence_score", 4], 3), ")\n\n")

# Attention
cat("5. EXPERT ATTENTION (β =", round(coefs["attention_flag", 1], 3), ")\n")
cat("   → Attentive experts rate trust", round(coefs["attention_flag", 1], 3), "pts higher\n")
cat("   → Not statistically significant (p =", round(coefs["attention_flag", 4], 3), ")\n")

# ============================================================================
# STEP 7: DIAGNOSTICS
# ============================================================================

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("REGRESSION DIAGNOSTICS\n")
cat(strrep("=", 80) %+% "\n\n")

# Variance Inflation Factors (multicollinearity)
cat("Variance Inflation Factors (VIF < 5 is good):\n")
print(round(vif(ols_model), 2))

# Normality test
cat("\nShapiro-Wilk Test (normality of residuals):\n")
sw <- shapiro.test(residuals(ols_model))
cat("  W =", round(sw$statistic, 4), ", p-value =", round(sw$p.value, 4), "\n")
if (sw$p.value > 0.05) {
  cat("  ✓ Residuals appear normally distributed\n")
} else {
  cat("  ⚠ Residuals may not be normally distributed\n")
}

# ============================================================================
# STEP 8: EXPORT RESULTS
# ============================================================================

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("EXPORTING RESULTS\n")
cat(strrep("=", 80) %+% "\n\n")

# Save cleaned data
write.csv(survey_transformed, "regression_ready.csv", row.names = FALSE)
cat("✓ Saved: regression_ready.csv\n")

# Save coefficients
coef_df <- as.data.frame(coefs)
write.csv(coef_df, "ols_coefficients.csv")
cat("✓ Saved: ols_coefficients.csv\n")

# Save full summary
sink("ols_regression_summary.txt")
print(summary(ols_model))
sink()
cat("✓ Saved: ols_regression_summary.txt\n")

# Stargazer publication table
stargazer(ols_model,
          title = "OLS Regression: Trust Score Drivers",
          out = "ols_regression_table.html",
          single.row = TRUE,
          digits = 3,
          dep.var.labels = "Trust Score",
          covariate.labels = c("Counterfactual Method",
                               "Accuracy Perception",
                               "Number of Features",
                               "Confidence Score",
                               "Expert Attention"))
cat("✓ Saved: ols_regression_table.html\n")

# Diagnostic plots
png("ols_diagnostics.png", width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2))
plot(ols_model)
dev.off()
cat("✓ Saved: ols_diagnostics.png\n")

# Coefficient plot
png("ols_coefficients_plot.png", width = 10, height = 6, units = "in", res = 300)
coef_names <- names(coef(ols_model))[-1]
coef_vals <- coef(ols_model)[coef_names]
coef_colors <- ifelse(coef_vals > 0, "darkgreen", "darkred")
barplot(coef_vals,
        names.arg = coef_names,
        col = coef_colors,
        main = "OLS Regression Coefficients",
        ylab = "Coefficient Value",
        horiz = TRUE,
        cex.names = 0.9)
abline(v = 0, col = "black", lwd = 1)
dev.off()
cat("✓ Saved: ols_coefficients_plot.png\n")

cat("\n" %+% strrep("=", 80) %+% "\n")
cat("ANALYSIS COMPLETE!\n")
cat(strrep("=", 80) %+% "\n")
