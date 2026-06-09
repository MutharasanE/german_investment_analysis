# ============================================================================
# HIERARCHICAL OLS REGRESSION
# Model 1: Baseline (5 predictors)
# Model 2: + Expert controls (role + experience)
# Model 3: + Counterfactual x Accuracy interaction
# ============================================================================

library(tidyverse)
library(mongolite)
library(stringr)
library(car)
library(stargazer)

# ---- LOAD DATA ----
mongo_uri <- Sys.getenv("MONGO_URI")
if (mongo_uri == "") stop("MONGO_URI environment variable not set")
collection <- mongolite::mongo(collection = "votes", db = "thesis_survey", url = mongo_uri)
survey_raw <- as_tibble(collection$find())
collection$disconnect()
cat("Loaded:", nrow(survey_raw), "rows\n")

# ---- TRANSFORM ----
survey_transformed <- survey_raw %>%
  mutate(
    num_features = str_extract(decision, "features:([^|]+)") %>%
      str_remove("features:") %>%
      str_count(",") + 1,
    attention_flag = if_else(str_detect(decision, "attention:pass"), 1, 0),
    is_counterfactual_b = if_else(method_for_b == "COUNTERFACTUAL", 1, 0),
    accuracy_score = case_when(
      mechanics_feedback == "No, they seem theoretical" ~ 0,
      mechanics_feedback == "Somewhat accurate" ~ 1,
      mechanics_feedback == "Yes, highly accurate" ~ 2,
      TRUE ~ NA_integer_
    ),
    experience_years = as.numeric(experience_years),
    role_pm   = if_else(expert_role == "Portfolio Manager", 1, 0),
    role_ra   = if_else(expert_role == "Risk Analyst", 1, 0),
    role_acad = if_else(expert_role == "Academic / Researcher", 1, 0),
    role_comp = if_else(expert_role == "Compliance Officer", 1, 0)
  ) %>%
  select(trust_score, is_counterfactual_b, accuracy_score, num_features,
         confidence_score, attention_flag,
         experience_years, role_pm, role_ra, role_acad, role_comp) %>%
  drop_na()

cat("Effective N:", nrow(survey_transformed), "\n\n")

# ============================================================================
# MODEL 1: BASELINE
# ============================================================================
m1 <- lm(trust_score ~ is_counterfactual_b + accuracy_score + num_features +
                       confidence_score + attention_flag,
         data = survey_transformed)

# ============================================================================
# MODEL 2: + EXPERT CONTROLS
# ============================================================================
m2 <- lm(trust_score ~ is_counterfactual_b + accuracy_score + num_features +
                       confidence_score + attention_flag +
                       experience_years + role_pm + role_ra + role_acad + role_comp,
         data = survey_transformed)

# ============================================================================
# MODEL 3: + INTERACTION
# ============================================================================
m3 <- lm(trust_score ~ is_counterfactual_b * accuracy_score +
                       num_features + confidence_score + attention_flag +
                       experience_years + role_pm + role_ra + role_acad + role_comp,
         data = survey_transformed)

# ---- SUMMARIES ----
cat("\n========== MODEL 1 (Baseline) ==========\n")
print(summary(m1))
cat("AIC:", AIC(m1), " BIC:", BIC(m1), "\n")

cat("\n========== MODEL 2 (+ Expert Controls) ==========\n")
print(summary(m2))
cat("AIC:", AIC(m2), " BIC:", BIC(m2), "\n")

cat("\n========== MODEL 3 (+ Interaction) ==========\n")
print(summary(m3))
cat("AIC:", AIC(m3), " BIC:", BIC(m3), "\n")

# ---- NESTED MODEL COMPARISON ----
cat("\n========== NESTED F-TESTS ==========\n")
cat("M1 vs M2 (do expert controls add value?):\n")
print(anova(m1, m2))
cat("\nM2 vs M3 (does interaction add value?):\n")
print(anova(m2, m3))

# ---- SIDE-BY-SIDE TABLE ----
stargazer(m1, m2, m3,
          type = "text",
          title = "Hierarchical OLS: Trust Score Drivers",
          column.labels = c("Baseline", "+ Controls", "+ Interaction"),
          dep.var.labels = "Trust Score",
          out = "hierarchical_regression.html",
          single.row = TRUE,
          digits = 3)

# Export coefficients
write.csv(as.data.frame(summary(m1)$coefficients), "m1_coefficients.csv")
write.csv(as.data.frame(summary(m2)$coefficients), "m2_coefficients.csv")
write.csv(as.data.frame(summary(m3)$coefficients), "m3_coefficients.csv")

cat("\nDone. Saved: hierarchical_regression.html, m1/m2/m3_coefficients.csv\n")
