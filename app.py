"""
Streamlit Demo: Causal vs Correlational Explanations for AI Investment Decisions
Master Thesis — Frankfurt School of Finance & Management

Two tabs:
  - Expert Survey (main): pick a stock, see SHAP vs LEWIS side by side, vote
  - Model Details: full technical overview with all plots and metrics

Usage: streamlit run app.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "survey.db"

LABEL_MAP = {0: "Sell", 1: "Hold", 2: "Buy"}
LABEL_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
LABEL_NAMES = ["Sell", "Hold", "Buy"]
FEATURE_COLS = ["volatility", "momentum", "volume_avg", "rsi_14",
                "max_drawdown", "vix", "eur_usd"]

MONGO_URI = None
try:
    MONGO_URI = st.secrets.get("MONGO_URI") if hasattr(st, "secrets") else None
except Exception:
    pass

FEATURE_DESCRIPTIONS = {
    "volatility": "How much the stock price swings (higher = riskier)",
    "momentum": "21-day price trend — is the stock going up or down?",
    "volume_avg": "How actively the stock is traded (liquidity)",
    "rsi_14": "Relative Strength Index — overbought (>70) or oversold (<30)?",
    "max_drawdown": "Worst peak-to-trough drop in the past 20 days (downside risk)",
    "vix": "Market fear index — expected volatility in the next 30 days",
    "eur_usd": "Euro to US Dollar exchange rate",
}

TICKER_NAMES = {
    # DAX 40
    "ADS.DE": "Adidas", "AIR.DE": "Airbus", "ALV.DE": "Allianz",
    "BAS.DE": "BASF", "BAYN.DE": "Bayer", "BEI.DE": "Beiersdorf",
    "BMW.DE": "BMW", "BNR.DE": "Brenntag", "CON.DE": "Continental",
    "1COV.DE": "Covestro", "DB1.DE": "Deutsche Boerse", "DBK.DE": "Deutsche Bank",
    "DHL.DE": "DHL Group", "DTE.DE": "Deutsche Telekom", "EOAN.DE": "E.ON",
    "FRE.DE": "Fresenius", "HEI.DE": "HeidelbergCement", "HEN3.DE": "Henkel",
    "IFX.DE": "Infineon", "LIN.DE": "Linde", "MBG.DE": "Mercedes-Benz",
    "MRK.DE": "Merck", "MTX.DE": "MTU Aero", "MUV2.DE": "Munich Re",
    "P911.DE": "Porsche", "PAH3.DE": "Porsche SE", "QGEN.DE": "Qiagen",
    "RWE.DE": "RWE", "SAP.DE": "SAP", "SHL.DE": "Siemens Healthineers",
    "SIE.DE": "Siemens", "SRT.DE": "Sartorius", "SY1.DE": "Symrise",
    "VNA.DE": "Vonovia", "VOW3.DE": "Volkswagen", "ZAL.DE": "Zalando",
    "RHM.DE": "Rheinmetall", "ENR.DE": "Siemens Energy",
    "DTG.DE": "Daimler Truck", "HNR1.DE": "Hannover Re",
    # S&P 500 (top 50)
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "GOOGL": "Alphabet", "META": "Meta", "BRK-B": "Berkshire Hathaway",
    "LLY": "Eli Lilly", "AVGO": "Broadcom", "JPM": "JPMorgan Chase",
    "TSLA": "Tesla", "UNH": "UnitedHealth", "XOM": "ExxonMobil",
    "V": "Visa", "PG": "Procter & Gamble", "MA": "Mastercard",
    "JNJ": "Johnson & Johnson", "COST": "Costco", "HD": "Home Depot",
    "ABBV": "AbbVie", "WMT": "Walmart", "NFLX": "Netflix",
    "CRM": "Salesforce", "BAC": "Bank of America", "CVX": "Chevron",
    "KO": "Coca-Cola", "AMD": "AMD", "PEP": "PepsiCo",
    "TMO": "Thermo Fisher", "ORCL": "Oracle", "ACN": "Accenture",
    "MCD": "McDonald's", "CSCO": "Cisco", "ADBE": "Adobe",
    "ABT": "Abbott Labs", "WFC": "Wells Fargo", "IBM": "IBM",
    "GE": "GE Aerospace", "PM": "Philip Morris", "NOW": "ServiceNow",
    "TXN": "Texas Instruments", "QCOM": "Qualcomm", "MS": "Morgan Stanley",
    "CAT": "Caterpillar", "INTU": "Intuit", "GS": "Goldman Sachs",
    "DHR": "Danaher", "AMGN": "Amgen",
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db():
    if MONGO_URI:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expert_name TEXT,
            expert_role TEXT,
            experience_years INTEGER,
            ticker TEXT,
            decision TEXT,
            preference TEXT NOT NULL,
            comment TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_vote(expert_name, expert_role, experience_years, ticker, decision,
              preference, comment):
    vote = {
        "expert_name": expert_name,
        "expert_role": expert_role,
        "experience_years": experience_years,
        "ticker": ticker,
        "decision": decision,
        "preference": preference,
        "comment": comment,
        "timestamp": datetime.now().isoformat(),
    }
    saved = False
    if MONGO_URI:
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            client["thesis_survey"]["votes"].insert_one(vote)
            saved = True
        except Exception:
            pass  # Fall through to SQLite
    if not saved:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO votes (expert_name, expert_role, experience_years,
                              ticker, decision, preference, comment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (expert_name, expert_role, experience_years, ticker, decision,
              preference, comment, vote["timestamp"]))
        conn.commit()
        conn.close()


def load_votes():
    if MONGO_URI:
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            docs = list(client["thesis_survey"]["votes"].find({}, {"_id": 0}))
            return pd.DataFrame(docs) if docs else pd.DataFrame()
        except Exception:
            pass  # Fall through to SQLite
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM votes", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_artifacts():
    """Load all saved artifacts from the pipeline."""
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    artifacts = {"meta": meta}

    # Test data (for per-ticker analysis)
    test_path = DATA_DIR / "processed" / "test.csv"
    if test_path.exists():
        artifacts["test_df"] = pd.read_csv(test_path, index_col=0, parse_dates=True)

    # Adjacency matrix
    adj_path = MODELS_DIR / "adj_matrix_directlingam.npy"
    if adj_path.exists():
        artifacts["adj"] = np.load(adj_path)

    # LEWIS scores
    for key, fname in [("lewis_causal", "lewis_scores_causal.csv"),
                        ("lewis_no_graph", "lewis_scores_no_graph.csv"),
                        ("shap_scores", "shap_scores.csv"),
                        ("comparison", "shap_vs_lewis_comparison.csv"),
                        ("classification_report", "classification_report.csv"),
                        ("classification_report_baseline", "classification_report_baseline.csv"),
                        ("model_comparison", "model_comparison.csv"),
                        ("hp_tuning", "hyperparameter_tuning_results.csv"),
                        ("reversal_scores", "reversal_scores.csv"),
                        ("label_dist", "label_distribution.csv")]:
        p = RESULTS_DIR / "tables" / fname
        if p.exists():
            artifacts[key] = pd.read_csv(p, index_col=0 if "classification_report" in key else None,
                                          parse_dates=False)

    # DAG stability
    stab_path = RESULTS_DIR / "tables" / "dag_stability_scores.csv"
    if stab_path.exists():
        artifacts["stability"] = pd.read_csv(stab_path, index_col=0)

    return artifacts


@st.cache_resource
def load_model():
    """Load the trained CatBoost model."""
    from catboost import CatBoostClassifier
    model_path = MODELS_DIR / "catboost_best.cbm"
    if not model_path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


@st.cache_data
def compute_ticker_shap(_model, test_df, ticker, feature_cols):
    """Compute SHAP values for a specific ticker's test data."""
    import shap
    ticker_data = test_df[test_df["ticker"] == ticker]
    if ticker_data.empty:
        return None

    X = ticker_data[feature_cols]
    y_true = ticker_data["label"].values
    y_pred = _model.predict(X).flatten().astype(int)
    y_prob = _model.predict_proba(X)

    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer.shap_values(X)

    # Normalize to (n_classes, n_samples, n_features)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        if shap_vals.shape[2] == 3 and shap_vals.shape[1] == len(feature_cols):
            shap_vals = shap_vals.transpose(2, 0, 1)
    elif isinstance(shap_vals, list):
        shap_vals = np.array(shap_vals)

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "shap_vals": shap_vals,
        "dates": ticker_data.index,
    }


# ---------------------------------------------------------------------------
# TAB 1: Expert Survey (main)
# ---------------------------------------------------------------------------

def render_survey_tab(artifacts, meta, feature_cols, model, test_df,
                      selected_ticker, expert_name, expert_role, experience_years):
    st.header("How does the AI explain its decision?")
    st.markdown(
        "Our AI model analyzes **7 market factors** (volatility, momentum, trading volume, "
        "RSI, drawdown risk, VIX fear index, and EUR/USD rate) to recommend whether "
        "to **Buy**, **Hold**, or **Sell** a stock. "
        "Select a stock from the sidebar to see the AI's recommendation and "
        "**two competing explanations** of the reasoning behind it."
    )

    ticker_info = None

    if selected_ticker and model and test_df is not None:
        ticker_info = compute_ticker_shap(model, test_df, selected_ticker, feature_cols)

        if ticker_info is None:
            st.warning(f"No test data available for {selected_ticker}")
        else:
            latest_idx = -1
            pred_label = int(ticker_info["y_pred"][latest_idx])
            true_label = int(ticker_info["y_true"][latest_idx])
            probs = ticker_info["y_prob"][latest_idx]

            st.subheader(f"{TICKER_NAMES.get(selected_ticker, selected_ticker)} ({selected_ticker})")

            c1, c2, c3 = st.columns(3)
            c1.metric("AI Recommendation", LABEL_NAMES[pred_label])
            c2.metric("Actual Outcome", LABEL_NAMES[true_label])
            c3.metric("Test Days", f"{len(ticker_info['y_pred'])}")

            st.markdown("---")

            # --- Side by side: Explanation 1 vs Explanation 2 ---
            st.subheader("Why did the AI decide this? Two competing explanations:")
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                st.markdown("### Explanation 1")
                st.markdown(
                    "Think of this like a **data analyst** looking at past patterns: "
                    "\"*Historically, when this factor was high, the AI tended to say Buy.*\" "
                    "It shows what **predicts** the decision, but a predictor isn't always "
                    "the real cause — it could be a coincidence."
                )
                shap_vals = ticker_info["shap_vals"]
                if shap_vals is not None and shap_vals.ndim == 3:
                    shap_mean = np.abs(shap_vals).mean(axis=(0, 1))
                    shap_df = pd.DataFrame({
                        "Feature": feature_cols, "Importance": shap_mean,
                    }).sort_values("Importance", ascending=True)

                    fig_s, ax_s = plt.subplots(figsize=(6, 4))
                    ax_s.barh(shap_df["Feature"], shap_df["Importance"], color="#3498db")
                    ax_s.set_xlabel("Feature Importance")
                    ax_s.set_title(f"Explanation 1 — {selected_ticker}")
                    plt.tight_layout()
                    st.pyplot(fig_s)
                    plt.close()

                    top_shap = shap_df.nlargest(3, "Importance")
                    st.markdown("**Top factors:**")
                    for _, row in top_shap.iterrows():
                        desc = FEATURE_DESCRIPTIONS.get(row["Feature"], "")
                        st.markdown(f"- **{row['Feature']}**: {desc}")

            with col_e2:
                st.markdown("### Explanation 2")
                st.markdown(
                    "Think of this like a **risk manager** asking: "
                    "\"*If we actually changed this factor, would the AI's recommendation "
                    "flip from Buy to Sell?*\" It identifies the **true drivers** of the "
                    "decision — the levers you could actually pull to change the outcome."
                )
                if "lewis_causal" in artifacts:
                    lewis_c = artifacts["lewis_causal"].copy()
                    lewis_c = lewis_c.sort_values("maxNesuf_avg", ascending=True)

                    fig_l, ax_l = plt.subplots(figsize=(6, 4))
                    ax_l.barh(lewis_c["feature"], lewis_c["maxNesuf_avg"], color="#e67e22")
                    ax_l.set_xlabel("Feature Importance")
                    ax_l.set_title(f"Explanation 2 — {selected_ticker}")
                    plt.tight_layout()
                    st.pyplot(fig_l)
                    plt.close()

                    top_lewis = lewis_c.nlargest(3, "maxNesuf_avg")
                    st.markdown("**Top factors:**")
                    for _, row in top_lewis.iterrows():
                        desc = FEATURE_DESCRIPTIONS.get(row["feature"], "")
                        st.markdown(f"- **{row['feature']}**: {desc}")
                else:
                    st.info("Explanation 2 scores not available. Run pipeline Step 7.")

            # Key disagreements
            if "comparison" in artifacts:
                comp = artifacts["comparison"]
                if "rank_diff" in comp.columns:
                    biggest = comp.nlargest(2, "rank_diff")
                    if not biggest.empty:
                        st.markdown("---")
                        st.markdown("### Where the two explanations disagree")
                        st.markdown(
                            "These are the factors where the two explanations "
                            "**tell a different story**. This matters because "
                            "a factor that *looks* important statistically may actually be "
                            "a coincidence — or vice versa."
                        )
                        for _, row in biggest.iterrows():
                            feat = row["feature"]
                            desc = FEATURE_DESCRIPTIONS.get(feat, "")
                            r_shap = int(row.get("rank_shap", 0))
                            r_lewis = int(row.get("rank_lewis", 0))
                            if r_shap < r_lewis:
                                note = ("Explanation 1 says this matters a lot, but Explanation 2 says "
                                        "it's not a real cause — likely a **spurious correlation**.")
                            else:
                                note = ("Explanation 2 says this is a real driver, but Explanation 1 "
                                        "undervalues it — a **hidden causal factor**.")
                            st.markdown(
                                f"- **{feat}** (#{r_shap} statistical vs #{r_lewis} causal) "
                                f"— {note}"
                            )

            # Causal graph
            st.markdown("---")
            st.subheader("How the AI sees cause and effect")
            st.markdown(
                "The diagram below shows the **cause-and-effect relationships** our algorithm "
                "discovered from the market data. An arrow from A to B means "
                "\"changes in A *cause* changes in B.\" This is what powers Explanation 2 "
                "above — it uses this map to trace which factors actually "
                "drive the AI's Buy/Hold/Sell decision."
            )
            cg_path = RESULTS_DIR / "plots" / "causal_graph_directlingam.png"
            if cg_path.exists():
                st.image(str(cg_path), width="stretch")
            else:
                st.info("Causal graph not available. Run pipeline Step 6.")

    st.markdown("---")

    # --- VOTE ---
    st.header("Which explanation do you trust more?")
    st.markdown(
        "As a professional in the financial industry, your judgement matters. "
        "After reviewing how both explanations describe the AI's recommendation above, "
        "please tell us which one you would **trust more when making real investment "
        "decisions** or presenting to clients and regulators."
    )

    with st.form("vote_form", clear_on_submit=True):
        st.markdown("""
        | | Explanation 2 | Explanation 1 |
        |---|---|---|
        | **In plain terms** | "This is *why* the AI decided — change this factor and the decision flips" | "This factor had the biggest *weight* in the prediction" |
        | **Best for** | Regulatory reporting, risk management, client communication | Quick overview, internal model monitoring |
        | **Watch out** | Requires discovering cause-and-effect structure first | May flag coincidences as important drivers |
        """)

        decision_label = "N/A"
        if ticker_info is not None:
            decision_label = LABEL_NAMES[int(ticker_info["y_pred"][-1])]

        preference = st.radio(
            "Which explanation do you find more trustworthy for investment decisions?",
            ["Explanation 2", "Explanation 1", "No preference"],
            horizontal=True,
        )

        comment = st.text_area(
            "Please share your reasoning (optional)",
            placeholder="e.g., 'I prefer Explanation 2 because it tells me what I can actually "
                       "change to influence the outcome.'",
            height=100,
        )

        submitted = st.form_submit_button("Submit Vote", type="primary",
                                          width="stretch")

        if submitted:
            pref_clean = "LEWIS" if preference == "Explanation 2" else (
                "SHAP" if preference == "Explanation 1" else "No")
            save_vote(
                expert_name=expert_name or "Anonymous",
                expert_role=expert_role,
                experience_years=experience_years,
                ticker=selected_ticker or "General",
                decision=decision_label,
                preference=pref_clean,
                comment=comment,
            )
            st.success("Thank you! Your vote has been recorded.")

    # Survey results
    votes_df = load_votes()
    if not votes_df.empty:
        st.subheader("Survey Results So Far")
        total = len(votes_df)
        vote_counts = votes_df["preference"].value_counts()
        e2_n = vote_counts.get("LEWIS", 0)
        e1_n = vote_counts.get("SHAP", 0)
        no_n = vote_counts.get("No", 0)

        r1, r2, r3 = st.columns(3)
        r1.metric("Explanation 2", f"{e2_n} ({100*e2_n/total:.0f}%)")
        r2.metric("Explanation 1", f"{e1_n} ({100*e1_n/total:.0f}%)")
        r3.metric("No Preference", f"{no_n} ({100*no_n/total:.0f}%)")

        with st.expander(f"Detailed Results ({total} responses)"):
            if "expert_role" in votes_df.columns:
                role_pref = pd.crosstab(votes_df["expert_role"], votes_df["preference"])
                st.dataframe(role_pref, width="stretch")
            csv = votes_df.to_csv(index=False)
            st.download_button("Download votes as CSV", csv, "survey_votes.csv", "text/csv")


# ---------------------------------------------------------------------------
# TAB 2: Model Details (full overview)
# ---------------------------------------------------------------------------

def render_details_tab(artifacts, meta, feature_cols, model, test_df, selected_ticker):

    # =================================================================
    # 1. Project Overview
    # =================================================================
    st.header("1. Project Overview")
    st.markdown("""
    This project compares two approaches to explaining AI investment decisions:

    | | Explanation 2 | Explanation 1 |
    |---|---|---|
    | **Approach** | Finds what *causes* the decision | Finds what *predicts* the decision |
    | **Question** | "If we changed this factor, would the decision flip?" | "How much does this factor contribute to the prediction?" |
    | **Regulatory fit** | Aligns with EU AI Act causal justification | Widely used but may not satisfy causal requirements |
    """)

    st.subheader("Dataset")
    st.markdown(f"""
    - **Universe**: DAX 40 + S&P 500 (top 50 by market cap)
    - **Frequency**: Daily OHLCV
    - **Train period**: Dec 2025 -- Feb 2026 ({meta.get('n_train', '?')} samples)
    - **Test period**: Mar 2026 ({meta.get('n_test', '?')} samples)
    - **Target**: 3-class Buy / Hold / Sell (from 5-day forward returns)
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CatBoost Accuracy", f"{meta.get('test_accuracy', 'N/A')}")
    col2.metric("Cohen's Kappa", f"{meta.get('cohen_kappa', 'N/A')}")
    col3.metric("Train Samples", meta.get("n_train", "N/A"))
    col4.metric("Test Samples", meta.get("n_test", "N/A"))

    # Feature table
    feat_data = []
    for feat in feature_cols:
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        category = "Macro" if feat in ("vix", "eur_usd") else "Stock"
        feat_data.append({"Feature": feat, "Type": category, "Description": desc})
    st.dataframe(pd.DataFrame(feat_data), width="stretch", hide_index=True)

    if "label_dist" in artifacts:
        st.subheader("Class Distribution")
        ld = artifacts["label_dist"]
        cols = st.columns(len(ld))
        for i, (_, row) in enumerate(ld.iterrows()):
            cols[i].metric(row["label"], f"{row['count']} ({row['percentage']}%)")

    st.markdown("---")

    # =================================================================
    # 2. Model Training & Performance
    # =================================================================
    st.header("2. Model Training & Performance")

    # Model comparison table
    if "model_comparison" in artifacts:
        st.subheader("Baseline vs Main Model")
        mc = artifacts["model_comparison"]
        st.dataframe(mc.round(4), width="stretch", hide_index=True)

    baseline_info = meta.get("baseline", {})
    bl_col, cb_col = st.columns(2)
    with bl_col:
        st.markdown("**Logistic Regression (Baseline)**")
        st.markdown(f"- Best C: `{baseline_info.get('best_C', 'N/A')}`")
        st.markdown(f"- CV Accuracy: `{baseline_info.get('cv_accuracy', 'N/A')}`")
        st.markdown(f"- Test Accuracy: `{baseline_info.get('test_accuracy', 'N/A')}`")
    with cb_col:
        st.markdown("**CatBoost (Main Model)**")
        bp = meta.get("best_params", {})
        st.markdown(f"- Iterations: `{bp.get('iterations', 'N/A')}`")
        st.markdown(f"- Learning rate: `{bp.get('learning_rate', 'N/A')}`")
        st.markdown(f"- Depth: `{bp.get('depth', 'N/A')}`")
        st.markdown(f"- CV Accuracy: `{meta.get('cv_accuracy', 'N/A')}`")
        st.markdown(f"- Test Accuracy: `{meta.get('test_accuracy', 'N/A')}`")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cm_l, cm_r = st.columns(2)
    with cm_l:
        cm_bl = RESULTS_DIR / "plots" / "confusion_matrix_baseline.png"
        if cm_bl.exists():
            st.image(str(cm_bl), caption="Logistic Regression (Baseline)",
                     width="stretch")
    with cm_r:
        cm_cb = RESULTS_DIR / "plots" / "confusion_matrix.png"
        if cm_cb.exists():
            st.image(str(cm_cb), caption="CatBoost (Main Model)",
                     width="stretch")
        else:
            st.info("Confusion matrix not found. Run pipeline Step 5.")

    # Classification reports
    cr_l, cr_r = st.columns(2)
    with cr_l:
        if "classification_report_baseline" in artifacts:
            st.subheader("Classification Report — Baseline")
            st.dataframe(artifacts["classification_report_baseline"].round(4),
                         width="stretch")
    with cr_r:
        if "classification_report" in artifacts:
            st.subheader("Classification Report — CatBoost")
            st.dataframe(artifacts["classification_report"].round(4),
                         width="stretch")

    roc_path = RESULTS_DIR / "plots" / "auc_roc.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curves (CatBoost)", width="stretch")

    # Hyperparameter tuning
    if "hp_tuning" in artifacts:
        with st.expander("CatBoost Hyperparameter Tuning Results (108 combinations)"):
            hp = artifacts["hp_tuning"].sort_values("mean_cv_accuracy", ascending=False)
            st.dataframe(hp.head(20).round(4), width="stretch", hide_index=True)

    # Per-ticker performance breakdown
    if model and test_df is not None and selected_ticker:
        with st.expander(f"Per-Ticker Performance: {TICKER_NAMES.get(selected_ticker, selected_ticker)}"):
            ticker_info = compute_ticker_shap(model, test_df, selected_ticker, feature_cols)
            if ticker_info is not None:
                from sklearn.metrics import accuracy_score, classification_report
                y_t, y_p = ticker_info["y_true"], ticker_info["y_pred"]
                tacc = accuracy_score(y_t, y_p)
                st.metric(f"Accuracy for {selected_ticker}", f"{tacc:.4f}")
                report = classification_report(y_t, y_p, target_names=LABEL_NAMES,
                                               output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    st.markdown("---")

    # =================================================================
    # 3. SHAP Feature Importance
    # =================================================================
    st.header("3. Explanation 1 — Feature Importance")

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        shap_bar = RESULTS_DIR / "plots" / "shap_bar_plot.png"
        if shap_bar.exists():
            st.image(str(shap_bar), width="stretch")
    with shap_col2:
        shap_sum = RESULTS_DIR / "plots" / "shap_summary_plot.png"
        if shap_sum.exists():
            st.image(str(shap_sum), width="stretch")

    if "shap_scores" in artifacts:
        st.dataframe(artifacts["shap_scores"].round(4), width="stretch",
                     hide_index=True)

    st.markdown("---")

    # =================================================================
    # 4. Causal Graph
    # =================================================================
    st.header("4. Discovered Causal Graph")
    st.markdown("""
    This graph shows **cause-and-effect relationships** discovered by DirectLiNGAM.
    An arrow from A to B means "A causes changes in B."
    Node colors: green = stock features, teal = macro features, red = target.
    """)

    cg_path = RESULTS_DIR / "plots" / "causal_graph_directlingam.png"
    if cg_path.exists():
        st.image(str(cg_path), width="stretch")

    if "adj" in artifacts:
        adj = artifacts["adj"]
        col_names = feature_cols + ["label"]
        edges = []
        for i in range(min(len(col_names), adj.shape[0])):
            for j in range(min(len(col_names), adj.shape[1])):
                if adj[i][j] == 1:
                    edges.append({"From": col_names[i], "To": col_names[j]})
        if edges:
            st.subheader("Causal Edges Found")
            st.dataframe(pd.DataFrame(edges), width="stretch", hide_index=True)

    stab_path = RESULTS_DIR / "plots" / "dag_stability_heatmap.png"
    if stab_path.exists():
        with st.expander("DAG Bootstrap Stability Heatmap"):
            st.image(str(stab_path), width="stretch")

    st.markdown("---")

    # =================================================================
    # 5. LEWIS Counterfactual Scores
    # =================================================================
    st.header("5. Explanation 2 — Counterfactual Scores")
    st.markdown("""
    These scores measure **causal feature importance** through counterfactual reasoning:
    - **Nesuf** (Necessity-Sufficiency): Overall causal importance
    - **Nec** (Necessity): "If this feature decreased, would the decision flip?"
    - **Suf** (Sufficiency): "If this feature increased, would the decision flip?"
    """)

    lewis_col1, lewis_col2 = st.columns(2)

    with lewis_col1:
        if "lewis_causal" in artifacts:
            st.subheader("With Graph")
            lewis_c = artifacts["lewis_causal"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(lewis_c["feature"], lewis_c["maxNesuf_avg"], color="#4ecdc4")
            ax.set_xlabel("maxNesuf (avg across pairwise comparisons)")
            ax.set_title("Explanation 2 — Feature Importance (With Causal Graph)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with lewis_col2:
        if "lewis_no_graph" in artifacts:
            st.subheader("Without Graph (Baseline)")
            lewis_ng = artifacts["lewis_no_graph"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(lewis_ng["feature"], lewis_ng["maxNesuf_avg"], color="#95e1d3")
            ax.set_xlabel("maxNesuf (avg across pairwise comparisons)")
            ax.set_title("Explanation 2 — Feature Importance (Without Causal Graph)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    rev_path = RESULTS_DIR / "plots" / "reversal_probabilities.png"
    if rev_path.exists():
        st.subheader("Reversal Probabilities (Nec vs Suf)")
        st.image(str(rev_path), width="stretch")

    st.markdown("---")

    # =================================================================
    # 6. SHAP vs LEWIS Comparison
    # =================================================================
    st.header("6. Explanation 1 vs Explanation 2: Where They Disagree")
    st.markdown("""
    A feature can be a strong predictor (high statistical importance) without being a cause
    (low causal importance), and vice versa. **Disagreements reveal spurious correlations.**
    """)

    comp_plot = RESULTS_DIR / "plots" / "shap_vs_lewis_comparison.png"
    if comp_plot.exists():
        st.image(str(comp_plot), width="stretch")

    if "comparison" in artifacts:
        comp_df = artifacts["comparison"]
        st.subheader("Ranking Table")
        display_cols = ["feature", "shap_importance", "lewis_causal", "rank_shap",
                        "rank_lewis", "rank_diff"]
        avail_cols = [c for c in display_cols if c in comp_df.columns]
        st.dataframe(
            comp_df[avail_cols].sort_values("rank_diff", ascending=False),
            width="stretch", hide_index=True,
        )

        from scipy import stats
        if "rank_shap" in comp_df.columns and "rank_lewis" in comp_df.columns:
            corr, p_val = stats.spearmanr(comp_df["rank_shap"], comp_df["rank_lewis"])
            st.metric("Spearman Rank Correlation (Explanation 1 vs Explanation 2)",
                      f"{corr:.4f} (p={p_val:.4f})")
            if abs(corr) < 0.3:
                st.success(
                    "Low correlation confirms that the two explanations "
                    "tell **fundamentally different stories**."
                )
            elif abs(corr) < 0.7:
                st.info("Moderate correlation — partial agreement with meaningful differences.")
            else:
                st.warning("High correlation — both explanations largely agree.")

    disagree_path = RESULTS_DIR / "reports" / "key_disagreements.txt"
    if disagree_path.exists():
        with st.expander("Key Disagreement Analysis"):
            st.text(disagree_path.read_text())

    st.markdown("---")

    # =================================================================
    # 7. Regulatory Compliance
    # =================================================================
    st.header("7. Regulatory Compliance Mapping")

    compliance_path = RESULTS_DIR / "reports" / "regulatory_compliance_report.md"
    if compliance_path.exists():
        st.markdown(compliance_path.read_text())
    else:
        st.info("Run pipeline Step 10 to generate the compliance report.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Causal XAI for Investment Decisions",
        layout="wide",
    )
    init_db()

    st.title("Causal Explainability for AI-Driven Investment Decisions")
    st.markdown("**Master Thesis** | Frankfurt School of Finance & Management | 2026")

    # Load data
    artifacts = load_artifacts()
    if artifacts is None:
        st.error("No pipeline artifacts found. Run `python src/pipeline.py --steps all` first.")
        return

    meta = artifacts["meta"]
    feature_cols = meta.get("feature_cols", FEATURE_COLS)
    model = load_model()
    test_df = artifacts.get("test_df")

    # Sidebar
    st.sidebar.header("Your Profile")
    expert_name = st.sidebar.text_input("Name (optional)")
    expert_role = st.sidebar.selectbox("Role", [
        "Portfolio Manager", "Risk Analyst", "Compliance Officer",
        "Data Scientist", "Academic / Researcher", "Student", "Other",
    ])
    experience_years = st.sidebar.slider("Years of experience", 0, 30, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("Select a Stock")
    available_tickers = sorted(test_df["ticker"].unique()) if test_df is not None else []
    ticker_options = [f"{TICKER_NAMES.get(t, t)} ({t})" for t in available_tickers]
    selected_idx = st.sidebar.selectbox(
        "Stock ticker",
        range(len(ticker_options)),
        format_func=lambda i: ticker_options[i],
        index=0,
    )
    selected_ticker = available_tickers[selected_idx] if available_tickers else None

    # Tabs
    tab_survey, tab_details = st.tabs(["Expert Survey", "Model Details"])

    with tab_survey:
        render_survey_tab(artifacts, meta, feature_cols, model, test_df,
                          selected_ticker, expert_name, expert_role, experience_years)

    with tab_details:
        render_details_tab(artifacts, meta, feature_cols, model, test_df, selected_ticker)


if __name__ == "__main__":
    main()
