"""
Streamlit Demo: Causal vs Correlational Explanations for AI Investment Decisions
Master Thesis — Frankfurt School of Finance & Management

Usage: streamlit run app.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/investment")
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "survey.db"

# MongoDB Atlas (set in Streamlit secrets or .streamlit/secrets.toml)
# If not configured, falls back to local SQLite
MONGO_URI = None
try:
    MONGO_URI = st.secrets.get("MONGO_URI")
except Exception:
    pass

# Feature descriptions for management users
FEATURE_DESCRIPTIONS = {
    "volatility": "How much the stock price swings up and down (higher = riskier)",
    "momentum": "Recent price trend — is the stock going up or down?",
    "volume_avg": "How actively the stock is traded (liquidity indicator)",
    "return_1y": "Total return over the past year (%)",
    "max_drawdown": "Worst peak-to-trough drop in the past year (downside risk)",
    "ecb_rate": "ECB main refinancing rate — the cost of borrowing in the eurozone",
    "eur_usd": "Euro to US Dollar exchange rate",
    "de_inflation": "German inflation rate (year-over-year %)",
    "vix": "Market fear index — expected volatility in the next 30 days",
}

# DAX ticker to company name mapping
TICKER_NAMES = {
    "ADS.DE": "Adidas", "AIR.DE": "Airbus", "ALV.DE": "Allianz",
    "BAS.DE": "BASF", "BEI.DE": "Beiersdorf", "BMW.DE": "BMW",
    "DB1.DE": "Deutsche Boerse", "DTE.DE": "Deutsche Telekom",
    "EOAN.DE": "E.ON", "FRE.DE": "Fresenius", "HEN3.DE": "Henkel",
    "IFX.DE": "Infineon", "MBG.DE": "Mercedes-Benz", "MTX.DE": "MTU Aero",
    "MUV2.DE": "Munich Re", "RWE.DE": "RWE", "SAP.DE": "SAP",
    "SIE.DE": "Siemens", "SY1.DE": "Symrise", "VOW3.DE": "Volkswagen",
}


# ---------------------------------------------------------------------------
# Database helpers (MongoDB Atlas or local SQLite fallback)
# ---------------------------------------------------------------------------

def _get_mongo_collection():
    """Get MongoDB votes collection."""
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    db = client["thesis_survey"]
    return db["votes"]


def init_db():
    """Create survey tables if they don't exist."""
    if MONGO_URI:
        return  # MongoDB creates collections automatically
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
    """Save a single vote."""
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
    if MONGO_URI:
        _get_mongo_collection().insert_one(vote)
    else:
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
    """Load all votes as a DataFrame."""
    if MONGO_URI:
        collection = _get_mongo_collection()
        docs = list(collection.find({}, {"_id": 0}))
        return pd.DataFrame(docs) if docs else pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM votes", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_artifacts():
    """Load all saved artifacts from run_investment.py and run_shap_comparison.py."""
    meta_files = sorted(MODELS_DIR.glob("metadata_*.json"))
    if not meta_files:
        return None

    meta_file = meta_files[0]
    key = meta_file.stem.replace("metadata_", "")

    with open(meta_file) as f:
        meta = json.load(f)

    df = pd.read_csv(DATA_DIR / "investment_dataset.csv")
    adj = np.load(MODELS_DIR / f"adj_matrix_{key}.npy")
    lewis_df = pd.read_csv(RESULTS_DIR / f"lewis_scores_{key}.csv")

    shap_path = RESULTS_DIR / f"shap_scores_{key}.csv"
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None

    shap_vals_path = MODELS_DIR / f"shap_values_{key}.npy"
    shap_values = np.load(shap_vals_path) if shap_vals_path.exists() else None

    comparison_path = RESULTS_DIR / f"lewis_vs_shap_{key}.csv"
    comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else None

    return {
        "key": key, "meta": meta, "df": df, "adj": adj,
        "lewis_df": lewis_df, "shap_df": shap_df,
        "shap_values": shap_values, "comparison_df": comparison_df,
    }


def ticker_display(ticker):
    """Convert ticker to readable name."""
    name = TICKER_NAMES.get(ticker, ticker)
    return f"{name} ({ticker})"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_causal_graph(adj, feature_cols):
    """Plot causal DAG."""
    import networkx as nx

    col_names = feature_cols + ["investment_decision"]
    G = nx.DiGraph()
    G.add_nodes_from(col_names)
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            if adj[i][j] == 1:
                G.add_edge(col_names[i], col_names[j])

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
            node_size=2000, font_size=9, font_weight="bold",
            arrows=True, arrowsize=20, edge_color="gray",
            connectionstyle="arc3,rad=0.1")
    ax.set_title("Discovered Causal Graph", fontsize=14)
    return fig


def plot_comparison_bar(lewis_vals, shap_vals, features):
    """Side-by-side horizontal bar chart."""
    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(x + width / 2, lewis_vals, width, label="LEWIS (causal)",
            color="forestgreen")
    ax.barh(x - width / 2, shap_vals, width, label="SHAP (correlational)",
            color="steelblue")
    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_title("Feature Importance: LEWIS vs SHAP", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)
    return fig


def plot_single_shap(shap_vals_row, features, title):
    """Bar chart for a single sample's SHAP values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["forestgreen" if v > 0 else "salmon" for v in shap_vals_row]
    ax.barh(features, shap_vals_row, color=colors)
    ax.set_xlabel("Impact on APPROVE probability", fontsize=11)
    ax.set_title(f"SHAP Explanation — {title}", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.5)
    return fig


def plot_single_lewis(lewis_df, features, title):
    """Bar chart for LEWIS Nesuf scores."""
    scores = lewis_df.set_index("feature").reindex(features)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(features, scores["maxNesuf"].values, color="forestgreen")
    ax.set_xlabel("Causal Importance (Nesuf)", fontsize=11)
    ax.set_title(f"LEWIS Explanation — {title}", fontsize=13)
    ax.set_xlim(0, 1.05)
    return fig


# ---------------------------------------------------------------------------
# Text interpretation helpers
# ---------------------------------------------------------------------------

def interpret_lewis_top_features(lewis_df, n=3):
    """Generate plain English interpretation of top LEWIS features."""
    top = lewis_df.nlargest(n, "maxNesuf")
    lines = []
    for _, row in top.iterrows():
        feat = row["feature"]
        score = row["maxNesuf"]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        if score > 0.5:
            strength = "the strongest causal driver"
        elif score > 0.2:
            strength = "a significant causal factor"
        else:
            strength = "a moderate causal factor"
        lines.append(f"- **{feat}** ({desc}) is {strength} "
                     f"with a score of {score:.3f}")
    return "\n".join(lines)


def interpret_shap_top_features(shap_df, n=3):
    """Generate plain English interpretation of top SHAP features."""
    top = shap_df.nlargest(n, "shap_normalized")
    lines = []
    for _, row in top.iterrows():
        feat = row["feature"]
        score = row["shap_normalized"]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        if score > 0.5:
            strength = "the strongest predictor"
        elif score > 0.2:
            strength = "a significant predictor"
        else:
            strength = "a moderate predictor"
        lines.append(f"- **{feat}** ({desc}) is {strength} "
                     f"with a score of {score:.3f}")
    return "\n".join(lines)


def interpret_company_decision(row, feature_cols):
    """Generate plain English explanation for a single company's decision."""
    decision = "APPROVE" if row["investment_decision"] == 1 else "REJECT"
    lines = [f"The AI recommends to **{decision}** this investment.\n",
             "Key data points for this decision:"]
    for feat in feature_cols:
        if feat in row.index:
            val = row[feat]
            desc = FEATURE_DESCRIPTIONS.get(feat, "")
            lines.append(f"- **{feat}**: {val:.4f} — {desc}")
    return "\n".join(lines)


def interpret_ranking_difference(comparison_df):
    """Explain the biggest ranking disagreements in plain English."""
    comp = comparison_df.copy()
    comp["rank_diff"] = abs(comp["lewis_rank"] - comp["shap_rank"])
    biggest = comp.nlargest(3, "rank_diff")

    lines = ["**Where LEWIS and SHAP disagree the most:**\n"]
    for _, row in biggest.iterrows():
        feat = row["feature"]
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        l_rank = int(row["lewis_rank"])
        s_rank = int(row["shap_rank"])
        if l_rank < s_rank:
            lines.append(
                f"- **{feat}** — LEWIS ranks it #{l_rank} (important cause), "
                f"but SHAP ranks it #{s_rank}. This means {feat.lower()} "
                f"*causes* the decision to change, even though it's not a strong "
                f"statistical predictor."
            )
        else:
            lines.append(
                f"- **{feat}** — SHAP ranks it #{s_rank} (strong predictor), "
                f"but LEWIS ranks it #{l_rank}. This means {feat.lower()} "
                f"is correlated with the outcome but doesn't actually *cause* it. "
                f"Changing this feature alone wouldn't flip the decision."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Causal XAI for Investment Decisions",
        page_icon="📊",
        layout="wide",
    )

    init_db()

    st.title("Causal Explainability for AI-Driven Investment Decisions")
    st.markdown(
        "**Master Thesis** | Frankfurt School of Finance & Management | 2026"
    )

    # Load artifacts
    artifacts = load_artifacts()
    if artifacts is None:
        st.error(
            "No saved artifacts found. Run `python run_investment.py` and "
            "`python run_shap_comparison.py` first."
        )
        return

    meta = artifacts["meta"]
    df = artifacts["df"]
    feature_cols = meta["feature_cols"]
    comparison_df = artifacts["comparison_df"]

    # -----------------------------------------------------------------------
    # Sidebar: Expert profile + Company selector
    # -----------------------------------------------------------------------
    st.sidebar.header("About You")
    expert_name = st.sidebar.text_input("Your name (optional)")
    expert_role = st.sidebar.selectbox("Your role", [
        "Portfolio Manager",
        "Risk Analyst",
        "Compliance Officer",
        "Data Scientist",
        "Academic / Researcher",
        "Student",
        "Other",
    ])
    experience_years = st.sidebar.slider("Years of experience", 0, 30, 5)

    st.sidebar.divider()
    st.sidebar.header("Select a Company")
    if "ticker" in df.columns:
        tickers = sorted(df["ticker"].unique())
        ticker_options = {ticker_display(t): t for t in tickers}
        selected_display = st.sidebar.selectbox(
            "DAX Company", list(ticker_options.keys())
        )
        selected_ticker = ticker_options[selected_display]
    else:
        selected_ticker = None

    # -----------------------------------------------------------------------
    # Tab layout
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "About", "Company View", "Causal Graph",
        "LEWIS vs SHAP", "Vote", "Survey Results",
    ])

    # --- Tab 1: About ---
    with tab1:
        st.header("What is this?")
        st.markdown("""
        When an AI system approves or rejects an investment, regulators and
        decision-makers need to understand **why**. This tool compares two
        approaches to explaining AI decisions:

        | | LEWIS (Our Method) | SHAP (Industry Standard) |
        |---|---|---|
        | **Approach** | Causal — finds what *causes* the decision | Correlational — finds what *predicts* the decision |
        | **Question it answers** | "If we changed this factor, would the decision flip?" | "How much does this factor contribute to the prediction?" |
        | **Analogy** | A doctor saying "your high blood pressure *caused* the diagnosis" | A doctor saying "patients with your profile usually get this diagnosis" |
        | **Regulatory fit** | Aligns with EU AI Act requirement for causal justification | Widely used but may not satisfy causal requirements |
        """)

        st.divider()

        st.header("Our Dataset")
        st.markdown(f"""
        We analyzed investment decisions for **{meta['n_tickers']} major DAX companies**
        (e.g., SAP, Siemens, Allianz, BMW) using **5 years of monthly data**
        ({meta['n_rows']} total data points).

        **What the AI considers:**
        """)

        # Feature table
        feat_data = []
        for feat in feature_cols:
            desc = FEATURE_DESCRIPTIONS.get(feat, "")
            category = "Macro" if feat in ["ecb_rate", "eur_usd", "de_inflation", "vix"] else "Stock"
            feat_data.append({"Feature": feat, "Category": category, "Description": desc})
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

        st.markdown(f"""
        **How the AI decides:** An investment is approved if the stock's
        risk-adjusted return (return / volatility) exceeds 0.5. The AI model
        (CatBoost) learned this rule with **{meta['accuracy']:.0%} accuracy**.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Companies", meta["n_tickers"])
        col2.metric("Data Points", meta["n_rows"])
        col3.metric("Model Accuracy", f"{meta['accuracy']:.0%}")

    # --- Tab 2: Company View ---
    with tab2:
        st.header("Company Investment Analysis")

        if selected_ticker and "ticker" in df.columns:
            company_name = TICKER_NAMES.get(selected_ticker, selected_ticker)
            st.subheader(f"{company_name} ({selected_ticker})")

            ticker_data = df[df["ticker"] == selected_ticker].copy()
            n_approve = (ticker_data["investment_decision"] == 1).sum()
            n_reject = (ticker_data["investment_decision"] == 0).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Data Points", len(ticker_data))
            col2.metric("APPROVE", n_approve)
            col3.metric("REJECT", n_reject)

            # Show latest data point
            latest = ticker_data.iloc[-1]
            st.markdown("**Latest assessment:**")
            st.markdown(interpret_company_decision(latest, feature_cols))

            # Feature values over time
            st.subheader("Feature Values")
            st.dataframe(
                ticker_data[feature_cols + ["investment_decision"]].describe().round(4),
                use_container_width=True,
            )

            # Per-sample SHAP for this company
            if artifacts["shap_values"] is not None:
                st.subheader("Explanations for this Company")
                ticker_indices = df.index[df["ticker"] == selected_ticker].tolist()

                sample_idx = st.slider(
                    "Select time period (0 = earliest, last = most recent)",
                    0, len(ticker_indices) - 1, len(ticker_indices) - 1,
                )
                idx = ticker_indices[sample_idx]
                row = df.iloc[idx]
                decision = "APPROVE" if row["investment_decision"] == 1 else "REJECT"

                st.markdown(f"**AI Decision: {decision}**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### LEWIS (Causal)")
                    fig_lewis = plot_single_lewis(
                        artifacts["lewis_df"], feature_cols, company_name
                    )
                    st.pyplot(fig_lewis)
                    plt.close()

                    st.markdown("**What this means:**")
                    st.markdown(
                        "The green bars show how much each factor *causally* "
                        "influences the investment decision. A longer bar means "
                        "changing that factor would more likely flip the decision."
                    )
                    st.markdown(interpret_lewis_top_features(artifacts["lewis_df"]))

                with col2:
                    st.markdown("### SHAP (Correlational)")
                    shap_row = artifacts["shap_values"][idx]
                    fig_shap = plot_single_shap(
                        shap_row, feature_cols, company_name
                    )
                    st.pyplot(fig_shap)
                    plt.close()

                    st.markdown("**What this means:**")
                    st.markdown(
                        "Green bars push toward APPROVE, red bars push toward REJECT. "
                        "Longer bars mean the feature had more influence on *this specific* "
                        "prediction — but this is correlation, not causation."
                    )
                    # Interpret this specific SHAP
                    top_pos = sorted(range(len(shap_row)),
                                     key=lambda i: abs(shap_row[i]), reverse=True)[:3]
                    for i in top_pos:
                        feat = feature_cols[i]
                        val = shap_row[i]
                        direction = "toward APPROVE" if val > 0 else "toward REJECT"
                        st.markdown(
                            f"- **{feat}** pushed {direction} "
                            f"(impact: {val:.4f})"
                        )
        else:
            st.info("Select a company from the sidebar to see its analysis.")

    # --- Tab 3: Causal Graph ---
    with tab3:
        st.header("Discovered Causal Graph")
        st.markdown("""
        This graph shows the **cause-and-effect relationships** between factors
        that the algorithm discovered from the data. An arrow from A to B means
        "A causes changes in B."

        **How to read it:** Follow the arrows to understand the chain of causation.
        For example, if `ecb_rate` points to `volatility`, it means changes in
        the ECB rate *cause* changes in stock volatility.
        """)

        fig_graph = plot_causal_graph(artifacts["adj"], feature_cols)
        st.pyplot(fig_graph)
        plt.close()

        # Plain English summary of edges
        col_names = feature_cols + ["investment_decision"]
        st.subheader("Causal Relationships Found")
        edges = []
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if artifacts["adj"][i][j] == 1:
                    edges.append(f"- **{col_names[i]}** causes **{col_names[j]}**")
        if edges:
            st.markdown("\n".join(edges))
        else:
            st.markdown("No direct causal edges found.")

        with st.expander("View raw adjacency matrix"):
            adj_df = pd.DataFrame(
                artifacts["adj"], index=col_names, columns=col_names
            )
            st.dataframe(adj_df, use_container_width=True)

    # --- Tab 4: LEWIS vs SHAP ---
    with tab4:
        st.header("LEWIS vs SHAP: Which Features Matter?")
        st.markdown("""
        Both methods rank features by importance — but they often **disagree**
        because they measure different things:
        - **LEWIS** asks: *"Does changing this feature cause the decision to change?"*
        - **SHAP** asks: *"How much does this feature contribute to the prediction?"*

        A feature can be a strong predictor (high SHAP) without being a cause
        (low LEWIS), and vice versa.
        """)

        if comparison_df is not None:
            fig_comp = plot_comparison_bar(
                comparison_df["lewis_normalized"].values,
                comparison_df["shap_normalized"].values,
                comparison_df["feature"].values,
            )
            st.pyplot(fig_comp)
            plt.close()

            # Plain English interpretation
            st.subheader("What does this mean?")
            st.markdown(interpret_ranking_difference(comparison_df))

            # Ranking table
            st.subheader("Full Ranking Comparison")
            display_df = comparison_df[["feature", "lewis_normalized", "shap_normalized",
                                        "lewis_rank", "shap_rank"]].copy()
            display_df.columns = ["Feature", "LEWIS Score", "SHAP Score",
                                  "LEWIS Rank", "SHAP Rank"]
            display_df["Rank Difference"] = abs(
                display_df["LEWIS Rank"] - display_df["SHAP Rank"]
            )
            display_df = display_df.sort_values("Rank Difference", ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            from scipy import stats
            corr, p_val = stats.spearmanr(
                comparison_df["lewis_normalized"], comparison_df["shap_normalized"]
            )
            st.metric("Spearman Rank Correlation", f"{corr:.4f}")
            if abs(corr) < 0.3:
                st.success(
                    "The low correlation confirms that causal and correlational "
                    "explanations tell **fundamentally different stories**. This is "
                    "the key finding — SHAP alone is not enough for causal understanding."
                )
            elif abs(corr) < 0.7:
                st.info(
                    "Moderate correlation — LEWIS and SHAP partially agree, but there "
                    "are meaningful differences in how they rank features."
                )
            else:
                st.warning(
                    "High correlation — LEWIS and SHAP largely agree on feature rankings."
                )
        else:
            st.warning("Run `python run_shap_comparison.py` to generate comparison data.")

    # --- Tab 5: Vote ---
    with tab5:
        st.header("Your Expert Opinion")
        st.markdown("""
        After reviewing the explanations, we'd like your professional opinion.
        This survey is part of the thesis research — your response helps us
        understand whether domain experts prefer causal or correlational
        explanations for investment decisions.
        """)

        if comparison_df is not None:
            with st.expander("View comparison chart (for reference)", expanded=True):
                fig_ref = plot_comparison_bar(
                    comparison_df["lewis_normalized"].values,
                    comparison_df["shap_normalized"].values,
                    comparison_df["feature"].values,
                )
                st.pyplot(fig_ref)
                plt.close()

        st.divider()

        with st.form("vote_form", clear_on_submit=True):
            st.subheader("Cast Your Vote")

            vote_ticker = "General (all companies)"
            if "ticker" in df.columns:
                ticker_opts = ["General (all companies)"] + [
                    ticker_display(t) for t in sorted(df["ticker"].unique())
                ]
                vote_ticker = st.selectbox(
                    "Which company are you evaluating? (optional)", ticker_opts
                )

            st.markdown("---")
            st.markdown("""
            **LEWIS (Causal)** explains decisions by identifying what *causes*
            the outcome. If a feature has a high LEWIS score, changing it would
            likely flip the decision.

            **SHAP (Correlational)** explains decisions by measuring each
            feature's statistical contribution. A high SHAP score means the
            feature is a strong predictor, but it may not be the actual cause.
            """)

            preference = st.radio(
                "Which explanation do you find more trustworthy for making "
                "investment decisions?",
                ["LEWIS (Causal)", "SHAP (Correlational)", "No preference"],
                horizontal=True,
            )

            comment = st.text_area(
                "Please share your reasoning (optional)",
                placeholder="e.g., 'I prefer LEWIS because it tells me what I "
                           "can actually change to influence the outcome, which is "
                           "more actionable for portfolio management.'",
                height=100,
            )

            submitted = st.form_submit_button("Submit Vote", type="primary")

            if submitted:
                pref_clean = preference.split(" ")[0]
                # Extract raw ticker from display name if selected
                raw_ticker = vote_ticker
                if vote_ticker != "General (all companies)":
                    for t in (df["ticker"].unique() if "ticker" in df.columns else []):
                        if t in vote_ticker:
                            raw_ticker = t
                            break

                decision_text = "N/A"
                save_vote(
                    expert_name=expert_name or "Anonymous",
                    expert_role=expert_role,
                    experience_years=experience_years,
                    ticker=raw_ticker,
                    decision=decision_text,
                    preference=pref_clean,
                    comment=comment,
                )
                st.success(
                    "Thank you for your input! Your vote has been recorded. "
                    "You can see aggregated results in the Survey Results tab."
                )

    # --- Tab 6: Survey Results ---
    with tab6:
        st.header("Survey Results")

        votes_df = load_votes()

        if votes_df.empty:
            st.info("No votes yet. Be the first to share your opinion in the Vote tab!")
            return

        total = len(votes_df)
        st.markdown(f"**{total} response{'s' if total != 1 else ''} collected**")

        # Overall counts
        st.subheader("Overall Preference")
        vote_counts = votes_df["preference"].value_counts()
        lewis_count = vote_counts.get("LEWIS", 0)
        shap_count = vote_counts.get("SHAP", 0)
        no_pref = vote_counts.get("No", 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("LEWIS (Causal)", f"{lewis_count} ({lewis_count/total:.0%})")
        col2.metric("SHAP (Correlational)", f"{shap_count} ({shap_count/total:.0%})")
        col3.metric("No Preference", f"{no_pref} ({no_pref/total:.0%})")

        fig_votes, ax = plt.subplots(figsize=(6, 3))
        colors_map = {"LEWIS": "forestgreen", "SHAP": "steelblue", "No": "gray"}
        ax.bar(vote_counts.index, vote_counts.values,
               color=[colors_map.get(v, "gray") for v in vote_counts.index])
        ax.set_ylabel("Votes")
        ax.set_title("Expert Preference Distribution")
        st.pyplot(fig_votes)
        plt.close()

        # Interpretation
        if lewis_count > shap_count:
            st.success(
                f"**Experts prefer causal explanations (LEWIS)** by a margin of "
                f"{lewis_count - shap_count} vote{'s' if lewis_count - shap_count != 1 else ''}. "
                f"This supports the thesis hypothesis that causal explanations "
                f"are more trusted for investment decisions."
            )
        elif shap_count > lewis_count:
            st.info(
                f"**Experts prefer correlational explanations (SHAP)** by a margin of "
                f"{shap_count - lewis_count} vote{'s' if shap_count - lewis_count != 1 else ''}."
            )
        else:
            st.info("**Results are tied** — no clear preference yet.")

        # By role
        st.subheader("Preference by Expert Role")
        role_pref = pd.crosstab(votes_df["expert_role"], votes_df["preference"])
        st.dataframe(role_pref, use_container_width=True)
        st.markdown(
            "*Do experts in regulated roles (compliance, risk) prefer causal "
            "explanations more than others?*"
        )

        # Comments
        comments_df = votes_df[votes_df["comment"].notna() & (votes_df["comment"] != "")]
        if not comments_df.empty:
            st.subheader("Expert Comments")
            for _, row in comments_df.iterrows():
                with st.expander(
                    f"{row['expert_name']} ({row['expert_role']}, "
                    f"{row['experience_years']}y exp) — Prefers {row['preference']}"
                ):
                    st.write(row["comment"])
                    st.caption(f"Submitted: {row['timestamp']}")

        # Export
        st.divider()
        csv = votes_df.to_csv(index=False)
        st.download_button(
            "Download all votes as CSV (for thesis analysis)",
            csv, "survey_votes.csv", "text/csv",
        )

        with st.expander("View raw vote data"):
            st.dataframe(votes_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
