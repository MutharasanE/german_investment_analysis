"""
Streamlit Demo: Causal vs Correlational Explanations for AI Investment Decisions
Master Thesis — Frankfurt School of Finance & Management

Usage: streamlit run app.py
"""

import json
import random
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
            timestamp TEXT NOT NULL,
            chosen_label TEXT,
            method_for_a TEXT,
            method_for_b TEXT,
            chosen_method TEXT,
            survey_variant TEXT,
            actionability_choice TEXT,
            trust_score INTEGER,
            confidence_score INTEGER
        )
    """)

    # Backward-compatible schema upgrades for existing local DBs.
    c.execute("PRAGMA table_info(votes)")
    existing_cols = {row[1] for row in c.fetchall()}
    optional_cols = {
        "chosen_label": "TEXT",
        "method_for_a": "TEXT",
        "method_for_b": "TEXT",
        "chosen_method": "TEXT",
        "survey_variant": "TEXT",
        "actionability_choice": "TEXT",
        "trust_score": "INTEGER",
        "confidence_score": "INTEGER",
        "mechanics_feedback": "TEXT",
    }
    for col_name, col_type in optional_cols.items():
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE votes ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def save_vote(expert_name, expert_role, experience_years, ticker, decision,
              preference, comment, chosen_label=None, method_for_a=None,
              method_for_b=None, chosen_method=None, survey_variant=None,
              actionability_choice=None, trust_score=None, confidence_score=None,
              mechanics_feedback=None):
    """Save a single vote."""
    vote = {
        "expert_name": expert_name,
        "expert_role": expert_role,
        "experience_years": experience_years,
        "ticker": ticker,
        "decision": decision,
        "preference": preference,
        "comment": comment,
        "chosen_label": chosen_label,
        "method_for_a": method_for_a,
        "method_for_b": method_for_b,
        "chosen_method": chosen_method,
        "survey_variant": survey_variant,
        "actionability_choice": actionability_choice,
        "trust_score": trust_score,
        "confidence_score": confidence_score,
        "mechanics_feedback": mechanics_feedback,
        "timestamp": datetime.now().isoformat(),
    }
    if MONGO_URI:
        _get_mongo_collection().insert_one(vote)
    else:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO votes (expert_name, expert_role, experience_years,
                                                            ticker, decision, preference, comment, timestamp,
                                                            chosen_label, method_for_a, method_for_b,
                                                            chosen_method, survey_variant,
                                                            actionability_choice, trust_score, confidence_score, mechanics_feedback)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (expert_name, expert_role, experience_years, ticker, decision,
                            preference, comment, vote["timestamp"], chosen_label,
                            method_for_a, method_for_b, chosen_method, survey_variant,
                            actionability_choice, trust_score, confidence_score, mechanics_feedback))
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


def ticker_to_file_stem(ticker):
    """Convert market ticker to local CSV stem naming convention."""
    return ticker.replace(".", "_")


@st.cache_data
def load_ticker_price_data(ticker):
    """Load per-ticker historical OHLCV data for plotting."""
    csv_path = DATA_DIR / f"{ticker_to_file_stem(ticker)}.csv"
    if not csv_path.exists():
        return None

    df_price = pd.read_csv(csv_path)
    if "Date" not in df_price.columns:
        return None

    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
    df_price = df_price.dropna(subset=["Date"]).sort_values("Date")
    return df_price[df_price["Date"] >= pd.Timestamp("2021-01-01")]


def score_buy_hold_sell(ticker_data):
    """Create a transparent proxy for Buy/Hold/Sell signal strengths."""
    if ticker_data.empty:
        return {"Buy": 0.0, "Hold": 1.0, "Sell": 0.0}

    approve_rate = float(ticker_data["investment_decision"].mean())
    buy = max(0.0, min(1.0, approve_rate))
    sell = max(0.0, min(1.0, 1.0 - approve_rate))
    hold = max(0.0, 1.0 - abs(buy - sell))

    total = buy + hold + sell
    if total <= 0:
        return {"Buy": 0.0, "Hold": 1.0, "Sell": 0.0}
    return {"Buy": buy / total, "Hold": hold / total, "Sell": sell / total}


def build_explanation_text(score_df, score_col, n=3):
    """Build neutral plain-language explanation text from top feature scores."""
    top = score_df.nlargest(n, score_col)[["feature", score_col]]
    if top.empty:
        return "This explanation is based on the strongest feature signals available."

    parts = []
    for _, row in top.iterrows():
        feat = row["feature"]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        parts.append(f"{feat} ({desc})")
    return "Top drivers in this explanation are: " + ", ".join(parts) + "."


def render_neutral_explanation_chart(score_df, score_col, title):
    """Render a neutral explanation chart without exposing method identity."""
    top = score_df.nlargest(6, score_col).sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(top["feature"], top[score_col], color="#2A6EA6")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    return fig


def plot_feature_trends(ticker_data, features, company_name):
    """Render a polished multi-line feature trend chart."""
    palette = ["#0F766E", "#2563EB", "#B45309", "#DC2626", "#7C3AED", "#0891B2"]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))

    for idx, feat in enumerate(features):
        series = ticker_data[["date", feat]].dropna().copy()
        if series.empty:
            continue

        std = float(series[feat].std()) if len(series) > 1 else 0.0
        if std > 0:
            y = (series[feat] - float(series[feat].mean())) / std
        else:
            y = series[feat] * 0.0

        color = palette[idx % len(palette)]
        ax.plot(series["date"], y, linewidth=2.2, color=color, label=feat)

    ax.set_title(f"{company_name} - Feature Trends Over Time", fontsize=12.5, pad=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Standardized Value", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def predict_decision_from_scores(signal_scores):
    """Convert Buy/Hold/Sell score dict into a displayed decision label."""
    decision = max(signal_scores, key=signal_scores.get)
    return decision.upper()


def explanation_for_decision(score_df, score_col, decision_label, n=3):
    """Build concise decision-grounded explanation from top-ranked features."""
    top = score_df.nlargest(n, score_col)[["feature", score_col]]
    if top.empty:
        return f"This explanation supports a {decision_label} view based on the available signals."

    lines = [f"Why {decision_label}: this view gives most weight to these drivers:"]
    for _, row in top.iterrows():
        feat = row["feature"]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        lines.append(f"- {feat}: {desc}")
    return "\n".join(lines)


def target_decision_for_counterfactual(current_decision):
    """Choose a practical target decision for counterfactual guidance."""
    if current_decision == "BUY":
        return "HOLD"
    if current_decision == "HOLD":
        return "BUY"
    return "HOLD"


def build_counterfactual_explanation(ticker_data, score_df, score_col, current_decision, n=3):
    """Create actionable counterfactual edits that could flip the decision."""
    target_decision = target_decision_for_counterfactual(current_decision)

    if ticker_data.empty or score_df is None or score_col not in score_df.columns:
        text = (
            f"Current decision: {current_decision}. "
            f"Counterfactual target: {target_decision}.\n"
            "No sufficient data was available to generate feature-change actions."
        )
        return target_decision, text

    latest = ticker_data.sort_values("date").iloc[-1]
    top_features = score_df.nlargest(8, score_col)["feature"].tolist()

    increase_for_buy = {"return_1y", "momentum", "volume_avg"}
    decrease_for_buy = {"volatility", "max_drawdown", "vix", "ecb_rate", "de_inflation"}

    actions = []
    for feat in top_features:
        if feat not in ticker_data.columns:
            continue

        series = ticker_data[feat].dropna()
        if series.empty:
            continue

        current_val = float(latest.get(feat, np.nan))
        if np.isnan(current_val):
            continue

        if target_decision == "BUY":
            if feat in increase_for_buy:
                desired = float(series.quantile(0.75))
                direction = "Increase"
            elif feat in decrease_for_buy:
                desired = float(series.quantile(0.25))
                direction = "Decrease"
            else:
                continue
        else:
            if feat in increase_for_buy:
                desired = float(series.quantile(0.45))
                direction = "Decrease"
            elif feat in decrease_for_buy:
                desired = float(series.quantile(0.55))
                direction = "Increase"
            else:
                continue

        delta = desired - current_val
        if abs(delta) < 1e-9:
            continue

        actions.append((feat, direction, current_val, desired, delta))
        if len(actions) >= n:
            break

    if not actions:
        text = (
            f"Current decision: {current_decision}. Counterfactual target: {target_decision}.\n"
            "No robust minimal edits were found from the top ranked drivers."
        )
        return target_decision, text

    lines = [
        f"Current decision: **{current_decision}**",
        f"Counterfactual target: **{target_decision}**",
        "If the following changes occur together, the decision is more likely to flip:",
    ]
    for feat, direction, current_val, desired, delta in actions:
        lines.append(
            f"- **{direction}** {feat} from **{current_val:.3f}** to **{desired:.3f}** (delta **{delta:+.3f}**)"
        )
    return target_decision, "\n".join(lines)


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
        page_title="Expert Survey",
        page_icon="🗳️",
        layout="wide",
    )

    init_db()

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 55%, #f8fbff 100%);
            }
            .survey-card {
                border: 1px solid #dbe7ff;
                background: #ffffff;
                border-radius: 14px;
                padding: 18px 20px;
                box-shadow: 0 6px 18px rgba(30, 64, 175, 0.08);
                margin-bottom: 12px;
            }
            .survey-title {
                font-size: 1.6rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 2px;
            }
            .survey-subtitle {
                color: #334155;
                font-size: 0.98rem;
            }
            .survey-question {
                color: #0f172a;
                font-weight: 700;
                font-size: 1.02rem;
                margin-bottom: 6px;
            }
            .explanation-box {
                border: 1px solid #d5e2ff;
                border-radius: 12px;
                padding: 12px;
                background: #ffffff;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            }
            label, .stMarkdown, p, .stRadio label, .stSelectbox label, .stMultiSelect label {
                color: #111827 !important;
            }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: inherit !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="survey-card">
            <div class="survey-title">Expert Survey</div>
            <div class="survey-subtitle">Please review the selected ticker and feature trends, then answer the short survey.</div>
        </div>
        """,
        unsafe_allow_html=True,
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

    if "ticker" not in df.columns or "date" not in df.columns:
        st.error("Dataset must contain 'ticker' and 'date' columns for this survey view.")
        return

    st.sidebar.header("Participant Profile")
    expert_name = st.sidebar.text_input("Name (optional)")
    expert_role = st.sidebar.selectbox("Role", [
        "Portfolio Manager",
        "Risk Analyst",
        "Compliance Officer",
        "Data Scientist",
        "Academic / Researcher",
        "Student",
        "Other",
    ])
    experience_years = st.sidebar.slider("Years of experience", 0, 30, 5)

    tickers = sorted(df["ticker"].dropna().unique())
    dax_tickers = [t for t in tickers if t.endswith(".DE")]

    controls_left, controls_right = st.columns([1.2, 1])
    with controls_left:
        if not dax_tickers:
            st.warning("No DAX tickers are available in the current dataset.")
            return

        ticker_options = {ticker_display(t): t for t in dax_tickers}
        selected_display = st.selectbox("Select ticker", list(ticker_options.keys()))
        selected_ticker = ticker_options[selected_display]
    with controls_right:
        st.markdown("<div class='survey-card'><b>Survey Design:</b> neutral labels, randomized option order, single-page flow.</div>", unsafe_allow_html=True)

    ticker_data = df[df["ticker"] == selected_ticker].copy()
    ticker_data["date"] = pd.to_datetime(ticker_data["date"], errors="coerce")
    ticker_data = ticker_data.dropna(subset=["date"]).sort_values("date")
    ticker_data = ticker_data[ticker_data["date"] >= pd.Timestamp("2021-01-01")]

    if ticker_data.empty:
        st.warning("No data available for this ticker in the selected period.")
        return

    # First show decision context to minimize cognitive load.
    signal_scores = score_buy_hold_sell(ticker_data)
    model_decision = predict_decision_from_scores(signal_scores)
    d1, d2, d3, d4 = st.columns([1, 1, 1, 1.2])
    d1.metric("BUY", f"{signal_scores['Buy']:.0%}")
    d2.metric("HOLD", f"{signal_scores['Hold']:.0%}")
    d3.metric("SELL", f"{signal_scores['Sell']:.0%}")
    d4.metric("Current Decision", model_decision)

    candidate_features = [f for f in feature_cols if f in ticker_data.columns]
    default_features = candidate_features[:4] if len(candidate_features) >= 4 else candidate_features
    selected_features = st.multiselect(
        "Features to display over time",
        options=candidate_features,
        default=default_features,
    )

    if selected_features:
        company_name = TICKER_NAMES.get(selected_ticker, selected_ticker)
        fig = plot_feature_trends(ticker_data, selected_features, company_name)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("Select at least one feature to render the line chart.")

    # Anonymous parallel explanation panels.
    map_key = f"ab_mapping_{selected_ticker}"
    
    st.markdown("### Feature Glossary & Causal Graph")
    st.markdown("""
    **How features are calculated:**
    - **Momentum (1m, 3m, 6m):** Relative price strength comparing recent close to historical averages. Calculated as `price / past_price - 1.0`.
    - **Volatility:** The standard deviation of daily returns over a 21-day rolling window, tracking price instability.
    - **Volume Avg:** The 21-day simple moving average of trading volume, indicating market participation.
    - **Return 1y:** Normalized 1-year trailing return.
    - **RSI (14):** Relative Strength Index, an oscillator from 0 to 100 measuring overbought/oversold conditions.
    - **MACD Signal:** Moving Average Convergence Divergence signal line, representing the difference between short-term and long-term EMA.
    - **Max Drawdown:** The maximum observed loss from a recent peak, capturing downside risk.
    - **Beta Market:** Sensitivity to the overall benchmark index (e.g., S&P 500).
    - **ECB Rate, Us 10y Yield, Inflation:** Macro-economic base interest rates and Consumer Price Indices.
    """)

    causal_graph_path = Path("results/plots/causal_graph_directlingam.png")
    if causal_graph_path.exists():
        st.image(str(causal_graph_path), caption="Causal Graph — this graph powers the counterfactual explanation below", use_container_width=True)

    if map_key not in st.session_state:
        if random.random() < 0.5:
            st.session_state[map_key] = {"A": "COUNTERFACTUAL", "B": "FEATURE_IMPORTANCE"}
        else:
            st.session_state[map_key] = {"A": "FEATURE_IMPORTANCE", "B": "COUNTERFACTUAL"}

    mapping = st.session_state[map_key]
    method_a = mapping["A"]
    method_b = mapping["B"]

    causal_label = "A" if method_a == "COUNTERFACTUAL" else "B"
    st.info(f"The causal graph above powers **Explanation {causal_label}** (the counterfactual explanation).")

    # Source data for arm rendering (internal only, never shown to participants).
    importance_df = artifacts["shap_df"] if artifacts["shap_df"] is not None else artifacts["lewis_df"]
    importance_metric = "shap_normalized" if artifacts["shap_df"] is not None else "maxNesuf"
    counterfactual_df = artifacts["lewis_df"] if artifacts["lewis_df"] is not None else artifacts["shap_df"]
    counterfactual_metric = "maxNesuf" if artifacts["lewis_df"] is not None else "shap_normalized"

    arm_defs = {
        "FEATURE_IMPORTANCE": (importance_df, importance_metric),
        "COUNTERFACTUAL": (counterfactual_df, counterfactual_metric),
    }

    result_a = model_decision
    result_b = model_decision

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        st.markdown("#### Explanation A")
        data_a, metric_a = arm_defs[method_a]
        if data_a is not None and metric_a in data_a.columns:
            fig_a = render_neutral_explanation_chart(data_a, metric_a, "Explanation A")
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)
            if method_a == "COUNTERFACTUAL":
                result_a, text_a = build_counterfactual_explanation(
                    ticker_data=ticker_data,
                    score_df=data_a,
                    score_col=metric_a,
                    current_decision=model_decision,
                )
                st.markdown(text_a)
            else:
                text_a = explanation_for_decision(data_a, metric_a, model_decision)
                st.markdown(text_a)
        else:
            st.info("Explanation A is unavailable for this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        st.markdown("#### Explanation B")
        data_b, metric_b = arm_defs[method_b]
        if data_b is not None and metric_b in data_b.columns:
            fig_b = render_neutral_explanation_chart(data_b, metric_b, "Explanation B")
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
            if method_b == "COUNTERFACTUAL":
                result_b, text_b = build_counterfactual_explanation(
                    ticker_data=ticker_data,
                    score_df=data_b,
                    score_col=metric_b,
                    current_decision=model_decision,
                )
                st.markdown(text_b)
            else:
                text_b = explanation_for_decision(data_b, metric_b, model_decision)
                st.markdown(text_b)
        else:
            st.info("Explanation B is unavailable for this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Result Summary")
    if result_a == result_b:
        st.success(f"Final Result: {result_a}")
    else:
        r1, r2 = st.columns(2)
        r1.metric("Explanation A Result", result_a)
        r2.metric("Explanation B Result", result_b)

    st.markdown("---")
    st.subheader("Survey")

    order_key = "survey_option_order"
    if order_key not in st.session_state:
        options = ["Explanation A", "Explanation B"]
        random.shuffle(options)
        st.session_state[order_key] = ["No preference"] + options

    with st.form("minimal_survey_form", clear_on_submit=True):
        st.markdown("<div class='survey-question'>Which explanation is more convincing for this decision?</div>", unsafe_allow_html=True)
        preference_display = st.radio(
            "Preference",
            st.session_state[order_key],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("<div class='survey-question'>Why did you choose this explanation? (short answer)</div>", unsafe_allow_html=True)
        preference_reason = st.text_area(
            "Preference Reason",
            placeholder="e.g. It felt more actionable / clearer drivers / better matched my intuition",
            height=70,
            label_visibility="collapsed",
        )

        st.markdown("<div class='survey-question'>Trust in your chosen explanation (1=Low, 7=High)</div>", unsafe_allow_html=True)
        trust_score = st.slider("Trust Score", 1, 7, 5, label_visibility="collapsed")

        st.markdown("<div class='survey-question'>Confidence in making a decision using your chosen explanation (1=Low, 7=High)</div>", unsafe_allow_html=True)
        confidence_score = st.slider("Confidence Score", 1, 7, 5, label_visibility="collapsed")

        st.markdown("<div class='survey-question'>Do you feel the highlighted counterfactual changes (in bold) accurately reflect real-world market mechanics?</div>", unsafe_allow_html=True)
        mechanics_display = st.radio(
            "Real World Mechanics",
            ["Yes, highly accurate", "Somewhat accurate", "No, they seem theoretical", "Unsure"],
            horizontal=False,
            label_visibility="collapsed",
        )

        st.markdown("<div class='survey-question'>Quality check: please select \"Strongly Agree\" below.</div>", unsafe_allow_html=True)
        attention_check = st.radio(
            "Attention Check",
            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
            horizontal=True,
            label_visibility="collapsed",
        )

        comment = st.text_area(
            "Optional comment or suggestions for improvement",
            placeholder="One short reason (optional)",
            height=70,
        )

        submitted = st.form_submit_button("Submit", type="primary")

        if submitted:
            attention_passed = attention_check == "Strongly Agree"
            chosen_label = "A" if preference_display == "Explanation A" else "B" if preference_display == "Explanation B" else "No"
            save_vote(
                expert_name=expert_name or "Anonymous",
                expert_role=expert_role,
                experience_years=experience_years,
                ticker=selected_ticker,
                decision=f"decision_a:{result_a}|decision_b:{result_b}|features:{','.join(selected_features)}|attention:{'pass' if attention_passed else 'fail'}",
                preference=chosen_label,
                comment=f"Reason: {preference_reason}\n---\n{comment}" if preference_reason else comment,
                chosen_label=chosen_label,
                method_for_a=method_a,
                method_for_b=method_b,
                chosen_method=(method_a if chosen_label == "A" else method_b if chosen_label == "B" else "No"),
                survey_variant="minimal_unbiased_single_page_v3_counterfactual",
                trust_score=trust_score,
                confidence_score=confidence_score,
                mechanics_feedback=mechanics_display,
            )

            # Re-randomize option order for each new submission to reduce order bias.
            if order_key in st.session_state:
                del st.session_state[order_key]

            st.success("Response recorded. Thank you.")


if __name__ == "__main__":
    main()
