import re

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update FEATURE_DESCRIPTIONS
code = re.sub(
    r"FEATURE_DESCRIPTIONS = \{.*?\n\}",
    """FEATURE_DESCRIPTIONS = {
    "volatility": "Price volatility (risk)",
    "momentum": "Recent price trend",
    "volume_avg": "Average trading volume",
    "return_1y": "Return over the past year",
    "max_drawdown": "Maximum drop in the past year",
    "ecb_rate": "ECB main refinancing rate",
    "eur_usd": "Euro to US Dollar exchange rate",
    "de_inflation": "German inflation rate",
    "vix": "Market fear index (VIX)",
}""",
    code,
    flags=re.DOTALL
)

# 2. Update format strings
code = code.replace(":.3f", ":.2f").replace(":.4f", ":.2f")

# 3. Update build_explanation_text
code = code.replace('parts.append(f"{feat} ({desc})")', 'parts.append(desc)')

# 4. Update explanation_for_decision
code = code.replace('lines.append(f"- {feat}: {desc}")', 'lines.append(f"- {desc}")')

# 5. Update render_neutral_explanation_chart
code = code.replace('ax.barh(top["feature"], top[score_col], color="#2A6EA6")',
                    'labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in top["feature"]]\n    ax.barh(labels, top[score_col], color="#2A6EA6")')

# 6. Update plot_feature_trends
code = code.replace('ax.plot(series["date"], y, linewidth=2.2, color=color, label=feat)',
                    'desc = FEATURE_DESCRIPTIONS.get(feat, feat)\n        ax.plot(series["date"], y, linewidth=2.2, color=color, label=desc)')

# 7. Update build_counterfactual_explanation
code = code.replace('lines.append(\n            f"- **{direction}** {feat} from **{current_val:.2f}** to **{desired:.2f}** (delta **{delta:+.2f}**)"\n        )',
                    'desc = FEATURE_DESCRIPTIONS.get(feat, feat)\n        lines.append(\n            f"- **{direction}** {desc} from **{current_val:.2f}** to **{desired:.2f}** (delta **{delta:+.2f}**)"\n        )')

# 8. Add Hello page to main()
hello_page_code = """
    if "survey_started" not in st.session_state:
        st.session_state.survey_started = False

    if not st.session_state.survey_started:
        st.markdown(
            \"\"\"
            <style>
            .hello-container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.05);
                font-family: 'Inter', sans-serif;
            }
            .title-text {
                color: #1e3a8a;
                font-size: 2rem;
                font-weight: 800;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .subtitle-text {
                color: #475569;
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            .step-box {
                background: #f8fafc;
                border-left: 4px solid #3b82f6;
                padding: 15px 20px;
                margin-bottom: 15px;
                border-radius: 0 8px 8px 0;
            }
            </style>
            \"\"\",
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='hello-container'>", unsafe_allow_html=True)
            try:
                st.image("assets/fs_logo.png", width=250)
            except Exception:
                pass
                
            st.markdown(
                \"\"\"
                <div class="title-text">Expert Survey</div>
                <div class="subtitle-text">
                    Welcome! We are conducting research on <b>Causal Explainability for AI-Driven Investment Decisions in German Banks</b>.<br><br>
                    <b>Researchers:</b> Kailash Selvan & Mutharasan (Frankfurt School MiM Students)<br><br>
                    We have developed, trained, and tested an AI model for investment decisions. We are conducting this survey to see if <b>explainability (causal inference)</b> and <b>counterfactual explanations</b> make a difference in user trust compared to traditional correlational methods.
                </div>
                
                <h3>Steps to complete:</h3>
                
                <div class="step-box">
                    <b>1. Choose a stock in DAX30</b><br>
                    <i>(We suggest to pick 2 stocks and see the results)</i>
                </div>
                
                <div class="step-box">
                    <b>2. Choose the indicators</b><br>
                    Review the trends over time for the stock you selected.
                </div>
                
                <div class="step-box">
                    <b>3. Check the output and the explanations</b><br>
                    Compare Explanation A and Explanation B, then answer the short survey below them.
                </div>
                \"\"\",
                unsafe_allow_html=True
            )
            
            st.write("")
            st.write("")
            if st.button("Start Survey", type="primary", use_container_width=True):
                st.session_state.survey_started = True
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
        return
"""
code = code.replace('    artifacts = load_artifacts()', hello_page_code + '\n    artifacts = load_artifacts()')

# 9. Update Causal Tree Explanation
causal_explanation = """
    st.markdown("### Causal Graph & Explanations")
    st.markdown(\"\"\"
    <div style='background-color: #eef2ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4f46e5;'>
    <b>What are you seeing?</b><br>
    • The <b>Causal Tree (Graph)</b> shows the cause-and-effect relationships between different financial indicators and the investment decision. Arrows indicate that changing one feature directly impacts the other.<br>
    • <b>Counterfactual Explanations</b> show you 'what-if' scenarios. They highlight the minimal changes in the indicators that would be needed to flip the AI's current decision (for example, from HOLD to BUY).
    </div>
    \"\"\", unsafe_allow_html=True)
"""
code = code.replace('    st.markdown("### Feature Glossary & Causal Graph")', causal_explanation + '\n    st.markdown("### Feature Glossary")')

# Optional: Add Inter font to the main CSS
inter_font = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
code = code.replace('.stApp {', inter_font + '\n            * { font-family: \'Inter\', sans-serif; }\n            .stApp {')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Changes applied successfully!")
