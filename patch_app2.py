import re

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Remove the st.image("assets/fs_logo.png") from the Hello container
code = code.replace("""            try:
                st.image("assets/fs_logo.png", width=250)
            except Exception:
                pass""", "")

# Add base64 logo injection at the top of the main UI
logo_injection = """
    import base64
    from pathlib import Path
    logo_path = Path("assets/fs_logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'''
                <style>
                .top-right-logo {{
                    position: fixed;
                    top: 60px;
                    right: 30px;
                    width: 120px;
                    z-index: 999999;
                }}
                </style>
                <img src="data:image/png;base64,{logo_base64}" class="top-right-logo">
                ''',
                unsafe_allow_html=True
            )
"""
code = code.replace('    # Load artifacts', '    # Load artifacts\n' + logo_injection)


# 2. Format multiselect options
code = code.replace("""    selected_features = st.multiselect(
        "Features to display over time",
        options=candidate_features,
        default=default_features,
    )""", """    selected_features = st.multiselect(
        "Features to display over time",
        options=candidate_features,
        default=default_features,
        format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
    )""")

# 3. Fix colors for the metrics
color_helper = """
    def color_decision(decision):
        color = "#16a34a" if decision == "BUY" else "#dc2626" if decision == "SELL" else "#ca8a04"
        return f"<div style='font-size: 2.4rem; font-weight: 700; color: {color}; line-height: 1.2;'>{decision}</div>"
"""

code = code.replace("""    d1, d2, d3, d4 = st.columns([1, 1, 1, 1.2])
    d1.metric("BUY", f"{signal_scores['Buy']:.0%}")
    d2.metric("HOLD", f"{signal_scores['Hold']:.0%}")
    d3.metric("SELL", f"{signal_scores['Sell']:.0%}")
    d4.metric("Current Decision", model_decision)""",
    color_helper + """
    d1, d2, d3, d4 = st.columns([1, 1, 1, 1.2])
    d1.metric("BUY", f"{signal_scores['Buy']:.0%}")
    d2.metric("HOLD", f"{signal_scores['Hold']:.0%}")
    d3.metric("SELL", f"{signal_scores['Sell']:.0%}")
    d4.markdown(f"<div><span style='font-size: 0.9rem; color: #475569;'>Current Decision</span><br>{color_decision(model_decision)}</div>", unsafe_allow_html=True)
""")

code = code.replace("""    st.markdown("### Result Summary")
    if result_a == result_b:
        st.success(f"Final Result: {result_a}")
    else:
        r1, r2 = st.columns(2)
        r1.metric("Explanation A Result", result_a)
        r2.metric("Explanation B Result", result_b)""",
"""    st.markdown("### Result Summary")
    if result_a == result_b:
        st.markdown(f"<div style='padding:15px; border-radius:8px; background:#f0fdf4; border:1px solid #bbf7d0;'><b>Final Result:</b><br> {color_decision(result_a)}</div>", unsafe_allow_html=True)
    else:
        r1, r2 = st.columns(2)
        r1.markdown(f"<div class='explanation-box'><span style='font-size: 0.9rem; color: #475569;'>Explanation A Result</span><br>{color_decision(result_a)}</div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='explanation-box'><span style='font-size: 0.9rem; color: #475569;'>Explanation B Result</span><br>{color_decision(result_b)}</div>", unsafe_allow_html=True)""")


with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Changes applied successfully!")
