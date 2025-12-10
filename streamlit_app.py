import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from datetime import time, date

# ---------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Path Finders – Banff Parking Dashboard",
    layout="wide",
)

# ---------------------------------------------------
# GLOBAL PASTEL + GLASS CSS
# ---------------------------------------------------
st.markdown(
    """
    <style>
        /* Pastel "prints" background */
        .stApp {
            background-color: #fdfbff;
            background-image:
                radial-gradient(circle at 0% 0%, rgba(244, 219, 255, 0.75) 0, transparent 55%),
                radial-gradient(circle at 100% 0%, rgba(210, 239, 253, 0.8) 0, transparent 55%),
                radial-gradient(circle at 0% 100%, rgba(209, 250, 229, 0.8) 0, transparent 60%),
                radial-gradient(circle at 100% 100%, rgba(254, 249, 195, 0.75) 0, transparent 55%),
                repeating-linear-gradient(135deg,
                    rgba(255, 255, 255, 0.4) 0px,
                    rgba(255, 255, 255, 0.4) 2px,
                    transparent 2px,
                    transparent 6px);
            color: #1f2933;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Center content */
        .block-container {
            max-width: 1100px;
            padding-top: 3.0rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }

        /* MAIN HERO ON HOME */
        .hero-wrapper {
            text-align: center;
            margin-bottom: 2rem;
        }
        .app-title {
            font-size: 3.4rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #0f172a;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 1rem;
            margin-top: 0.4rem;
        }

        .hero-banner {
            margin: 1.1rem auto 0;
            max-width: 820px;
            padding: 1rem 1.25rem;
            border-radius: 1.2rem;
            background: linear-gradient(135deg, #e0f2fe 0%, #fef3c7 50%, #e9d5ff 100%);
            border: 1px solid rgba(148, 163, 184, 0.55);
            color: #1f2933;
            font-size: 0.9rem;
            box-shadow: 0 18px 40px rgba(148, 163, 184, 0.4);
        }

        /* Sub-page header */
        .sub-header {
            text-align: left;
            margin-bottom: 1.2rem;
        }
        .sub-app-title {
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #0f172a;
        }
        .sub-app-subtitle {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }

        /* Glass cards */
        .glass-card {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(203, 213, 225, 0.9);
            box-shadow: 0 12px 35px rgba(148, 163, 184, 0.35);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }
        .card {
            padding: 0.8rem 1rem;
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(209, 213, 219, 0.9);
        }
        .section-title {
            font-weight: 600;
            margin-bottom: 0.25rem;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b7280;
        }

        /* Glassy metrics */
        .stMetric {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 0.9rem;
            padding: 0.65rem 0.75rem;
            border: 1px solid rgba(209, 213, 219, 0.9);
        }

        /* Global button style (home + back + actions) */
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.7);
            background: linear-gradient(135deg, #bfdbfe, #e5e7eb);
            color: #111827;
            padding: 0.4rem 0.9rem;
            font-size: 0.9rem;
            font-weight: 500;
            box-shadow: 0 7px 18px rgba(148, 163, 184, 0.5);
        }
        .stButton > button:hover {
            border-color: rgba(59, 130, 246, 0.95);
            background: linear-gradient(135deg, #93c5fd, #e5e7eb);
        }

        /* Shrink sliders a little */
        .stSlider > div > div > div {
            padding-top: 0.25rem;
            padding-bottom: 0.05rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# OPENAI CLIENT (SAFE OFFLINE MODE)
# ---------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    reg = joblib.load("banff_best_xgb_reg.pkl")
    cls = joblib.load("banff_best_xgb_cls.pkl")
    scaler = joblib.load("banff_scaler.pkl")
    features = joblib.load("banff_features.pkl")

    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


best_xgb_reg, best_xgb_cls, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

# ===== CSV for dashboard =====
DASHBOARD_CSV = "banff_parking_engineered_HOURLY (1).csv"


@st.cache_data
def load_dashboard_data():
    if os.path.exists(DASHBOARD_CSV):
        df = pd.read_csv(DASHBOARD_CSV)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        df["Hour"] = df["Timestamp"].dt.hour
        return df
    return None

# ---------------------------------------------------
# RAG: LOAD KNOWLEDGE + BUILD VECTORIZER
# ---------------------------------------------------
@st.cache_resource
def load_rag_knowledge():
    knowledge_path = "banff_knowledge.txt"

    if not os.path.exists(knowledge_path):
        docs = [
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt file "
            "is missing, so answers are based only on general parking logic."
        ]
    else:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_embeddings = vectorizer.fit_transform(docs)
    return docs, vectorizer, doc_embeddings


def retrieve_context(query, docs, vectorizer, doc_embeddings, k=5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    selected = [docs[i] for i in top_idx if sims[i] > 0.0]
    if not selected:
        return "No strong matches in the knowledge base. Answer based on general parking logic."
    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    if client is None:
        return (
            "🚫 Chat is running in offline mode (no OpenAI API key is set).\n\n"
            "Here is the most relevant information from the Banff project notes:\n\n"
            f"{context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply."
            ),
        },
        {"role": "system", "content": f"Context from project notes:\n{context}"},
    ]

    for h in chat_history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (
            "⚠️ I could not contact the language model service right now.\n\n"
            "Here is the most relevant information from your notes:\n\n"
            f"{context}"
        )

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def get_time_features_from_inputs(selected_date: date, selected_time: time):
    month = selected_date.month
    day_of_week = selected_date.weekday()
    hour = selected_time.hour
    is_weekend = 1 if day_of_week in [5, 6] else 0
    return month, day_of_week, hour, is_weekend

# ---------------------------------------------------
# PAGE STATE
# ---------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "Home"

def go(view_name: str):
    st.session_state.view = view_name
    st.experimental_rerun()

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if st.session_state.view == "Home":
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="app-title">PATH FINDERS</div>
            <div class="app-subtitle">
                Smart, simple parking insights for Banff – powered by machine learning and pastel-soft XAI.
            </div>
            <div class="hero-banner">
                Choose what you want to see: demand prediction, lot comparisons, explainable AI views,
                or a friendly chat about your project. One click takes you to a focused page.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered vertical stack of navigation buttons
    col_l, col_c, col_r = st.columns([0.3, 0.4, 0.3])
    with col_c:
        st.markdown('<div class="section-title" style="text-align:center;">Open a section</div>', unsafe_allow_html=True)
        if st.button("🎯 Demand prediction", use_container_width=True):
            go("Predict")
        st.write("")  # small gap
        if st.button("📊 Lot overview", use_container_width=True):
            go("Lots")
        st.write("")
        if st.button("🔍 SHAP & XAI", use_container_width=True):
            go("XAI")
        st.write("")
        if st.button("💬 Project assistant", use_container_width=True):
            go("Chat")

# ---------------------------------------------------
# COMMON SUB-PAGE HEADER + BACK BUTTON
# ---------------------------------------------------
def render_sub_header(subtitle: str):
    st.markdown(
        f"""
        <div class="sub-header">
            <div class="sub-app-title">PATH FINDERS</div>
            <div class="sub-app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    back_col, _ = st.columns([0.22, 0.78])
    with back_col:
        if st.button("⬅ Back to home"):
            go("Home")
    st.write("")

# ---------------------------------------------------
# PREDICT PAGE
# ---------------------------------------------------
if st.session_state.view == "Predict":
    render_sub_header("Scenario prediction – pick a date, time, lot, and weather to see occupancy and full-lot risk.")

    st.markdown("#### Scenario prediction")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("**When?**")
        pred_date = st.date_input("Prediction date", value=date.today(), key="pred_date")
        pred_time = st.time_input("Prediction time", value=time(13, 0), key="pred_time")

    with col_right:
        st.markdown("**Weather**")
        max_temp = st.slider("Max temp (°C)", -20.0, 40.0, 22.0, key="pred_temp")
        total_precip = st.slider("Total precip (mm)", 0.0, 30.0, 0.5, key="pred_precip")
        wind_gust = st.slider("Max gust (km/h)", 0.0, 100.0, 12.0, key="pred_gust")

    month, day_of_week, hour, is_weekend = get_time_features_from_inputs(pred_date, pred_time)

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    st.markdown("")
    col_l1, col_l2 = st.columns([1.2, 1])

    with col_l1:
        if lot_features:
            selected_lot_label = st.selectbox("Parking lot", lot_display_names, index=0)
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None
            st.warning("No features starting with 'Unit_' found – lot selection disabled.")

    with col_l2:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Tip</div>
                <div>Set date, time, lot and weather, then click <b>Predict</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    base_input = {f: 0 for f in FEATURES}
    if "Month" in base_input:
        base_input["Month"] = month
    if "DayOfWeek" in base_input:
        base_input["DayOfWeek"] = day_of_week
    if "Hour" in base_input:
        base_input["Hour"] = hour
    if "IsWeekend" in base_input:
        base_input["IsWeekend"] = is_weekend
    if "Max Temp (°C)" in base_input:
        base_input["Max Temp (°C)"] = max_temp
    if "Total Precip (mm)" in base_input:
        base_input["Total Precip (mm)"] = total_precip
    if "Spd of Max Gust (km/h)" in base_input:
        base_input["Spd of Max Gust (km/h)"] = wind_gust

    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    st.markdown("")
    if st.button("🔮 Predict", key="predict_button"):
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted occupancy", f"{occ_pred:.2f}")
        with c2:
            st.metric("Full-lot probability", f"{full_prob:.1%}")

        if full_prob > 0.7:
            st.warning("High risk this lot will be full.")
        elif full_prob > 0.4:
            st.info("Medium risk – expect busy conditions.")
        else:
            st.success("Low risk of the lot being full.")

# ---------------------------------------------------
# LOTS PAGE
# ---------------------------------------------------
if st.session_state.view == "Lots":
    render_sub_header("Lot snapshot – compare all lots at one moment under the same conditions.")

    st.markdown("#### Compare lots at one moment")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        lots_date = st.date_input("Date for status", value=date.today(), key="lots_date")
        lots_time = st.time_input("Time for status", value=time(14, 0), key="lots_time")

    with col_right:
        max_temp = st.slider("Max temp (°C)", -20.0, 40.0, 22.0, key="lots_temp")
        total_precip = st.slider("Total precip (mm)", 0.0, 30.0, 0.5, key="lots_precip")
        wind_gust = st.slider("Max gust (km/h)", 0.0, 100.0, 12.0, key="lots_gust")

    month, day_of_week, hour, is_weekend = get_time_features_from_inputs(lots_date, lots_time)

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    if not lot_features:
        st.error("No features with prefix 'Unit_' in FEATURES. Cannot show lot overview.")
    else:
        base_input = {f: 0 for f in FEATURES}
        if "Month" in base_input:
            base_input["Month"] = month
        if "DayOfWeek" in base_input:
            base_input["DayOfWeek"] = day_of_week
        if "Hour" in base_input:
            base_input["Hour"] = hour
        if "IsWeekend" in base_input:
            base_input["IsWeekend"] = is_weekend
        if "Max Temp (°C)" in base_input:
            base_input["Max Temp (°C)"] = max_temp
        if "Total Precip (mm)" in base_input:
            base_input["Total Precip (mm)"] = total_precip
        if "Spd of Max Gust (km/h)" in base_input:
            base_input["Spd of Max Gust (km/h)"] = wind_gust

        if st.button("Compute lot status", key="lots_button"):
            rows = []
            for lot_feat, lot_name in zip(lot_features, lot_display_names):
                lot_input = base_input.copy()
                if lot_feat in lot_input:
                    lot_input[lot_feat] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = scaler.transform(x_vec)

                occ_pred = best_xgb_reg.predict(x_scaled)[0]
                full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

                if full_prob > 0.7:
                    status = "🟥 High risk"
                elif full_prob > 0.4:
                    status = "🟧 Busy"
                else:
                    status = "🟩 Comfortable"

                rows.append(
                    {
                        "Lot": lot_name,
                        "Predicted occupancy": occ_pred,
                        "Probability full": full_prob,
                        "Status": status,
                    }
                )

            df = pd.DataFrame(rows).sort_values("Lot")

            def lot_status_row_style(row):
                if "High risk" in row["Status"]:
                    return ["background-color: #fee2e2"] * len(row)
                elif "Busy" in row["Status"]:
                    return ["background-color: #fef3c7"] * len(row)
                else:
                    return ["background-color: #dcfce7"] * len(row)

            styled_df = (
                df.style
                .format(
                    {"Predicted occupancy": "{:.2f}", "Probability full": "{:.1%}"}
                )
                .apply(lot_status_row_style, axis=1)
            )

            st.dataframe(styled_df, use_container_width=True)
            st.caption("Row colour shows risk level: red = high, yellow = busy, green = comfortable.")

# ---------------------------------------------------
# XAI PAGE
# ---------------------------------------------------
if st.session_state.view == "XAI":
    render_sub_header("Explainable AI – SHAP, partial dependence plots, and residual checks.")

    st.markdown("#### Explainable AI views")

    st.markdown("**SHAP summary (regression)**")
    try:
        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        fig1, ax1 = plt.subplots(figsize=(6, 3))
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False,
        )
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("**Feature importance (bar)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            plot_type="bar",
            show=False,
        )
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as e:
        st.error(f"Could not generate SHAP plots: {e}")

    st.markdown("**Partial dependence plots**")
    pd_feature_names = [name for name in ["Max Temp (°C)", "Month", "Hour"] if name in FEATURES]
    if pd_feature_names:
        feature_indices = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(7, 3))
        PartialDependenceDisplay.from_estimator(
            best_xgb_reg,
            X_test_scaled,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3,
        )
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Configured PDP features not found in FEATURES; adjust names if needed.")

    st.markdown("**Residual plot**")
    try:
        y_pred = best_xgb_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted occupancy")
        ax4.set_ylabel("Residual (actual − predicted)")
        st.pyplot(fig4)
        plt.close(fig4)
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

# ---------------------------------------------------
# CHAT PAGE
# ---------------------------------------------------
if st.session_state.view == "Chat":
    render_sub_header("Project assistant – ask questions about patterns, busy times, or model behaviour.")

    st.markdown("#### Banff parking assistant")

    if client is None:
        st.warning(
            "Chat is in **offline mode** (no OpenAI API key in environment). "
            "Answers are generated only from `banff_knowledge.txt`."
        )

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about Banff parking…")

    if user_input:
        st.session_state.rag_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            answer = generate_chat_answer(
                user_input,
                st.session_state.rag_chat_history,
            )
            st.markdown(answer)

        st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})

    st.caption(
        "Edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
