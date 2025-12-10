import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# RAG / Chatbot imports
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
# GLOBAL PASTEL CSS
# ---------------------------------------------------
st.markdown(
    """
    <style>
        /* Base app background */
        .stApp {
            background: radial-gradient(circle at top left, #fef6ff 0, #f3f7ff 35%, #f2fbf7 100%);
            color: #1f2933;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Use full width */
        .block-container {
            max-width: 100%;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }

        /* HERO */
        .hero-wrapper {
            text-align: center;
            margin-bottom: 1.2rem;
        }
        .app-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #111827;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .hero-pill-row {
            margin-top: 0.9rem;
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .hero-pill {
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-size: 0.8rem;
            border: 1px solid rgba(148, 163, 184, 0.45);
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: #2563eb;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .hero-banner {
            margin: 0.9rem auto 0;
            max-width: 880px;
            padding: 0.9rem 1.1rem;
            border-radius: 1.1rem;
            background: linear-gradient(135deg, #e0f2fe 0%, #fef3c7 50%, #e9d5ff 100%);
            border: 1px solid rgba(148, 163, 184, 0.45);
            color: #1f2933;
            font-size: 0.9rem;
            box-shadow: 0 16px 40px rgba(148, 163, 184, 0.35);
        }

        /* Cards & glass */
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
            margin-bottom: 0.2rem;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b7280;
        }

        /* Metrics glassy */
        .stMetric {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 0.9rem;
            padding: 0.65rem 0.75rem;
            border: 1px solid rgba(209, 213, 219, 0.9);
        }

        /* Tabs – pastel pills */
        button[role="tab"] {
            border-radius: 999px !important;
            padding: 0.35rem 1.2rem !important;
            border: 1px solid transparent !important;
            font-size: 0.86rem !important;
            background: rgba(255, 255, 255, 0.9) !important;
            color: #4b5563 !important;
            margin-right: 0.25rem !important;
        }
        button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #60a5fa, #a5b4fc) !important;
            color: white !important;
            border-color: rgba(37, 99, 235, 0.4) !important;
        }

        /* Hide sidebar if Streamlit shows it */
        [data-testid="stSidebar"] {
            background-color: transparent !important;
        }

        /* Slightly shrink widgets */
        .stSlider > div > div > div {
            padding-top: 0.3rem;
            padding-bottom: 0.1rem;
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
    """Load trained models, scaler, feature list, and test data."""
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
    """Loads banff_knowledge.txt and builds TF-IDF vectors."""
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
    """Returns top-k most relevant lines from the knowledge base."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    selected = [docs[i] for i in top_idx if sims[i] > 0.0]
    if not selected:
        return "No strong matches in the knowledge base. Answer based on general parking logic."
    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """
    Calls OpenAI ONLY if an API key exists.
    Otherwise, returns a fallback answer based only on retrieved context.
    """
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
                "parking analytics project. Speak clearly and simply for classmates and "
                "instructors who are not data scientists. Use the provided Context as "
                "your main source of truth."
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
# SMALL HELPERS
# ---------------------------------------------------
def get_time_features_from_inputs(selected_date: date, selected_time: time):
    month = selected_date.month
    day_of_week = selected_date.weekday()  # 0=Monday
    hour = selected_time.hour
    is_weekend = 1 if day_of_week in [5, 6] else 0
    return month, day_of_week, hour, is_weekend

# ---------------------------------------------------
# HERO HEADER
# ---------------------------------------------------
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="app-title">PATH FINDERS</div>
        <div class="app-subtitle">
            Banff parking insights with machine learning & pastel-soft explainable AI.
        </div>
        <div class="hero-pill-row">
            <div class="hero-pill">🚗 Demand prediction</div>
            <div class="hero-pill">📊 Lot overview</div>
            <div class="hero-pill">🧊 SHAP & PDP</div>
            <div class="hero-pill">🤖 Project assistant</div>
        </div>
        <div class="hero-banner">
            Pick a date, time, weather and lot – Path Finders predicts occupancy and full-lot risk
            in a few clicks for your CMPT 3830 project demo.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# HORIZONTAL NAV – TABS
# ---------------------------------------------------
tab_overview, tab_predict, tab_lots, tab_xai, tab_chat = st.tabs(
    ["🏠 Overview", "🎯 Predict", "📊 Lots", "🔍 XAI", "💬 Chat"]
)

# ---------------------------------------------------
# TAB 1 – OVERVIEW + DASHBOARD
# ---------------------------------------------------
with tab_overview:
    df_dash = load_dashboard_data()

    col_top_left, col_top_right = st.columns([2, 1])

    with col_top_left:
        st.markdown("#### Today’s snapshot")

        if df_dash is not None:
            total_lots = df_dash["Unit"].nunique()
            total_rows = len(df_dash)
            date_min = df_dash["Date"].min()
            date_max = df_dash["Date"].max()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">Lots</div>
                        <div style="font-size:1.2rem;font-weight:700;">{total_lots}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">Rows</div>
                        <div style="font-size:1.2rem;font-weight:700;">{total_rows}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">Date range</div>
                        <div style="font-size:0.9rem;font-weight:600;">{date_min} → {date_max}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("CSV file not found – dashboard will show once the file is in the app folder.")

    with col_top_right:
        today = date.today()
        selected_date = st.date_input("Overview date", value=today)

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">Selected date</div>
                <div style="font-size:1rem;font-weight:600;">
                    {selected_date.strftime('%b %d, %Y')} ({selected_date.strftime('%A')})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    if df_dash is not None:
        day_df = df_dash[df_dash["Date"] == selected_date]

        if day_df.empty:
            st.info("No data for this date in the CSV.")
        else:
            lots = sorted(day_df["Unit"].unique())
            lot_choice = st.selectbox("Lot filter", ["All lots"] + lots)

            if lot_choice != "All lots":
                day_df = day_df[day_df["Unit"] == lot_choice]

            avg_occ = day_df["Percent_Occupancy"].mean()
            peak_occ = day_df["Percent_Occupancy"].max()
            full_hours = int(day_df["Is_Full"].sum())

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Avg % occupancy", f"{avg_occ:.1f}%")
            with k2:
                st.metric("Peak % occupancy", f"{peak_occ:.1f}%")
            with k3:
                st.metric("Hours near full", f"{full_hours}")
            with k4:
                st.metric("Rows", f"{len(day_df)}")

            hourly = (
                day_df.groupby("Hour")["Percent_Occupancy"]
                .mean()
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(6, 3))  # smaller pastel chart
            ax.plot(hourly.index, hourly.values, marker="o")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Avg % occupancy")
            ax.set_xticks(list(hourly.index))
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

            st.caption("Average occupancy by hour for the selected date and lot filter.")

# ---------------------------------------------------
# TAB 2 – PREDICTION (calendar + time input)
# ---------------------------------------------------
with tab_predict:
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

    # Lot selection
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
                <div>Pick a date, time, lot and weather, then click <b>Predict</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Build feature dict
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
# TAB 3 – LOT STATUS OVERVIEW
# ---------------------------------------------------
with tab_lots:
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
                    return ["background-color: #fee2e2"] * len(row)  # pastel red
                elif "Busy" in row["Status"]:
                    return ["background-color: #fef3c7"] * len(row)  # pastel yellow
                else:
                    return ["background-color: #dcfce7"] * len(row)  # pastel green

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
# TAB 4 – XAI
# ---------------------------------------------------
with tab_xai:
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
# TAB 5 – CHAT (RAG)
# ---------------------------------------------------
with tab_chat:
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
