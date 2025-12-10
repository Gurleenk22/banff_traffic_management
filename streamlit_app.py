import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# ==== RAG / Chatbot imports ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from datetime import datetime, time, date

# ---------------------------------------------------
# BASIC PAGE CONFIG + LIGHT CSS
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f7;
        }
        .app-header {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            background-color: white;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.09);
        }
        .section-title {
            font-weight: 600;
            margin-bottom: 0.4rem;
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

# ===== Optional CSV for dashboard =====
DASHBOARD_CSV = "banff_dashboard.csv"


@st.cache_data
def load_dashboard_data():
    if os.path.exists(DASHBOARD_CSV):
        try:
            return pd.read_csv(DASHBOARD_CSV)
        except Exception:
            return None
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

    # --- Offline mode (no key) ---
    if client is None:
        return (
            "🚫 Chat is running in offline mode (no OpenAI API key is set).\n\n"
            "Here is the most relevant information from the Banff project notes:\n\n"
            f"{context}"
        )

    # --- Online mode ---
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
# TOP HEADER
# ---------------------------------------------------
st.markdown('<div class="app-header">Banff Parking Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Quick insights, predictions, and explainability for Banff lots.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# HORIZONTAL NAV – TABS (no sidebar pages)
# ---------------------------------------------------
tab_overview, tab_predict, tab_lots, tab_xai, tab_chat = st.tabs(
    ["🏠 Overview", "🎯 Predict", "📊 Lots", "🔍 XAI", "💬 Chat"]
)

# ---------------------------------------------------
# TAB 1 – OVERVIEW + SIMPLE DASHBOARD
# ---------------------------------------------------
with tab_overview:
    col_top_left, col_top_right = st.columns([2, 1])

    with col_top_left:
        st.markdown("### Dashboard snapshot")
        st.caption("High-level view of current parking behaviour.")

    with col_top_right:
        today = date.today()
        selected_date = st.date_input("Date", value=today, label_visibility="collapsed")

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">Selected date</div>
                <div>{selected_date.strftime('%b %d, %Y')} ({selected_date.strftime('%A')})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # KPI cards row
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Model targets</div>
                <div>Hourly occupancy & full-lot risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Season focus</div>
                <div>May – September (tourist season)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Lots modelled</div>
                <div>Multiple Banff units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### CSV overview")

    df_dash = load_dashboard_data()
    if df_dash is None:
        st.info(
            f"Place a CSV named **{DASHBOARD_CSV}** in the app folder to feed this dashboard "
            "with real metrics (e.g., daily occupancy, arrivals, departures)."
        )
    else:
        # show a quick summary of the CSV
        st.dataframe(df_dash.head(), use_container_width=True)
        st.caption("Preview of your CSV. You can customise charts here if needed.")

# ---------------------------------------------------
# Helper: convert calendar date/time -> model inputs
# ---------------------------------------------------
def get_time_features_from_inputs(selected_date: date, selected_time: time):
    month = selected_date.month
    day_of_week = selected_date.weekday()  # 0=Monday
    hour = selected_time.hour
    is_weekend = 1 if day_of_week in [5, 6] else 0
    return month, day_of_week, hour, is_weekend

# ---------------------------------------------------
# TAB 2 – PREDICTION (calendar instead of sliders)
# ---------------------------------------------------
with tab_predict:
    st.markdown("### Scenario prediction")

    col_left, col_right = st.columns([1.2, 1])

    # date + time pickers
    with col_left:
        st.markdown("**When?**")
        pred_date = st.date_input("Prediction date", value=date.today())
        pred_time = st.time_input("Prediction time", value=time(13, 0))

    with col_right:
        st.markdown("**Weather**")
        max_temp = st.slider("Max temp (°C)", -20.0, 40.0, 22.0)
        total_precip = st.slider("Total precip (mm)", 0.0, 30.0, 0.5)
        wind_gust = st.slider("Max gust (km/h)", 0.0, 100.0, 12.0)

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
    if st.button("🔮 Predict"):
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted occupancy (model units)", f"{occ_pred:.2f}")
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
    st.markdown("### Compare lots at one moment")

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

        if st.button("Compute lot status"):
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
                    return ["background-color: #ffe5e5"] * len(row)
                elif "Busy" in row["Status"]:
                    return ["background-color: #fff4e0"] * len(row)
                else:
                    return ["background-color: #e9f7ef"] * len(row)

            styled_df = (
                df.style
                .format(
                    {"Predicted occupancy": "{:.2f}", "Probability full": "{:.1%}"}
                )
                .apply(lot_status_row_style, axis=1)
            )

            st.dataframe(styled_df, use_container_width=True)
            st.caption("Row colour shows risk level: red = high, orange = busy, green = comfortable.")

# ---------------------------------------------------
# TAB 4 – XAI
# ---------------------------------------------------
with tab_xai:
    st.markdown("### Explainable AI views")

    # SHAP
    st.markdown("**SHAP summary (regression)**")
    try:
        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False,
        )
        st.pyplot(fig1)

        st.markdown("**Feature importance (bar)**")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            plot_type="bar",
            show=False,
        )
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Could not generate SHAP plots: {e}")

    # PDP
    st.markdown("**Partial dependence plots**")
    pd_feature_names = [name for name in ["Max Temp (°C)", "Month", "Hour"] if name in FEATURES]
    if pd_feature_names:
        feature_indices = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            best_xgb_reg,
            X_test_scaled,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3,
        )
        st.pyplot(fig3)
    else:
        st.info("Configured PDP features not found in FEATURES; adjust names if needed.")

    # Residuals
    st.markdown("**Residual plot**")
    try:
        y_pred = best_xgb_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted occupancy")
        ax4.set_ylabel("Residual (actual − predicted)")
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

# ---------------------------------------------------
# TAB 5 – CHAT (RAG)
# ---------------------------------------------------
with tab_chat:
    st.markdown("### Banff parking assistant")

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
            with st.spinner("Thinking with project context…"):
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
