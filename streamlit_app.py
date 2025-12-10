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
from datetime import date, time

# ==============================
# BASIC PAGE CONFIG + GLOBAL CSS
# ==============================
st.set_page_config(
    page_title="Path Finders – Banff Parking",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* background */
        .main {
            background: radial-gradient(circle at top left, #fde2ff 0, #f5f7ff 30%, #e2fbff 60%, #fdf6ff 100%);
        }

        /* center container */
        .pf-container {
            max-width: 1050px;
            margin: 0 auto;
            padding: 1.5rem 0 3rem 0;
        }

        /* title */
        .pf-title {
            text-align: center;
            letter-spacing: 0.3em;
            font-size: 2.6rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.5rem;
            margin-bottom: 0.3rem;
        }

        .pf-subtitle {
            text-align: center;
            font-size: 0.98rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }

        /* glass hero card */
        .pf-hero {
            background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
            border-radius: 1.4rem;
            padding: 1.2rem 1.6rem;
            box-shadow: 0 20px 40px rgba(148,163,184,0.28);
            border: 1px solid rgba(148,163,184,0.25);
            text-align: center;
            margin-bottom: 1.6rem;
        }

        /* vertical menu */
        .pf-menu-card {
            background: rgba(255,255,255,0.92);
            border-radius: 1.4rem;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 18px 35px rgba(148,163,184,0.30);
            border: 1px solid rgba(148,163,184,0.25);
        }

        .pf-menu-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 0.6rem;
            text-align: center;
        }

        .pf-menu-btn {
            width: 100%;
            text-align: left;
            padding: 0.55rem 0.9rem;
            margin-bottom: 0.4rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.45);
            background: rgba(248,250,252,0.96);
            cursor: pointer;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #0f172a;
        }

        .pf-menu-btn span.icon {
            font-size: 1.05rem;
        }

        .pf-menu-btn-active {
            background: linear-gradient(90deg, #fee2ff, #e0f2fe);
            border-color: #fb7185;
            color: #111827;
            box-shadow: 0 10px 25px rgba(248,113,113,0.35);
        }

        /* section header inside pages */
        .pf-section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.8rem;
        }
        .pf-section-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }
        .pf-back-btn {
            font-size: 0.86rem;
            padding: 0.34rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.6);
            background: rgba(255,255,255,0.8);
        }

        /* smaller plots */
        .stPlotlyChart, .stAltairChart, .stVegaLiteChart {
            padding: 0.4rem 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============
# OPENAI (safe)
# ===============
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# =========================
# LOAD MODELS + TEST DATA
# =========================
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

# =========================
# CSV FOR DASHBOARD
# =========================
DASHBOARD_CSV = "banff_parking_engineered_HOURLY (1).csv"


@st.cache_data
def load_dashboard_data():
    if os.path.exists(DASHBOARD_CSV):
        df = pd.read_csv(DASHBOARD_CSV)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        # Month & Hour already exist but make sure
        if "Month" not in df.columns:
            df["Month"] = df["Timestamp"].dt.month
        if "Hour" not in df.columns:
            df["Hour"] = df["Timestamp"].dt.hour
        return df
    return None


# ================
# AUTO WEATHER
# ================
SEASONAL_DEFAULTS = {
    1: (-8.0, 2.0, 18.0),
    2: (-6.0, 2.0, 18.0),
    3: (-2.0, 3.0, 20.0),
    4: (4.0, 3.0, 22.0),
    5: (10.0, 4.0, 24.0),
    6: (14.0, 5.0, 26.0),
    7: (18.0, 6.0, 24.0),
    8: (17.0, 5.0, 24.0),
    9: (12.0, 4.0, 24.0),
    10: (6.0, 3.0, 22.0),
    11: (0.0, 3.0, 20.0),
    12: (-7.0, 3.0, 18.0),
}


def get_auto_weather(selected_date: date, selected_time: time, df_dash: pd.DataFrame):
    """Use CSV if possible, otherwise season defaults (handles winter / minus temps)."""
    m = selected_date.month
    h = selected_time.hour

    if df_dash is not None:
        subset = df_dash[(df_dash["Month"] == m) & (df_dash["Hour"] == h)]
        if subset.empty:
            subset = df_dash[df_dash["Month"] == m]
        if not subset.empty:
            return (
                float(subset["Max Temp (°C)"].mean()),
                float(subset["Total Precip (mm)"].mean()),
                float(subset["Spd of Max Gust (km/h)"].mean()),
            )

    # fallback: typical seasonal values
    return SEASONAL_DEFAULTS.get(m, (10.0, 3.0, 20.0))


def get_time_features_from_inputs(selected_date: date, selected_time: time):
    month = selected_date.month
    day_of_week = selected_date.weekday()  # 0=Mon
    hour = selected_time.hour
    is_weekend = 1 if day_of_week in [5, 6] else 0
    return month, day_of_week, hour, is_weekend

# ====================
# RAG / CHAT HELPERS
# ====================
@st.cache_resource
def load_rag_knowledge():
    knowledge_path = "banff_knowledge.txt"
    if not os.path.exists(knowledge_path):
        docs = [
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt "
            "file is missing, so answers are based only on general parking logic."
        ]
    else:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_embeddings = vectorizer.fit_transform(docs)
    return docs, vectorizer, doc_embeddings


def retrieve_context(query, docs, vectorizer, doc_embeddings, k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_embeddings).flatten()
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
            "🚫 Chat is running in offline mode (no OpenAI API key set).\n\n"
            "Here is the most relevant information from your notes:\n\n"
            f"{context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply for classmates and "
                "instructors who are not data scientists. Use the supplied Context as "
                "your main source of truth."
            ),
        },
        {"role": "system", "content": f"Context from project notes:\n{context}"},
    ]

    for h in chat_history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_question})

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return (
            "⚠️ I couldn’t contact the language-model service right now.\n\n"
            "Here is what I can tell from your notes:\n\n"
            f"{context}"
        )

# ====================
# NAVIGATION STATE
# ====================
if "pf_page" not in st.session_state:
    st.session_state.pf_page = "home"


def go(page_name: str):
    st.session_state.pf_page = page_name
    st.rerun()

# ====================
# HOME PAGE
# ====================
def render_home():
    with st.container():
        st.markdown('<div class="pf-container">', unsafe_allow_html=True)

        st.markdown('<div class="pf-title">PATH FINDERS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pf-subtitle">Smart, simple parking insights for Banff – powered by machine learning and pastel-soft XAI.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pf-hero">
                Choose what you want to explore: demand prediction, lot comparisons, 
                explainable AI views, or a friendly project assistant chat.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="pf-menu-card">', unsafe_allow_html=True)
            st.markdown('<div class="pf-menu-title">Open a section</div>', unsafe_allow_html=True)

            if st.button("🎯  Demand prediction", key="home_predict", use_container_width=True):
                go("predict")

            if st.button("📊  Lot overview", key="home_lots", use_container_width=True):
                go("lots")

            if st.button("🔍  XAI views", key="home_xai", use_container_width=True):
                go("xai")

            if st.button("💬  Project assistant", key="home_chat", use_container_width=True):
                go("chat")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown(
                """
                <div class="pf-menu-card" style="min-height: 180px; display:flex; align-items:center; justify-content:center; text-align:center;">
                    <div>
                        <div style="font-size:0.95rem; color:#6b7280; margin-bottom:0.4rem;">
                            Hint
                        </div>
                        <div style="font-size:0.9rem; color:#374151;">
                            Use this screen in your demo to quickly explain what each page does.  
                            Then click a button on the left to jump into a focused view.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ====================
# DEMAND PREDICTION PAGE
# ====================
def render_predict():
    df_dash = load_dashboard_data()

    st.markdown('<div class="pf-container">', unsafe_allow_html=True)
    top_col1, top_col2 = st.columns([4, 1])

    with top_col1:
        st.markdown(
            """
            <div class="pf-section-header">
                <div class="pf-section-title">Demand prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Pick date, time, lot and weather – the model predicts occupancy and full-lot risk.")

    with top_col2:
        if st.button("← Back home", key="back_from_predict"):
            go("home")

    col_left, col_right = st.columns([1.3, 1])

    # --- inputs ---
    with col_left:
        st.markdown("**When?**")
        pred_date = st.date_input("Prediction date", value=date.today(), key="pred_date")
        pred_time = st.time_input("Prediction time", value=time(13, 0), key="pred_time")

    with col_right:
        st.markdown("**Weather**")

        use_auto_weather = st.checkbox(
            "Auto weather from data / season",
            value=True,
            help="Use typical weather for this date & time (from CSV if available, otherwise seasonal defaults).",
            key="predict_auto_weather",
        )

        if use_auto_weather:
            auto_temp, auto_precip, auto_gust = get_auto_weather(pred_date, pred_time, df_dash)
        else:
            auto_temp, auto_precip, auto_gust = 22.0, 0.5, 12.0

        max_temp = st.slider(
            "Max temp (°C)",
            -25.0,
            40.0,
            float(auto_temp),
            key="pred_temp",
        )
        total_precip = st.slider(
            "Total precip (mm)",
            0.0,
            40.0,
            float(auto_precip),
            key="pred_precip",
        )
        wind_gust = st.slider(
            "Max gust (km/h)",
            0.0,
            120.0,
            float(auto_gust),
            key="pred_gust",
        )

    month, day_of_week, hour, is_weekend = get_time_features_from_inputs(pred_date, pred_time)

    # lot selection
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if lot_features:
        pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    st.markdown("---")

    col_l1, col_l2 = st.columns([1.2, 1])

    with col_l1:
        if lot_features:
            selected_lot_label = st.selectbox(
                "Parking lot",
                lot_display_names,
                index=0,
            )
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None
            st.warning("No features starting with 'Unit_' found – lot selection disabled.")

    with col_l2:
        st.info("Click **Predict** to see occupancy and full-lot probability for this scenario.")

    # build feature vector
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
            st.metric("Predicted occupancy (model units)", f"{occ_pred:.2f}")
        with c2:
            st.metric("Full-lot probability", f"{full_prob:.1%}")

        if full_prob > 0.7:
            st.warning("High risk this lot will be full.")
        elif full_prob > 0.4:
            st.info("Medium risk – expect busy conditions.")
        else:
            st.success("Low risk of the lot being full.")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================
# LOTS OVERVIEW PAGE
# ====================
def render_lots():
    df_dash = load_dashboard_data()

    st.markdown('<div class="pf-container">', unsafe_allow_html=True)
    top_col1, top_col2 = st.columns([4, 1])

    with top_col1:
        st.markdown(
            """
            <div class="pf-section-header">
                <div class="pf-section-title">Lot overview</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Compare all lots at one moment – see occupancy and full-lot risk side-by-side.")

    with top_col2:
        if st.button("← Back home", key="back_from_lots"):
            go("home")

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        lots_date = st.date_input("Status date", value=date.today(), key="lots_date")
        lots_time = st.time_input("Status time", value=time(14, 0), key="lots_time")

    with col_right:
        use_auto_weather = st.checkbox(
            "Auto weather from data / season",
            value=True,
            key="lots_auto_weather",
        )

        if use_auto_weather:
            auto_temp, auto_precip, auto_gust = get_auto_weather(lots_date, lots_time, df_dash)
        else:
            auto_temp, auto_precip, auto_gust = 22.0, 0.5, 12.0

        max_temp = st.slider(
            "Max temp (°C)",
            -25.0,
            40.0,
            float(auto_temp),
            key="lots_temp",
        )
        total_precip = st.slider(
            "Total precip (mm)",
            0.0,
            40.0,
            float(auto_precip),
            key="lots_precip",
        )
        wind_gust = st.slider(
            "Max gust (km/h)",
            0.0,
            120.0,
            float(auto_gust),
            key="lots_gust",
        )

    month, day_of_week, hour, is_weekend = get_time_features_from_inputs(lots_date, lots_time)

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if lot_features:
        pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    if not lot_features:
        st.error("No features with prefix 'Unit_' in FEATURES. Cannot show lot overview.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
                return ["background-color: #ffe5e5"] * len(row)
            elif "Busy" in row["Status"]:
                return ["background-color: #fff4e0"] * len(row)
            else:
                return ["background-color: #e0f7f4"] * len(row)

        styled_df = (
            df.style.format(
                {"Predicted occupancy": "{:.2f}", "Probability full": "{:.1%}"}
            ).apply(lot_status_row_style, axis=1)
        )

        st.dataframe(styled_df, use_container_width=True)
        st.caption("Row colour shows risk level: red = high, orange = busy, green = comfortable.")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================
# XAI PAGE
# ====================
def render_xai():
    st.markdown('<div class="pf-container">', unsafe_allow_html=True)
    top_col1, top_col2 = st.columns([4, 1])

    with top_col1:
        st.markdown(
            """
            <div class="pf-section-header">
                <div class="pf-section-title">XAI – explainable AI views</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Look under the hood: SHAP, partial dependence, and residuals for the regression model.")

    with top_col2:
        if st.button("← Back home", key="back_from_xai"):
            go("home")

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
    except Exception as e:
        st.error(f"Could not generate SHAP plots: {e}")

    st.markdown("**Partial dependence plots**")
    pd_feature_names = [name for name in ["Max Temp (°C)", "Month", "Hour"] if name in FEATURES]
    if pd_feature_names:
        idx = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(7, 3))
        PartialDependenceDisplay.from_estimator(
            best_xgb_reg,
            X_test_scaled,
            idx,
            feature_names=FEATURES,
            ax=ax3,
        )
        st.pyplot(fig3)
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
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================
# CHAT PAGE
# ====================
def render_chat():
    st.markdown('<div class="pf-container">', unsafe_allow_html=True)
    top_col1, top_col2 = st.columns([4, 1])

    with top_col1:
        st.markdown(
            """
            <div class="pf-section-header">
                <div class="pf-section-title">Project assistant chat</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Ask questions about patterns, busy times, or how the model works.")

    with top_col2:
        if st.button("← Back home", key="back_from_chat"):
            go("home")

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
                    user_input, st.session_state.rag_chat_history
                )
                st.markdown(answer)

        st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})

    st.caption(
        "Edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ====================
# ROUTER
# ====================
page = st.session_state.pf_page

if page == "home":
    render_home()
elif page == "predict":
    render_predict()
elif page == "lots":
    render_lots()
elif page == "xai":
    render_xai()
elif page == "chat":
    render_chat()
