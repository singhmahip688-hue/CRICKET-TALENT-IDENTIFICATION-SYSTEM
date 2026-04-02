import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- UI STYLE (FIXED FOR VISIBILITY) ----------------
st.set_page_config(page_title="AI Cricket Talent System", layout="wide")

st.markdown("""
<style>
    /* Light Background Gradient */
    .stApp {
        background: linear-gradient(to bottom, #f0f2f6, #ffffff);
    }
    
    /* Title and Header Colors - Deep Navy for visibility */
    h1, h2, h3, .stMarkdown {
        color: #1e3a8a !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Subheader specific fix */
    .st-emotion-cache-10trblm {
        color: #1e3a8a;
    }

    /* Making all standard text dark gray/black */
    p, span, label {
        color: #333333 !important;
    }

    /* Metric Box Styling - Light mode */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* File Uploader styling */
    .stFileUploader {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 AI Cricket Talent Identification System")
st.markdown("### 🚀 Final Year AI + ML + Deep Learning Project")

# ---------------- NORMALIZATION ----------------
def normalize(series, invert=False):
    series = pd.to_numeric(series, errors='coerce').fillna(0)

    if series.max() == series.min():
        return pd.Series([0]*len(series))

    norm = (series - series.min()) / (series.max() - series.min())
    return 1 - norm if invert else norm

# ---------------- CNN MODEL ----------------
@st.cache_resource
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

cnn_model = build_cnn()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Wkts": "Wickets",
        "SR": "Strike_Rate",
        "4s": "Fours",
        "6s": "Sixes"
    }, inplace=True)

    df = df.replace("-", 0).fillna(0)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- TYPE DETECTION ----------------
    is_bowling = "Wickets" in df.columns and "Runs" not in df.columns
    is_batting = "Runs" in df.columns and "Wickets" not in df.columns

    # ---------------- BATTING ----------------
    batting_scores = []
    if not is_bowling:
        if "Runs" in df.columns:
            batting_scores.append(normalize(df["Runs"]))

        if "Strike_Rate" in df.columns:
            sr = pd.to_numeric(df["Strike_Rate"], errors='coerce').fillna(0)
            sr_filtered = sr.where(sr >= 100, 0)
            if sr_filtered.sum() > 0:
                batting_scores.append(normalize(sr_filtered))

    df["Batting_Talent"] = np.mean(batting_scores, axis=0)*100 if batting_scores else 0

    # ---------------- BOWLING ----------------
    bowling_scores = []
    if not is_batting:
        if "Wickets" in df.columns:
            bowling_scores.append(normalize(df["Wickets"]))

        if "Econ" in df.columns:
            bowling_scores.append(normalize(df["Econ"], invert=True))

    df["Bowling_Talent"] = np.mean(bowling_scores, axis=0)*100 if bowling_scores else 0

    # ---------------- FINAL SCORE ----------------
    def compute(row):
        bat, bowl = row["Batting_Talent"], row["Bowling_Talent"]

        if bat == 0:
            return bowl
        elif bowl == 0:
            return bat
        else:
            return 0.5*bat + 0.5*bowl

    df["Talent_Score"] = df.apply(compute, axis=1)

    # ---------------- ML MODEL ----------------
    features = []
    for col in ["Runs","Strike_Rate","Wickets","Econ"]:
        if col in df.columns:
            features.append(col)

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df["Talent_Score"]

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)

    df["ML_Score"] = model.predict(X)

    # ==============================
    # 📊 MODEL EVALUATION
    # ==============================
    st.subheader("📊 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        r2 = r2_score(y, df["ML_Score"])
        st.metric("R² Score", round(r2,3))

    with col2:
        mae = mean_absolute_error(y, df["ML_Score"])
        st.metric("MAE Error", round(mae,3))

    # ==============================
    # 🏆 TOP PLAYERS
    # ==============================
    st.subheader("🏆 Top Players")
    top_df = df.sort_values("ML_Score", ascending=False).head(10)
    st.dataframe(top_df, use_container_width=True)

    # ==============================
    # 📊 FULL DATASET
    # ==============================
    st.subheader("📋 Full Dataset (All Players)")

    sort_option = st.selectbox(
        "Sort Full Dataset By",
        ["ML_Score", "Talent_Score", "Batting_Talent", "Bowling_Talent"]
    )

    full_df = df.sort_values(sort_option, ascending=False)

    st.dataframe(full_df, use_container_width=True, height=400)

    csv = full_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Dataset", csv, "full_player_data.csv")

    # ==============================
    # 📊 GRAPH
    # ==============================
    st.subheader("📊 Top Players Graph")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(top_df["Player"], top_df["ML_Score"], color='#1e3a8a') # Navy bars
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # ==============================
    # ⚔ PLAYER COMPARISON
    # ==============================
    st.subheader("⚔ Player Comparison")

    p1 = st.selectbox("Player 1", df["Player"].unique())
    p2 = st.selectbox("Player 2", df["Player"].unique())

    row1 = df[df["Player"] == p1].iloc[0]
    row2 = df[df["Player"] == p2].iloc[0]

    categories = ["Batting", "Bowling", "ML Score"]

    val1 = [row1["Batting_Talent"], row1["Bowling_Talent"], row1["ML_Score"]]
    val2 = [row2["Batting_Talent"], row2["Bowling_Talent"], row2["ML_Score"]]

    x = np.arange(len(categories))
    width = 0.3

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.bar(x - width/2, val1, width, label=p1, color='#1e3a8a')
    ax2.bar(x + width/2, val2, width, label=p2, color='#3b82f6')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig2)

    # ==============================
    # 📊 PLAYER BREAKDOWN
    # ==============================
    st.subheader("📊 Player Breakdown")

    selected = st.selectbox("Select Player", df["Player"].unique())
    row = df[df["Player"] == selected].iloc[0]

    fig3, ax3 = plt.subplots(figsize=(5,3))
    ax3.bar(["Batting","Bowling"], [row["Batting_Talent"], row["Bowling_Talent"]], color=['#1e3a8a', '#3b82f6'])
    plt.tight_layout()
    st.pyplot(fig3)

# ==============================
# 📸 IMAGE SECTION
# ==============================
st.subheader("📸 Image Talent")

image_file = st.file_uploader("Upload Image", type=["jpg","png"])

def preprocess(img):
    image = Image.open(img).convert("RGB")
    image = image.resize((64,64))
    image = np.array(image)/255.0
    return np.expand_dims(image, axis=0)

if image_file:
    img = preprocess(image_file)
    score = cnn_model.predict(img)[0][0]
    st.success(f"🎯 Image Talent Score: {round(score,2)}")