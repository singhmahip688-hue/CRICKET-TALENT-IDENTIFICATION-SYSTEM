import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import cv2

# Import modern MediaPipe Tasks modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Set up global page config
st.set_page_config(page_title="Cricket Talent Analytics", layout="wide")
st.title("🏏 Cricket Talent & Pose Analytics Dashboard")

# ----------------------------------------------------------------
# HELPER FUNCTIONS & ANGLE CALCULATIONS
# ----------------------------------------------------------------
def normalize(series, invert=False):
    """Safely normalizes numeric pandas series to a 0-1 scale."""
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    if series.max() == series.min():
        return pd.Series([0.0] * len(series))
    
    norm = (series - series.min()) / (series.max() - series.min())
    return 1.0 - norm if invert else norm

def calculate_angle(a, b, c):
    """Calculates the angle between three points (dictionaries with x,y)."""
    a_vec = np.array([a['x'], a['y']])
    b_vec = np.array([b['x'], b['y']])  # Vertex
    c_vec = np.array([c['x'], c['y']])
    
    radians = np.arctan2(c_vec[1]-b_vec[1], c_vec[0]-b_vec[0]) - np.arctan2(a_vec[1]-b_vec[1], a_vec[0]-b_vec[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# ----------------------------------------------------------------
# TAB 1: CSV DATA TALENT ANALYTICS
# ----------------------------------------------------------------
st.header("📊 Tabular Performance Analytics")
uploaded_file = st.file_uploader("📂 Upload Performance CSV Dataset", type=["csv"], key="csv_mesh")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean and Standardize Columns
    df.columns = df.columns.str.strip().str.lower()
    column_mapping = {
        "player": "Player",
        "runs": "Runs",
        "wkts": "Wickets",
        "wickets": "Wickets",
        "sr": "Strike_Rate",
        "strike_rate": "Strike_Rate",
        "strike rate": "Strike_Rate",
        "econ": "Econ",
        "economy": "Econ",
        "4s": "Fours",
        "6s": "Sixes"
    }
    df.rename(columns=column_mapping, inplace=True)
    df = df.replace("-", 0).fillna(0)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Segmentation Logic
    is_bowling = "Wickets" in df.columns and "Runs" not in df.columns
    is_batting = "Runs" in df.columns and "Wickets" not in df.columns

    # Process Batting Matrix
    batting_scores = []
    if not is_bowling:
        if "Runs" in df.columns:
            batting_scores.append(normalize(df["Runs"]))
        if "Strike_Rate" in df.columns:
            sr = pd.to_numeric(df["Strike_Rate"], errors='coerce').fillna(0)
            sr_filtered = sr.where(sr >= 100, 0)
            if sr_filtered.sum() > 0:
                batting_scores.append(normalize(sr_filtered))

    df["Batting_Talent"] = np.mean(batting_scores, axis=0) * 100 if batting_scores else 0.0

    # Process Bowling Matrix
    bowling_scores = []
    if not is_batting:
        if "Wickets" in df.columns:
            bowling_scores.append(normalize(df["Wickets"]))
        if "Econ" in df.columns:
            bowling_scores.append(normalize(df["Econ"], invert=True))

    df["Bowling_Talent"] = np.mean(bowling_scores, axis=0) * 100 if bowling_scores else 0.0

    # Composite Score Derivation
    def compute_talent(row):
        bat = row["Batting_Talent"]
        bowl = row["Bowling_Talent"]
        if bat == 0: return bowl
        if bowl == 0: return bat
        return 0.5 * bat + 0.5 * bowl

    df["Talent_Score"] = df.apply(compute_talent, axis=1)

    # Machine Learning Regression Engine
    features = [col for col in ["Runs", "Strike_Rate", "Wickets", "Econ"] if col in df.columns]
    
    if features:
        X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df["Talent_Score"]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        df["ML_Score"] = model.predict(X)

        # Model Evaluation Metrics
        st.subheader("📈 Model Performance Metrics")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Model R² Score", round(r2_score(y, df["ML_Score"]), 3))
        with m_col2:
            st.metric("Mean Absolute Error (MAE)", round(mean_absolute_error(y, df["ML_Score"]), 3))

        # Leaderboard
        st.subheader("🏆 Predictive Performance Leaderboard (Top 10)")
        top_df = df.sort_values("ML_Score", ascending=False).head(10)
        st.dataframe(top_df, use_container_width=True)

        # Interactive Explorer
        st.subheader("📋 Complete Scout Database")
        sort_option = st.selectbox(
            "Prioritize Column Rank By:",
            ["ML_Score", "Talent_Score", "Batting_Talent", "Bowling_Talent"]
        )
        full_df = df.sort_values(sort_option, ascending=False)
        st.dataframe(full_df, use_container_width=True, height=300)

        # Export Capability
        csv_data = full_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Analytics Report (.CSV)", csv_data, "cricket_talent_report.csv", "text/csv")

        # Visual Plot Implementations
        st.subheader("📊 Visual Distribution Analytics")
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.markdown("**Top 10 Roster Talent Distribution**")
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.bar(top_df["Player"], top_df["ML_Score"], color='#1e3a8a')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with g_col2:
            st.markdown("**Dynamic Player Head-to-Head Comparison**")
            p1 = st.selectbox("Select Athlete 1", df["Player"].unique(), index=0)
            p2 = st.selectbox("Select Athlete 2", df["Player"].unique(), index=min(1, len(df["Player"].unique())-1))
            
            row1 = df[df["Player"] == p1].iloc[0]
            row2 = df[df["Player"] == p2].iloc[0]
            
            categories = ["Batting", "Bowling", "ML Score"]
            val1 = [row1["Batting_Talent"], row1["Bowling_Talent"], row1["ML_Score"]]
            val2 = [row2["Batting_Talent"], row2["Bowling_Talent"], row2["ML_Score"]]
            
            x_axis = np.arange(len(categories))
            width = 0.35
            
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            ax2.bar(x_axis - width/2, val1, width, label=p1, color='#1e3a8a')
            ax2.bar(x_axis + width/2, val2, width, label=p2, color='#3b82f6')
            ax2.set_xticks(x_axis)
            ax2.set_xticklabels(categories)
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            
    else:
        st.error("The uploaded CSV file does not contain mandatory statistical parameters.")

# ----------------------------------------------------------------
# TAB 2: BIOMECHANICAL POSE TALENT SCOUT (MODERN TASKS API)
# ----------------------------------------------------------------
st.markdown("---")
st.header("📸 Biomechanical Posture & Mechanics Identification")
st.markdown("Upload a dynamic cricket action photograph to map skeletal angles and analyze current posture setup mechanics.")

image_file = st.file_uploader("Upload Player Biomechanics Profile Image", type=["jpg", "png", "jpeg"], key="img_mesh")

if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = cv_img.shape
    
    temp_img_path = "temp_scout_frame.jpg"
    cv2.imwrite(temp_img_path, cv_img)
    
    mp_image = mp.Image.create_from_file(temp_img_path)
    
    try:
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP = 11, 13, 15, 23
                
                shoulder = {'x': landmarks[LEFT_SHOULDER].x, 'y': landmarks[LEFT_SHOULDER].y}
                elbow = {'x': landmarks[LEFT_ELBOW].x, 'y': landmarks[LEFT_ELBOW].y}
                wrist = {'x': landmarks[LEFT_WRIST].x, 'y': landmarks[LEFT_WRIST].y}
                hip = {'x': landmarks[LEFT_HIP].x, 'y': landmarks[LEFT_HIP].y}
                
                trunk_angle = calculate_angle(shoulder, hip, {'x': hip['x'], 'y': 0.0}) 
                arm_extension = calculate_angle(shoulder, elbow, wrist) 
                
                spine_score = max(0, 100 - abs(20 - trunk_angle) * 2)
                extension_score = (arm_extension / 180.0) * 100
                composite_form_score = round((0.4 * spine_score) + (0.6 * extension_score), 2)
                
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(cv_img, (cx, cy), 4, (245, 117, 66), -1)
                
                pt_shoulder = (int(landmarks[LEFT_SHOULDER].x * w), int(landmarks[LEFT_SHOULDER].y * h))
                pt_elbow = (int(landmarks[LEFT_ELBOW].x * w), int(landmarks[LEFT_ELBOW].y * h))
                pt_wrist = (int(landmarks[LEFT_WRIST].x * w), int(landmarks[LEFT_WRIST].y * h))
                pt_hip = (int(landmarks[LEFT_HIP].x * w), int(landmarks[LEFT_HIP].y * h))
                
                cv2.line(cv_img, pt_shoulder, pt_elbow, (245, 66, 230), 2)
                cv2.line(cv_img, pt_elbow, pt_wrist, (245, 66, 230), 2)
                cv2.line(cv_img, pt_shoulder, pt_hip, (246, 82, 59), 2)  
                
                ui_col1, ui_col2 = st.columns([3, 2])
                
                with ui_col1:
                    annotated_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Biomechanical Frame Mapping Analytics", use_container_width=True)
                    
                with ui_col2:
                    st.success("✅ Kinematic Pose Structural Check Complete")
                    st.metric("🎯 Pose Performance Alignment Score", f"{min(composite_form_score, 100.0)} / 100")
                    
                    # --- UPDATED CRICKET SCORING SYSTEM LOGIC ---
                    if composite_form_score >= 70.0:
                        st.success("🏆 **result:** Excellent batsmen")
                    elif composite_form_score >= 60.0:
                        st.info("📈 **result:** Average batsmen")
                    elif composite_form_score >= 50.0:
                        st.warning("⚠️ **result:** Need Improvement in Techniques: Good batsmen")
                    else:
                        st.error("🚨 **result:** Mechanical Adjustments Required")
                    
                    st.markdown("### 🧬 Segment Calibration Data")
                    st.write(f"**Trunk Spine Deviation Tilt Angle:** {round(trunk_angle, 1)}°")
                    st.write(f"**Elbow Position Extension Radius:** {round(arm_extension, 1)}°")
                    
                    st.markdown("### 📋 Automation Coaching Insights")
                    if arm_extension < 130:
                        st.warning("⚠️ **Form Warning:** Marginal arm extension flag. Extend the front elbow through the target plane path during ball release/impact steps.")
                    else:
                        st.info("💎 **Form Match:** Clean technical structure. Player maintains linear extension metrics across target vectors.")
            else:
                st.error("❌ MediaPipe loaded successfully, but could not detect a clear human form outline in the image. Please use a closer, well-lit action shot.")
                
    except Exception as e:
        st.error(f"Failed to load standard pose asset model. Ensure the 'pose_landmarker.task' file is downloaded inside your root folder. Details: {e}")
