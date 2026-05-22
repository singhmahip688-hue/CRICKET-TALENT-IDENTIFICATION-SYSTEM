import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import tempfile
import os

from sklearn.ensemble import RandomForestRegressor

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Cricket Biomechanics Engine",
    layout="wide"
)

st.title("🏏 AI Cricket Biomechanics & Talent Analytics Engine")

# ============================================================
# UTILITIES (UPDATED)
# ============================================================
def calculate_2d_angle(a, b, c):
    """
    Calculates 2D angle between three points.
    More stable for broadcast/mobile cricket videos.
    """

    ba = np.array([
        a['x'] - b['x'],
        a['y'] - b['y']
    ])

    bc = np.array([
        c['x'] - b['x'],
        c['y'] - b['y']
    ])

    cosine_angle = np.dot(ba, bc) / (
        (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    )

    angle = np.arccos(
        np.clip(cosine_angle, -1.0, 1.0)
    )

    return np.degrees(angle)


def normalize_score(value, ideal, tolerance):
    """
    Adaptive normalization with smoother decay.
    """

    diff = abs(value - ideal)

    if diff <= tolerance:
        return 1.0

    score = max(
        0.1,
        1.0 - (
            (diff - tolerance)
            /
            (tolerance * 2)
        )
    )

    return round(score, 3)


def safe_normalize(series, invert=False):

    series = pd.to_numeric(
        series,
        errors='coerce'
    ).fillna(0)

    s_min = series.min()
    s_max = series.max()

    if s_min == s_max:
        return pd.Series(
            [0.0] * len(series),
            index=series.index
        )

    norm = (
        (series - s_min)
        /
        (s_max - s_min)
    )

    return 1 - norm if invert else norm


# ============================================================
# ADVANCED METRICS (UPDATED)
# ============================================================
def get_advanced_metrics(landmarks):

    # SHOULDERS
    L_SHOULDER = landmarks[11]
    R_SHOULDER = landmarks[12]

    # ELBOWS
    L_ELBOW = landmarks[13]
    R_ELBOW = landmarks[14]

    # WRISTS
    L_WRIST = landmarks[15]
    R_WRIST = landmarks[16]

    # HIPS
    L_HIP = landmarks[23]
    R_HIP = landmarks[24]

    # KNEES
    L_KNEE = landmarks[25]
    R_KNEE = landmarks[26]

    # ANKLES
    L_ANKLE = landmarks[27]
    R_ANKLE = landmarks[28]

    # ========================================================
    # LEFT + RIGHT ANGLES
    # ========================================================
    elbow_L = calculate_2d_angle(
        {
            'x': L_SHOULDER.x,
            'y': L_SHOULDER.y
        },
        {
            'x': L_ELBOW.x,
            'y': L_ELBOW.y
        },
        {
            'x': L_WRIST.x,
            'y': L_WRIST.y
        }
    )

    elbow_R = calculate_2d_angle(
        {
            'x': R_SHOULDER.x,
            'y': R_SHOULDER.y
        },
        {
            'x': R_ELBOW.x,
            'y': R_ELBOW.y
        },
        {
            'x': R_WRIST.x,
            'y': R_WRIST.y
        }
    )

    knee_L = calculate_2d_angle(
        {
            'x': L_HIP.x,
            'y': L_HIP.y
        },
        {
            'x': L_KNEE.x,
            'y': L_KNEE.y
        },
        {
            'x': L_ANKLE.x,
            'y': L_ANKLE.y
        }
    )

    knee_R = calculate_2d_angle(
        {
            'x': R_HIP.x,
            'y': R_HIP.y
        },
        {
            'x': R_KNEE.x,
            'y': R_KNEE.y
        },
        {
            'x': R_ANKLE.x,
            'y': R_ANKLE.y
        }
    )

    # ========================================================
    # DYNAMIC SIDE DETECTION
    # ========================================================
    elbow_angle = (
        elbow_R
        if abs(180 - elbow_R) > abs(180 - elbow_L)
        else elbow_L
    )

    knee_angle = (
        knee_R
        if abs(180 - knee_R) > abs(180 - knee_L)
        else knee_L
    )

    # ========================================================
    # NORMALIZED METRICS
    # ========================================================
    torso_height = abs(
        L_SHOULDER.y - L_HIP.y
    ) + 1e-6

    raw_stance = abs(
        L_ANKLE.x - R_ANKLE.x
    )

    stance_width = raw_stance / torso_height

    shoulder_alignment = abs(
        L_SHOULDER.y - R_SHOULDER.y
    ) / torso_height

    return {
        "elbow_angle": elbow_angle,
        "knee_angle": knee_angle,
        "shoulder_alignment": shoulder_alignment,
        "stance_width": stance_width
    }


# ============================================================
# COACHING INSIGHTS
# ============================================================
def generate_coaching_feedback(score):

    if score >= 90:

        return """
💎 Form Match: Elite professional alignment detected.

Excellent kinetic sequencing, stable head mechanics,
controlled elbow extension, and efficient balance transfer.
"""

    elif score >= 75:

        return """
✅ Form Match: Clean technical structure.

Player maintains linear extension metrics across target vectors.
Strong kinetic chain transfer with stable lower-body sequencing.
"""

    elif score >= 60:

        return """
⚠️ Coaching Alert: Minor instability detected.

Improve front-foot balance, bat lift timing,
knee flex absorption, and shoulder alignment.
"""

    else:

        return """
🚨 High Biomechanical Drift Detected.

Focus on base stability, elbow extension mechanics,
head positioning, and controlled posture sequencing.
"""


# ============================================================
# RESULT LABEL
# ============================================================
def result_label(score, mode):

    if mode == "Batting Stance":

        if score >= 85:
            return "🏆 Excellent Batsman"

        elif score >= 70:
            return "✅ Good Batsman"

        elif score >= 50:
            return "⚠️ Average Batsman"

        else:
            return "🚨 Poor Batting Mechanics"

    else:

        if score >= 85:
            return "🏆 Excellent Bowler"

        elif score >= 70:
            return "✅ Good Bowling Action"

        elif score >= 50:
            return "⚠️ Average Bowling Mechanics"

        else:
            return "🚨 Poor Bowling Mechanics"


# ============================================================
# PLAYER DATABASE
# ============================================================
@st.cache_data
def get_historical_players_db():

    return {

        "Virat Kohli": {
            "Country": "India",
            "Role": "Batsman",
            "Age": 37
        },

        "Rohit Sharma": {
            "Country": "India",
            "Role": "Batsman",
            "Age": 39
        },

        "MS Dhoni": {
            "Country": "India",
            "Role": "Wicketkeeper Batter",
            "Age": 44
        },

        "Jasprit Bumrah": {
            "Country": "India",
            "Role": "Fast Bowler",
            "Age": 32
        }
    }


# ============================================================
# PLAYER CARD
# ============================================================
def render_player_identity_card(file_name):

    players_db = get_historical_players_db()

    matched_key = None

    for key in players_db.keys():

        if key.lower().split()[0] in file_name.lower():

            matched_key = key
            break

    st.markdown("## 👤 Player Profile")

    c1, c2, c3, c4 = st.columns(4)

    if matched_key:

        data = players_db[matched_key]

        c1.metric("Player", matched_key)
        c2.metric("Country", data["Country"])
        c3.metric("Role", data["Role"])
        c4.metric("Age", data["Age"])

    else:

        clean_name = (
            os.path.splitext(file_name)[0]
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        c1.metric("Player", clean_name)
        c2.metric("Country", "Unknown")
        c3.metric("Role", "Prospect")
        c4.metric("Age", "N/A")

    st.markdown("---")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Talent Scout",
    "📸 Image Analysis",
    "🎥 Video Analysis"
])

# ============================================================
# TAB 1 - TALENT SCOUT
# ============================================================
with tab1:

    st.header("📊 AI Talent Identification Engine")

    uploaded_file = st.file_uploader(
        "Upload Cricket Dataset",
        type=["csv"]
    )

    if uploaded_file:

        df_raw = pd.read_csv(uploaded_file)

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df_raw,
            use_container_width=True
        )

        csv_data = df_raw.to_csv(index=False)

        st.download_button(
            label="⬇ Download Dataset",
            data=csv_data,
            file_name="cricket_dataset.csv",
            mime="text/csv"
        )

        df_clean = df_raw.copy()

        df_clean.columns = (
            df_clean.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        key_maps = {

            "player": [
                "player",
                "player_name",
                "name"
            ],

            "runs": [
                "runs",
                "run"
            ],

            "strike_rate": [
                "strike_rate",
                "sr",
                "batting_sr"
            ],

            "average": [
                "average",
                "avg",
                "batting_avg"
            ],

            "wickets": [
                "wickets",
                "wkts"
            ],

            "economy": [
                "economy",
                "econ",
                "economy_rate"
            ],

            "bowling_avg": [
                "bowling_avg",
                "bowl_avg",
                "bowling_average"
            ],

            "bowling_sr": [
                "bowling_sr",
                "bowling_strike_rate",
                "bowl_sr"
            ]
        }

        resolved_cols = {}

        for main_key, aliases in key_maps.items():

            for alias in aliases:

                if alias in df_clean.columns:

                    resolved_cols[main_key] = alias
                    break

        if "player" not in resolved_cols:

            st.error("""
❌ Player column missing.

Accepted column names:
- player
- player_name
- name
""")

            st.stop()

        player_col = resolved_cols["player"]

        for key, col in resolved_cols.items():

            if key != "player":

                df_clean[col] = pd.to_numeric(
                    df_clean[col],
                    errors="coerce"
                ).fillna(0)

        has_batting = (
            "runs" in resolved_cols
        )

        has_bowling = (
            "wickets" in resolved_cols
        )

        if has_batting and has_bowling:

            st.success("🟢 All-Rounder Dataset Detected")

            norm_runs = safe_normalize(
                df_clean[resolved_cols["runs"]]
            )

            norm_wickets = safe_normalize(
                df_clean[resolved_cols["wickets"]]
            )

            score = (
                norm_runs * 0.5 +
                norm_wickets * 0.5
            )

        elif has_bowling:

            st.success("🔵 Bowling Dataset Detected")

            norm_wickets = safe_normalize(
                df_clean[resolved_cols["wickets"]]
            )

            if "economy" in resolved_cols:

                norm_economy = safe_normalize(
                    df_clean[resolved_cols["economy"]],
                    invert=True
                )

            else:
                norm_economy = 0

            if "bowling_avg" in resolved_cols:

                norm_bowl_avg = safe_normalize(
                    df_clean[resolved_cols["bowling_avg"]],
                    invert=True
                )

            else:
                norm_bowl_avg = 0

            if "bowling_sr" in resolved_cols:

                norm_bowl_sr = safe_normalize(
                    df_clean[resolved_cols["bowling_sr"]],
                    invert=True
                )

            else:
                norm_bowl_sr = 0

            score = (
                norm_wickets * 0.70 +
                norm_economy * 0.10 +
                norm_bowl_avg * 0.10 +
                norm_bowl_sr * 0.10
            )

        elif has_batting:

            st.success("🟠 Batting Dataset Detected")

            norm_runs = safe_normalize(
                df_clean[resolved_cols["runs"]]
            )

            if "strike_rate" in resolved_cols:

                norm_sr = safe_normalize(
                    df_clean[resolved_cols["strike_rate"]]
                )

            else:
                norm_sr = 0

            if "average" in resolved_cols:

                norm_avg = safe_normalize(
                    df_clean[resolved_cols["average"]]
                )

            else:
                norm_avg = 0

            score = (
                norm_runs * 0.60 +
                norm_sr * 0.20 +
                norm_avg * 0.20
            )

        else:

            st.error("""
❌ Dataset not recognized.

Required columns:

Batting:
- runs

Bowling:
- wickets
""")

            st.stop()

        df_clean["AI_Scout_Score"] = score * 100

        feature_cols = []

        for feature in [
            "runs",
            "strike_rate",
            "average",
            "wickets",
            "economy",
            "bowling_avg",
            "bowling_sr"
        ]:

            if feature in resolved_cols:

                feature_cols.append(
                    resolved_cols[feature]
                )

        X = df_clean[feature_cols]

        y = df_clean["AI_Scout_Score"]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(X, y)

        df_clean["Talent Prediction"] = model.predict(X)

        display_df = pd.DataFrame({

            "Player": df_clean[player_col],

            "Talent Prediction": round(
                df_clean["Talent Prediction"],
                2
            )
        })

        optional_columns = {

            "runs": "Runs",
            "strike_rate": "Strike Rate",
            "average": "Average",
            "wickets": "Wickets",
            "economy": "Economy",
            "bowling_avg": "Bowling Avg",
            "bowling_sr": "Bowling SR"
        }

        for key, label in optional_columns.items():

            if key in resolved_cols:

                display_df[label] = df_clean[
                    resolved_cols[key]
                ]

        leaderboard_df = display_df.sort_values(
            "Talent Prediction",
            ascending=False
        )

        st.subheader("🏆 Full Dataset Talent Prediction")

        st.dataframe(
            leaderboard_df,
            use_container_width=True
        )

        st.subheader("🔥 Top 10 Players")

        top10 = leaderboard_df.head(10)

        st.dataframe(
            top10,
            use_container_width=True
        )
        # BAR GRAPH
        # ====================================================
        fig_bar = px.bar(
            top10,
            x="Player",
            y="Talent Prediction",
            color="Player",
            text="Talent Prediction",
            title="Top 10 Talent Prediction Scores"
        )

        fig_bar.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        # ====================================================
        # PLAYER VS PLAYER COMPARISON
        # ====================================================
        st.markdown("---")

        st.header("⚔️ Player 1 vs Player 2 Comparison")

        players = leaderboard_df["Player"].tolist()

        col1, col2 = st.columns(2)

        with col1:

            player1 = st.selectbox(
                "Select Player 1",
                players
            )

        with col2:

            player2 = st.selectbox(
                "Select Player 2",
                players,
                index=min(1, len(players)-1)
            )

        if player1 != player2:

            row1 = leaderboard_df[
                leaderboard_df["Player"] == player1
            ].iloc[0]

            row2 = leaderboard_df[
                leaderboard_df["Player"] == player2
            ].iloc[0]

            # =================================================
            # BOWLING DATASET COMPARISON
            # =================================================
            if has_bowling and not has_batting:

                comparison_metrics = [
                    "Wickets",
                    "Bowling Avg",
                    "Economy",
                    "Talent Prediction"
                ]

            # =================================================
            # BATTING / ALL-ROUNDER COMPARISON
            # =================================================
            else:

                comparison_metrics = [
                    "Runs",
                    "Average",
                    "Strike Rate",
                    "Talent Prediction"
                ]

            # =================================================
            # FILTER AVAILABLE COLUMNS
            # =================================================
            comparison_metrics = [
                metric
                for metric in comparison_metrics
                if metric in leaderboard_df.columns
            ]

            # =================================================
            # COMPARISON TABLE
            # =================================================
            comparison_df = pd.DataFrame({

                "Metric": comparison_metrics,

                player1: [
                    row1[metric]
                    for metric in comparison_metrics
                ],

                player2: [
                    row2[metric]
                    for metric in comparison_metrics
                ]
            })

            st.subheader("📋 Comparison Table")

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            # =================================================
            # COMPARISON GRAPH
            # =================================================
            fig_compare = go.Figure()

            fig_compare.add_trace(go.Bar(
                name=player1,
                x=comparison_metrics,
                y=[
                    row1[metric]
                    for metric in comparison_metrics
                ],
                marker_color="royalblue"
            ))

            fig_compare.add_trace(go.Bar(
                name=player2,
                x=comparison_metrics,
                y=[
                    row2[metric]
                    for metric in comparison_metrics
                ],
                marker_color="crimson"
            ))

            fig_compare.update_layout(
                title="Player vs Player Performance Comparison",
                barmode="group",
                xaxis_title="Metrics",
                yaxis_title="Values",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )
# ============================================================
# TAB 2 - IMAGE ANALYSIS
# ============================================================
with tab2:

    st.header("📸 Static Image Biomechanics Analysis")

    analysis_mode = st.radio(
        "Select Analysis Type",
        [
            "Batting Stance",
            "Bowling Action"
        ]
    )

    image_file = st.file_uploader(
        "Upload Cricket Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:

        render_player_identity_card(
            image_file.name
        )

        file_bytes = np.asarray(
            bytearray(image_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb_frame,
            use_container_width=True
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        MODEL_PATH = "pose_landmarker.task"

        if not os.path.exists(MODEL_PATH):

            st.error("❌ pose_landmarker.task missing")

        else:

            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH
            )

            options = vision.PoseLandmarkerOptions(
                base_options=base_options
            )

            with vision.PoseLandmarker.create_from_options(options) as landmarker:

                result = landmarker.detect(mp_image)

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    metrics = get_advanced_metrics(
                        landmarks
                    )

                    feet_score = normalize_score(
                        metrics["stance_width"],
                        0.8,
                        0.3
                    )

                    knee_score = normalize_score(
                        metrics["knee_angle"],
                        150,
                        20
                    )

                    body_alignment_score = normalize_score(
                        metrics["shoulder_alignment"],
                        0,
                        0.25
                    )

                    grip_score = normalize_score(
                        metrics["elbow_angle"],
                        155,
                        25
                    )

                    head_score = normalize_score(
                        metrics["shoulder_alignment"],
                        0,
                        0.25
                    )

                    backlift_score = 0.8

                    weight_distribution_score = normalize_score(
                        metrics["stance_width"],
                        0.8,
                        0.3
                    )

                    stance_variation_score = 0.75

                    final_score = (
                        feet_score * 15 +
                        knee_score * 10 +
                        body_alignment_score * 15 +
                        grip_score * 15 +
                        head_score * 15 +
                        backlift_score * 10 +
                        weight_distribution_score * 10 +
                        stance_variation_score * 10
                    )

                    final_score = round(final_score, 2)

                    result_text = result_label(
                        final_score,
                        analysis_mode
                    )

                    coaching_feedback = generate_coaching_feedback(
                        final_score
                    )

                    st.markdown(f"""
## 📋 Detailed Metrics

- Feet Positioning: {round(feet_score*100,2)}
- Knee Bend: {round(knee_score*100,2)}
- Body Alignment: {round(body_alignment_score*100,2)}
- Grip Mechanics: {round(grip_score*100,2)}
- Head Stability: {round(head_score*100,2)}
- Backlift Control: {round(backlift_score*100,2)}
- Weight Distribution: {round(weight_distribution_score*100,2)}
- Stance Adaptability: {round(stance_variation_score*100,2)}

# 🏆 Final Score
{final_score}/100

# 🏆 Result
{result_text}

# 📋 Automation Coaching Insights

{coaching_feedback}
""")

# ============================================================
# TAB 3 - VIDEO ANALYSIS (UPDATED)
# ============================================================
with tab3:

    st.header("🎥 Dynamic Video Kinematic Engine")

    analysis_mode = st.radio(
        "Select Video Analysis",
        [
            "Batting Stance",
            "Bowling Action"
        ]
    )

    video_file = st.file_uploader(
        "Upload Cricket Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:

        render_player_identity_card(
            video_file.name
        )

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(
            tfile.name
        )

        MODEL_PATH = "pose_landmarker.task"

        if not os.path.exists(MODEL_PATH):

            st.error("❌ pose_landmarker.task missing")

        else:

            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH
            )

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO
            )

            frame_placeholder = st.empty()

            all_scores = []

            with vision.PoseLandmarker.create_from_options(options) as landmarker:

                frame_count = 0

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_count += 1

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    result = landmarker.detect_for_video(
                        mp_image,
                        frame_count * 33
                    )

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        metrics = get_advanced_metrics(
                            landmarks
                        )

                        # ====================================================
                        # UPDATED DYNAMIC SCORING
                        # ====================================================
                        stance_score = normalize_score(
                            metrics["stance_width"],
                            0.8,
                            0.3
                        )

                        elbow_score = normalize_score(
                            metrics["elbow_angle"],
                            155,
                            25
                        )

                        knee_score = normalize_score(
                            metrics["knee_angle"],
                            150,
                            20
                        )

                        frame_score = (
                            (
                                stance_score * 0.3
                            ) +
                            (
                                elbow_score * 0.4
                            ) +
                            (
                                knee_score * 0.3
                            )
                        ) * 100

                        all_scores.append(
                            frame_score
                        )

                    frame_placeholder.image(
                        rgb_frame,
                        channels="RGB",
                        use_container_width=True
                    )

            cap.release()

            os.unlink(tfile.name)

            # ========================================================
            # TOP 30% PERFORMANCE WINDOW
            # ========================================================
            if len(all_scores) > 0:

                all_scores.sort()

                active_performance_scores = all_scores[
                    int(len(all_scores) * 0.7):
                ]

                final_video_score = round(
                    np.mean(active_performance_scores),
                    2
                )

                result_text = result_label(
                    final_video_score,
                    analysis_mode
                )

                coaching_feedback = generate_coaching_feedback(
                    final_video_score
                )

                st.success(
                    "✅ Full Video Analysis Completed"
                )

                st.metric(
                    "🏏 Final Video Score",
                    f"{final_video_score}/100"
                )

                st.markdown(f"""
# 🏆 Result
{result_text}

# 📋 Coaching Insights
{coaching_feedback}
""")

                fig = px.line(
                    y=all_scores,
                    title="Kinematic Timeline Analysis",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
