import streamlit as st
import joblib
import pickle
import numpy as np
import plotly.graph_objects as go
import math
import os
import base64

# --- Load model & scaler ---
regressor = joblib.load('iplmodel_ridge.sav')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Page config ---
st.set_page_config(page_title="IPL Score Predictor", page_icon="🏏", layout="centered")

banner_path ="Images/ipl_banner.jpg"
if os.path.isfile(banner_path):
    st.image(banner_path, use_container_width=True)
else:
    st.warning("IPL banner not found. Please check the image path.")

st.title("🏏 IPL Score Predictor")

# Team logos dict with user's provided absolute paths
team_logos = {
    'Chennai Super Kings': 'Images/csk.png',
    'Delhi Daredevils': 'Images/dd.png',
    'Kings XI Punjab': 'Images/kxip.png',
    'Kolkata Knight Riders': 'Images/kkr.png',
    'Mumbai Indians': 'Images/mi.jpg',
    'Rajasthan Royals': 'Images/rr.png',
    'Royal Challengers Bangalore': 'Images/rcb.png',
    'Sunrisers Hyderabad': 'Images/srh.png'
}

venue_options = [
    'ACA-VDCA Stadium, Visakhapatnam', 'Barabati Stadium, Cuttack',
    'Dr DY Patil Sports Academy, Mumbai', 'Dubai International Cricket Stadium, Dubai',
    'Eden Gardens, Kolkata', 'Feroz Shah Kotla, Delhi',
    'Himachal Pradesh Cricket Association Stadium, Dharamshala',
    'Holkar Cricket Stadium, Indore', 'JSCA International Stadium Complex, Ranchi',
    'M Chinnaswamy Stadium, Bangalore', 'MA Chidambaram Stadium, Chepauk',
    'Maharashtra Cricket Association Stadium, Pune',
    'Punjab Cricket Association Stadium, Mohali',
    'Raipur International Cricket Stadium, Raipur',
    'Rajiv Gandhi International Stadium, Uppal',
    'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium, Jaipur',
    'Sharjah Cricket Stadium, Sharjah', 'Sheikh Zayed Stadium, Abu-Dhabi',
    'Wankhede Stadium, Mumbai'
]

team_options = list(team_logos.keys())

def convert_overs_and_format(overs_str):
    try:
        if "." in overs_str:
            over_part, ball_part = overs_str.split(".")
            over_int = int(over_part)
            ball_int = int(ball_part)
            if ball_int < 6:
                overs_float = over_int + ball_int / 6
            else:
                added_overs = ball_int // 6
                leftover_balls = ball_int % 6
                overs_float = over_int + added_overs + leftover_balls / 6
        else:
            overs_float = float(overs_str)
        whole_overs = int(math.floor(overs_float))
        fractional_balls = round((overs_float - whole_overs) * 6)
        if fractional_balls == 0:
            overs_str_formatted = f"{whole_overs}"
        else:
            overs_str_formatted = f"{whole_overs}.{fractional_balls}"
        return overs_float, overs_str_formatted
    except:
        return None, None

def is_valid_int(val, min_val=None, max_val=None):
    try:
        v = int(val)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return False
        return True
    except:
        return False

def img_to_base64(img_path):
    with open(img_path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode()

tab1, tab2 = st.tabs(["Prediction", "How It Works"])

with tab1:
    st.header("Match Settings")
    team1_col, team2_col = st.columns(2)
    with team1_col:
        batting_team = st.selectbox("Batting Team", ['Select Team'] + team_options, key='bat_team')
        bat_logo_path = team_logos.get(batting_team, "") if batting_team != "Select Team" else ""
        if bat_logo_path and os.path.isfile(bat_logo_path):
            st.image(bat_logo_path, width=80)
    with team2_col:
        bowling_team = st.selectbox("Bowling Team", ['Select Team'] + team_options, key='bowl_team')
        bowl_logo_path = team_logos.get(bowling_team, "") if bowling_team != "Select Team" else ""
        if bowl_logo_path and os.path.isfile(bowl_logo_path):
            st.image(bowl_logo_path, width=80)

    venue = st.selectbox("Select Venue", ['Select Venue'] + venue_options)
    overs_input = st.text_input("Overs Completed", placeholder="e.g. 10.2 (balls must be 0 to 5 only)")
    runs = st.text_input("Current Runs", placeholder="e.g. 85")
    wickets = st.text_input("Wickets Fallen", placeholder="e.g. 3")
    runs_last_5 = st.text_input("Runs in Last 5 Overs", placeholder="e.g. 40")
    wickets_last_5 = st.text_input("Wickets in Last 5 Overs", placeholder="e.g. 1")

    error_placeholder = st.empty()
    overs_float, overs_str_formatted = convert_overs_and_format(overs_input)

    if overs_input.strip() == "":
        error_placeholder.warning("Please enter overs completed.")
        overs_valid = False
    elif overs_float is None:
        error_placeholder.error("Invalid overs format! Please use format like 10.2 or 15.4")
        overs_valid = False
    elif overs_float > 20:
        error_placeholder.error("Overs completed cannot exceed 20.")
        overs_valid = False
    elif overs_float == 20:
        overs_valid = False
        st.success("🎉 The innings are over! No further score prediction is needed.")
    else:
        overs_valid = True

    run_valid = is_valid_int(runs, 0, 300)
    wickets_valid = is_valid_int(wickets, 0, 10)
    runs_last_5_valid = is_valid_int(runs_last_5, 0)
    wickets_last_5_valid = is_valid_int(wickets_last_5, 0, 5)

    if not run_valid and runs != "":
        error_placeholder.error("Current Runs must be a non-negative integer up to 300.")
    elif not wickets_valid and wickets != "":
        error_placeholder.error("Wickets Fallen must be an integer between 0 and 10.")
    elif not runs_last_5_valid and runs_last_5 != "":
        error_placeholder.error("Runs in Last 5 Overs must be a non-negative integer.")
    elif not wickets_last_5_valid and wickets_last_5 != "":
        error_placeholder.error("Wickets in Last 5 Overs must be an integer between 0 and 5.")
    elif venue == 'Select Venue' or batting_team == 'Select Team' or bowling_team == 'Select Team':
        error_placeholder.warning("Please select valid teams and venue.")
    elif batting_team == bowling_team:
        error_placeholder.error("Batting and Bowling teams cannot be the same.")
    elif not overs_valid:
        pass
    else:
        if runs == "" or wickets == "" or runs_last_5 == "" or wickets_last_5 == "":
            error_placeholder.warning("Please fill in all numeric inputs with valid values.")
        else:
            if st.button("Predict Final Score"):
                try:
                    runs_val = int(runs)
                    wickets_val = int(wickets)
                    runs_last_5_val = int(runs_last_5)
                    wickets_last_5_val = int(wickets_last_5)
                    run_rate = runs_val / overs_float if overs_float > 0 else 0
                    overs_remaining = 20 - overs_float
                    wickets_in_hand = 10 - wickets_val
                    input_numbers = [
                        runs_val, wickets_val, overs_float, overs_remaining,
                        run_rate, runs_last_5_val, wickets_last_5_val, wickets_in_hand
                    ]
                    numeric = np.array([input_numbers])
                    scaled_numeric = scaler.transform(numeric)
                    venue_vector = [1 if venue == v else 0 for v in venue_options]
                    bat_vector = [1 if batting_team == t else 0 for t in team_options]
                    bowl_vector = [1 if bowling_team == t else 0 for t in team_options]
                    if overs_float <= 6:
                        phase_vector = [1, 0, 0]
                    elif overs_float <= 15:
                        phase_vector = [0, 1, 0]
                    else:
                        phase_vector = [0, 0, 1]
                    input_vector = np.array(venue_vector + bat_vector + bowl_vector + phase_vector).reshape(1, -1)
                    final_input = np.concatenate((input_vector, scaled_numeric), axis=1)
                    if final_input.shape[1] < regressor.n_features_in_:
                        missing = regressor.n_features_in_ - final_input.shape[1]
                        padding = np.zeros((1, missing))
                        final_input = np.concatenate((final_input, padding), axis=1)
                    elif final_input.shape[1] > regressor.n_features_in_:
                        st.error("Input vector has more features than expected by the model.")
                        st.stop()
                    prediction_arr = regressor.predict(final_input)
                    prediction = int(prediction_arr[0])
                    predicted_run_rate = prediction / 20
                    st.success(f"🏏 **Predicted Final Score Range: {max(0, prediction - 5)} – {prediction + 10}**")
                    st.metric("Projected Final Run Rate", f"{predicted_run_rate:.2f}")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={'text': "Projected Final Score"},
                        gauge={'axis': {'range': [100, 300]}, 'bar': {'color': "#009e60"}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Context Card with logo path used in base64 encoding ---
                    bat_logo_path = team_logos.get(batting_team, "") if batting_team != "Select Team" else ""
                    bowl_logo_path = team_logos.get(bowling_team, "") if bowling_team != "Select Team" else ""
                    bat_logo_data_uri = ""
                    bowl_logo_data_uri = ""
                    if bat_logo_path and os.path.isfile(bat_logo_path):
                        bat_logo_data_uri = "data:image/png;base64," + img_to_base64(bat_logo_path)
                    if bowl_logo_path and os.path.isfile(bowl_logo_path):
                        bowl_logo_data_uri = "data:image/png;base64," + img_to_base64(bowl_logo_path)

                    match_context_html = f"""
<style>
.match-context-card {{
    background: #fff9db;
    box-shadow: 0 4px 16px 0 #ffea00, 0 0px 8px 0 #fff176, 0 0 16px 2px #ffeb3b;
    border-radius: 14px;
    padding: 28px 18px 18px 18px;
    margin-bottom: 24px;
    margin-top: 15px;
    border: 2.5px solid #ffe082;
}}
.team-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}}
.team-section {{
    display: flex;
    align-items: center;
}}
.team-logo {{
    width: 58px; height: 58px;
    background: white;
    border-radius: 50%;
    margin-right: 14px;
    border: 1.5px solid #ffe082;
    box-shadow: 0 0 8px #ffe08244;
}}
.team-logo-right {{
    margin-left: 14px;
    margin-right: 0;
}}
.vs-text {{
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffb300;
    margin: 0 18px;
}}
.card-title {{
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 10px;
    color: #e6b400;
}}
.stat-table {{
    width: 100%;
    margin-top: 18px;
    border-collapse: collapse;
}}
.stat-table th, .stat-table td {{
    text-align: center;
    padding: 8px 4px;
    font-size: 1.08rem;
}}
.stat-label {{
    font-weight: 600;
    color: #888651;
}}
.stat-value {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #009e60;
}}
@media (max-width: 700px) {{
    .team-logo {{ width:38px; height:38px; }}
    .stat-value {{ font-size:1.05rem; }}
    .card-title {{ font-size:1.14rem; }}
}}
</style>
<div class="match-context-card">
    <div class="card-title">📝 Match Context</div>
    <div class="team-row">
        <div class="team-section">
            <img src="{bat_logo_data_uri}" class="team-logo" alt="Batting Logo">
            <div>
                <div style="font-size:1.03rem;font-weight:700;">Batting</div>
                <div style="font-size:1.15rem;color:#2962ff;font-weight:700;">{batting_team}</div>
            </div>
        </div>
        <div class="vs-text">⚡</div>
        <div class="team-section" style="flex-direction: row-reverse;">
            <img src="{bowl_logo_data_uri}" class="team-logo team-logo-right" alt="Bowling Logo">
            <div style="text-align:right;">
                <div style="font-size:1.03rem;font-weight:700;">Bowling</div>
                <div style="font-size:1.15rem;color:#e0af15;font-weight:700;">{bowling_team}</div>
            </div>
        </div>
    </div>
    <table class="stat-table">
        <tr>
            <th class="stat-label">Overs Completed</th>
            <th class="stat-label">Runs Scored</th>
            <th class="stat-label">Wickets Lost</th>
            <th class="stat-label">Runs (Last 5)</th>
            <th class="stat-label">Wkts (Last 5)</th>
            <th class="stat-label">Wickets in Hand</th>
        </tr>
        <tr>
            <td class="stat-value">{overs_str_formatted}</td>
            <td class="stat-value">{runs_val}</td>
            <td class="stat-value">{wickets_val}</td>
            <td class="stat-value">{runs_last_5_val}</td>
            <td class="stat-value">{wickets_last_5_val}</td>
            <td class="stat-value">{10 - wickets_val}</td>
        </tr>
    </table>
</div>
"""
                    st.markdown(match_context_html, unsafe_allow_html=True)

                except Exception as e:
                    error_placeholder.error("Input error or internal error occurred! Check values and try again.")
                    st.exception(e)

with tab2:
    st.header("How It Works")
    st.markdown("""
This app predicts the projected **first-innings final score in IPL matches** using:
- **Inputs:** Venue, batting team, bowling team, runs, wickets, overs, and last-5 overs stats.
- **Feature Engineering:**  
  - Venue and teams are converted into one-hot encoded vectors.  
  - Numeric inputs like runs, wickets, and overs are standardized using a `scaler`.  
  - Match phase (Powerplay, Middle, Death) is encoded separately.  
- **Model:** A Ridge Regression model trained on IPL historical ball-by-ball data.
- **Prediction Logic:**  
  The model combines current run rate, wickets in hand, overs remaining, and recent scoring momentum (last 5 overs)  
  to estimate the final first-innings score.
✅ The app validates all inputs before prediction.  
⚠️ Predictions are **estimates** and should be treated as guidance — cricket is unpredictable!  
---
*Developed for cricket analytics enthusiasts.* 🏏
    """)
