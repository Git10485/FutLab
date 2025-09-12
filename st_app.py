import streamlit as st
import pandas as pd

import pickle

#---------------------------- Load model ----------------------------
with open("model.pkl", "rb") as file:
    classifier, ct, sc = pickle.load(file)

st.markdown(
    """
    <style>
    /* Remove main container & sidebar padding */
    .css-18e3th9, .css-1d391kg, [data-testid="stSidebar"] {
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    /* Remove gaps between columns */
    .css-1lcbmhc > div {
        gap: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")


# --- HEADER ---
st.markdown(
    "<h1 style='color:white; text-align:center;'>⚽ FutLab 🏆</h1>",
    unsafe_allow_html=True,
)

# ------------------------------- TOURNAMENT TYPES ----------------------------
tournament_map = {
    1: "Friendly (Low Importance)",
    2: "Regional (Medium Importance)",
    3: "Major Tournament (High Importance)"
}

# ---------------------------- Menu ----------------------------
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Navigate:", ["Reference Data", "About App"])


if menu == "Reference Data":
    data = {
    "Team": ["Argentina", "Spain", "France", "England", "Brazil", "Portugal", "Netherlands", "Belgium", "Germany", "Croatia",
             "Italy", "Mexico", "Colombia", "United States", "Uruguay", "Japan", "Switzerland", "Iran", "Denmark", "South Korea",
             "Australia", "Turkey", "Sweden", "Panama", "Egypt", "Algeria", "Poland", "Costa Rica", "Peru", "Nigeria",
             "Ivory Coast", "Tunisia", "Cameroon", "Qatar", "Uzbekistan", "South Africa", "Chile", "Iraq", "Saudi Arabia", "Jordan",
             "United Arab Emirates", "Jamaica", "Iceland", "Ghana", "Oman", "Zambia", "El Salvador", "Uganda", "Bahrain", "Honduras",
             "China PR", "Thailand", "Malaysia", "Estonia", "Kuwait"],
    "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55],
    "Goal Difference": [108, 103, 72, 42, 187, 77, 47, 63, 52, -5,
                        -2, 62, 66, 57, 45, 97, 4, 45, -9, 44,
                        24, -20, -1, -74, 1, 7, -33, -58, -42, -12,
                        0, -6, -23, -1, -4, -7, 10, -42, -20, -23,
                        9, -79, -81, -12, -31, -25, -93, -31, -54, -86,
                        -47, -50, -76, -118, -57]
    }

    
    df = pd.DataFrame(data)
    st.sidebar.subheader("Team Rankings & Goal Difference")
    st.sidebar.dataframe(df, use_container_width=True)

elif menu == "About App":
    st.sidebar.subheader("ℹ️ About This App")
    st.sidebar.markdown(
        """
        **FutLab - Football Match Result Predictor** is an interactive web app that lets you simulate football match outcomes.

        This project predicts the outcome of football matches based on team rankings, goal differences, and tournament type.  
        It uses a **Support Vector Machine (SVM) ** trained on match data to simulate realistic results.

        **Project Objectives:**
        - Provide an interactive platform for predicting match outcomes.
        - Demonstrate practical usage of ML models in sports analytics.
        
        **Technologies Used:**
        - Python, Pandas, Streamlit for web interface
        - Scikit-learn for machine learning
        - Pickle for saving/loading the trained model

        """
    )


# ------------------------------- FORM FOR MATCH INPUT -------------------------------
with st.form("match_form"):
    st.markdown("<h3 style='color:cyan;'>Enter Match Details</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🏠 **Enter the HOME TEAM**")
        home_team = st.text_input("Name of Home Team")
        st.markdown("✈️ **Enter the AWAY TEAM**")
        away_team = st.text_input(" Away Team")
        st.markdown("**Enter Home Team's Rank (1–55)**")
        home_rank = st.number_input("Home Rank", min_value=1, max_value=55, step=1, key='home_rank')

        st.markdown("**Enter Away Team's Rank (1–55)**")
        away_rank = st.number_input("Away Rank", min_value=1, max_value=55, step=1, key='away_rank')

        

    with col2:
        st.markdown("**Select Tournament Type**")
        tour_choice = st.radio(
            "Tournament Importance",
            options=[1, 2, 3],
            format_func=lambda x: tournament_map[x],
            )
        st.markdown(" **Enter Home Team's Goal Difference**")
        home_gd = st.number_input(" Home GD", step=1)
        st.markdown(" **Enter Away Team's Goal Difference**")
        away_gd = st.number_input(" Away GD", step=1)


   
    # ------------------------------- Submit Button -------------------------------
    submit = st.form_submit_button("🔮 Predict")
    

    if submit:
        if not home_team or not away_team:
            st.warning("⚠️ Please fill in all required fields.")
        else:
            prediction = classifier.predict(sc.transform(ct.transform([[home_team, away_team, tour_choice, home_rank, away_rank, home_gd, away_gd]])))[0]

            if prediction == 1:
                result = f"🏆 {home_team} Wins!"
            else:
                result = f"🏆 {away_team} Wins !"

            st.subheader(" Prediction Result")
            st.success(result)

