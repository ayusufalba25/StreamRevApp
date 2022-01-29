# Import required packages
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.write(
    """
    # Predict Streamer Revenue Based on PaidStarPerWatchedHour
    The model used is [Random Forest Regressor](https://github.com/ayusufalba25/StreamersFeatures).
    """
)

# Import model, mean_std, and dummy_columns
model = joblib.load("RFModel.pkl")
scaler = joblib.load("scaler.pkl")

# Create a function for input parameter
def input_parameter():

    # Input parameter
    st.header("Input Features Values")
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    country = st.selectbox(
        "Country",
        ["Philippines", "Other"]
    )
    game = st.selectbox(
        "Game",
        ["Free Fire - Battlegrounds", "PUBG", "Other"]
    )
    Character_Facet_Cont_Dutifulness = st.number_input(
        "Character_Facet_Cont_Dutifulness",
        min_value = 0.0,
        value = 0.36,
        step = 0.01
    )
    Character_Facet_Cont_Morality = st.number_input(
        "Character_Facet_Cont_Morality",
        min_value = 0.0,
        value = 0.15,
        step = 0.01
    )
    Self_Esteem_Cont_SEDiscrepancyIntelligence = st.number_input(
        "Self_Esteem_Cont_SEDiscrepancyIntelligence",
        min_value = 0.0,
        value = 0.49,
        step = 0.01
    )
    
    # Transform the numerical features using the normalization scaler
    numerical_features = pd.DataFrame({
        "Character_Facet_Cont_Dutifulness": [Character_Facet_Cont_Dutifulness],
        "Character_Facet_Cont_Morality": [Character_Facet_Cont_Morality],
        "Self_Esteem_Cont_SEDiscrepancyIntelligence": [Self_Esteem_Cont_SEDiscrepancyIntelligence]
    })
    numerical_features_scaled = pd.DataFrame(scaler.transform(numerical_features))
    numerical_features_scaled.columns = numerical_features.columns

    # Create dummy variables for categorical features
    # 1. Gender
    if gender == "Female":
        gender = 1
    else:
        gender = 0
    
    # 2. Country
    if country == "Philippines":
        country = 1
    else:
        country = 0
    
    # 3. Game
    if game == "Free - Fire Battlegrounds":
        game_ff = 1
        game_pubg = 0
    elif game == "PUBG":
        game_ff = 0
        game_pubg = 1
    else:
        game_ff = 0
        game_pubg = 0
    
    categorical_features = pd.DataFrame({
        "Gender_Female": [gender],
        "Country_PH": [country],
        "Game_Free Fire - Battlegrounds": [game_ff],
        "Game_PUBG": [game_pubg]
    })

    # Merge the categorical and numerical features
    input_data = pd.concat([categorical_features, numerical_features], axis = 1)

    return input_data

# Call the input parameter function
input_data = input_parameter()

# Prediction
pred = model.predict(input_data)

# Output
st.header("Result")
st.metric("Predicted PaidStarPerWatchedHour", pred[0])