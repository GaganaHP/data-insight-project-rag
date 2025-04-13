import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
from joblib import load
import random

# Import the simulation function from utils (you can also directly embed it here)
from utils.simulate_data import simulate_row

# Load the pre-trained ML model
model = load("ml_model/model.pkl")

# Global dataframe to hold simulated data (in-memory)
if 'data' not in st.session_state:
    st.session_state.data = []

# Sidebar controls
st.sidebar.header("Controls")
simulation_speed = st.sidebar.slider("Simulation Interval (seconds)", 0.5, 5.0, 2.0, step=0.5)
start_simulation = st.sidebar.button("Start Simulation")

# Placeholder for real-time data and predictions
data_placeholder = st.empty()
prediction_placeholder = st.empty()

def update_dashboard():
    while st.session_state.get("simulate", False):
        new_row = simulate_row()
        # Make prediction using model; for a regression model predicting "sales":
        X_input = [[new_row['price'], new_row['clicks']]]
        new_row['predicted_sales'] = model.predict(X_input)[0]
        st.session_state.data.append(new_row)
        
        # Create a DataFrame and update display
        df = pd.DataFrame(st.session_state.data)
        data_placeholder.dataframe(df)
        prediction_placeholder.line_chart(df[['predicted_sales']])
        time.sleep(simulation_speed)

if start_simulation:
    st.session_state.simulate = True
    # Run simulation in a separate thread to allow the Streamlit app to update
    simulation_thread = threading.Thread(target=update_dashboard, daemon=True)
    simulation_thread.start()

st.header("Real-Time Data Simulation & Prediction")
st.write("New simulated data will appear below along with predictions.")


