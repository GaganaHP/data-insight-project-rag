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

from utils.simulate_data import simulate_row
from rag.rag_chatbot import generate_answer

# Load the pre-trained ML model
model = load("ml_model/model.pkl")

# Initialize session state for simulation
if 'data' not in st.session_state:
    st.session_state.data = []
if 'simulate' not in st.session_state:
    st.session_state.simulate = False

# Create tabs for simulation and document Q&A
tab1, tab2 = st.tabs(["📈 Real-Time Prediction", "📚 Document Q&A Chatbot"])

# === TAB 1: Real-Time Data Simulation & Prediction ===
with tab1:
    st.header("📈 Real-Time Data Simulation & Prediction")

    st.sidebar.header("Simulation Controls")
    simulation_speed = st.sidebar.slider("Simulation Interval (seconds)", 0.5, 5.0, 2.0, step=0.5)
    start_simulation = st.sidebar.button("Start Simulation")

    data_placeholder = st.empty()
    prediction_placeholder = st.empty()

    def update_dashboard():
        while st.session_state.get("simulate", False):
            new_row = simulate_row()
            X_input = [[new_row['price'], new_row['clicks']]]
            new_row['predicted_sales'] = model.predict(X_input)[0]
            st.session_state.data.append(new_row)

            df = pd.DataFrame(st.session_state.data)
            data_placeholder.dataframe(df)
            prediction_placeholder.line_chart(df[['predicted_sales']])
            time.sleep(simulation_speed)

    if start_simulation:
        st.session_state.simulate = True
        threading.Thread(target=update_dashboard, daemon=True).start()

    st.write("New simulated data will appear below along with predictions.")

# === TAB 2: Document Q&A Chatbot ===
with tab2:
    st.header("📚 Document Q&A Chatbot")

    user_query = st.text_input("Ask a question based on uploaded documents (PDFs)")
    if st.button("Get Answer") and user_query:
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query)
        st.success("Answer:")
        st.write(answer)
