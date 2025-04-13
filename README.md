# Data Insight Platform with RAG-Based Document Q&A

## Overview

This project is an end-to-end data insight platform that simulates real-time data, predicts outcomes using a trained machine learning model, and integrates a document-based Q&A assistant using Retrieval-Augmented Generation (RAG). The final product is presented via an interactive Streamlit dashboard.

## Features

- **Real-Time Data Simulation:**
  - Generates synthetic data (product type, price, clicks) with timestamps.
  - Displays live data and predictions in a dynamic dashboard.
- **Machine Learning Model:**
  - Trains a regression model (RandomForestRegressor) on synthetic data.
  - Saves the model for reuse.
- **RAG Chatbot Integration:**
  - Extracts text from PDFs using PyMuPDF.
  - Indexes document chunks in a Pinecone vector database.
  - Retrieves and re-ranks relevant content to generate responses using a transformer-based language model.
- **Interactive Streamlit Dashboard:**
  - Provides separate tabs for real-time prediction and document Q&A.
  - Offers simulation controls (e.g., simulation interval) via sidebar inputs.

## Folder Structure

data_insight_platform/
├── app/
│ ├── **init**.py
│ └── main.py # Streamlit dashboard combining simulation & Q&A
├── config/
│ └── config.yaml # API keys & simulation settings (e.g., Pinecone API key)
├── data/
│ ├── simulated_data.csv # Optional log file for simulated data
│ └── docs/ # PDFs for the document QA assistant (3-5 open-source PDFs)
├── ml_model/
│ ├── train_model.py # Script to train and save the ML model
│ └── model.pkl # Saved model (via joblib)
├── rag/
│ ├── extract_text.py # Extract PDF text using PyMuPDF (or pdfplumber)
│ ├── build_index.py # Generate embeddings and index documents in Pinecone
│ └── rag_chatbot.py # RAG pipeline including retrieval, enhancement, and answer generation
├── utils/
│ ├── **init**.py
│ ├── simulate_data.py # Module to generate simulated real-time data
│ └── visualization.py # Utility functions for plotting and data handling
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation (detailed later)

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/data_insight_project_rag.git
   cd data_insight_project_rag

   ```

2. **Create and Activate a Virtual Environment:**
   On Windows:
   python -m venv venv
   venv\Scripts\activate
   On macOS/Linux:  
    python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies:**
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

4. **Configure API Keys:**
   Create a file named .env in the root directory with your Pinecone API key:
   PINECONE_API_KEY=your_actual_pinecone_api_key_here

## Running the Project

**Initial Setup:**
Train the ML Model:
Run the following command to generate synthetic data, train, and save your model:
python ml_model/train_model.py
Confirm that ml_model/model.pkl is created.

**Index Your Documents:**
(Ensure you have at least one sample PDF in data/docs/, e.g., python_pdf_sample.pdf.)
Run:
python rag/build_index.py
This will index the documents and print "Indexing complete!".

## Launch the Dashboard

**Run the complete Streamlit application using:**
streamlit run app/main.py
This single command launches the dashboard with two tabs:
Real-Time Prediction Tab: Displays live simulated data and predictions.
Document Q&A Chatbot Tab: Allows you to enter queries and displays answers generated using the RAG pipeline.
