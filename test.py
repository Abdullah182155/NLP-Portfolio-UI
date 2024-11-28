import streamlit as st
from transformers import pipeline
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from utils.model_utils import predict_with_model_in_memory
from utils.data_utils import divide_results_and_save_to_memory
from utils.portfolio_utils import get_next_dates, window_weight

# Load QA model from Hugging Face
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

def run_app():
    st.title("NLP Portfolio App with Chatbot")
    
    # Main application components (existing code)
    data_folder = st.text_input("Data Folder:", "Processed_Datasets/")
    model_folder = st.text_input("Model Folder:", "best_models/")
    start_date_input = st.date_input("Start Date:", datetime(2021, 1, 1))
    end_date_input = st.date_input("End Date:", datetime(2024, 10, 1))
    stocks = st.multiselect("Select Stocks:", options=[f.split('.')[0] for f in os.listdir(model_folder) if f.endswith('.pkl')])
    strategies = st.multiselect("Select Strategies:", ['return_based', 'risk_adjusted', 'expected_return', 'positive_news'])
    initial_invested_cash = st.number_input("Total Cash:", min_value=0, value=100000)
    original_invested_cash = initial_invested_cash  # Store the original invested cash value

    # Sidebar inputs for threshold ratios
    peak_threshold_ratio = st.sidebar.number_input("Peak Threshold Ratio:", min_value=0, value=5)
    bottom_threshold_ratio = st.sidebar.number_input("Bottom Threshold Ratio:", min_value=-10, value=0)

    results = []

    if st.button("Run Portfolio Simulation"):
        rolling_windows_data = {}
        for stock_symbol in stocks:
            model_file_path = os.path.join(model_folder, f'{stock_symbol}.pkl')
            csv_file_path = os.path.join(data_folder, f'{stock_symbol}.csv')
            results_df = predict_with_model_in_memory(model_file_path, csv_file_path, stock_symbol=stock_symbol, start_date=start_date_input, end_date=end_date_input)
            
            # Store results in memory
            rolling_windows_data[stock_symbol] = divide_results_and_save_to_memory(results_df, stock_symbol)
            
        st.success("Predictions completed and stored in memory.")

        # Your existing portfolio simulation code continues here...

    # Add Chatbot Section
    st.title("Chatbot for QA")
    st.write("Ask questions about the stock market, portfolio, or strategies.")

    context = st.text_area("Enter Context (e.g., Stock Analysis Details):", height=200, value="Provide a detailed description here.")
    question = st.text_input("Your Question:")

    if st.button("Ask"):
        if context and question:
            with st.spinner("Generating answer..."):
                answer = qa_model(question=question, context=context)
                st.subheader("Answer:")
                st.write(answer['answer'])
        else:
            st.warning("Please provide both a context and a question.")

if __name__ == "__main__":
    run_app()
