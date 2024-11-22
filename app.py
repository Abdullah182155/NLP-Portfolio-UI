import streamlit as st
from datetime import datetime
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.model_utils import predict_with_model_in_memory
from utils.data_utils import divide_results_and_save_to_memory
from utils.portfolio_utils import get_next_dates, window_weight

def run_app():
    st.title("NLP Portfolio App")

    data_dict = {}  # In-memory storage

    data_folder = st.text_input("Data Folder:", "Processed_Datasets/")
    model_folder = st.text_input("Model Folder:", "best_models/")
    start_date_input = st.date_input("Start Date:", datetime(2020, 10, 1))
    end_date_input = st.date_input("End Date:", datetime(2024, 10, 1))
    stocks = st.multiselect("Select Stocks:", options=[f.split('.')[0] for f in os.listdir(model_folder) if f.endswith('.pkl')])
    strategies = st.multiselect("Select Strategies:", ['return_based', 'risk_adjusted', 'expected_return', 'positive_news'])
    initial_invested_cash = st.number_input("Total Cash:", min_value=0, value=100000)
    original_invested_cash = initial_invested_cash  # Store the original invested cash value
    
    results = []

    if st.button("Run Predictions"):
        rolling_windows_data = {}
        for stock_symbol in stocks:
            model_file_path = os.path.join(model_folder, f'{stock_symbol}.pkl')
            csv_file_path = os.path.join(data_folder, f'{stock_symbol}.csv')
            results_df = predict_with_model_in_memory(model_file_path, csv_file_path, stock_symbol=stock_symbol, start_date=start_date_input, end_date=end_date_input)
            
            # Store results in memory
            rolling_windows_data[stock_symbol] = divide_results_and_save_to_memory(results_df, stock_symbol)
            
        st.success("Predictions completed and stored in memory.")

        # Count the number of windows dynamically
        window_num = max(len(rolling_windows_data[s]) for s in rolling_windows_data)

        # Initialize traces for plotting
        model_return_traces = []
        invested_cash_traces = []
        accuracy_traces = []

        for strategy in strategies:
            strategy_results = []
            current_start_date = start_date_input  # Start date for the first window
            initial_invested_cash = original_invested_cash  # Reset cash for each strategy
            
            for i in range(1, window_num + 1):
                start_date_str = current_start_date.strftime('%Y-%m-%d')
                end_date_str = get_next_dates(start_date_str, months=3)[1]  # End date is three months ahead

                total_model_return_cash, total_model_return, accuracy, weights, filtered_returns, portfolio_results = window_weight(
                    i, start_date_str, end_date_str, initial_invested_cash, strategy, rolling_windows_data
                )
              
                total_cash = initial_invested_cash + total_model_return_cash


                strategy_results.append({
                    "Window": i,
                    "Start Date": start_date_str,
                    "End Date": end_date_str,
                    "Initial Invested Cash": initial_invested_cash,
                    "Total Model Return Cash": total_model_return_cash,
                    "Model Return (%)": total_model_return,
                    "Accuracy (%)": accuracy,
                    "Invested Cash": total_cash,
                    "Weights": weights
                })
                initial_invested_cash = total_cash
                current_start_date = datetime.strptime(end_date_str, '%Y-%m-%d') + pd.DateOffset(days=1)

            results_df = pd.DataFrame(strategy_results)
            st.write(f"Portfolio Results by Window for Strategy: {strategy}")
            st.dataframe(results_df)

            # Check if weight data is available
            weight_data = [
                {"Window": result["Window"], "Company": company, "Weight": weight}
                for result in strategy_results
                if "Weights" in result and isinstance(result["Weights"], dict)
                for company, weight in result["Weights"].items()
            ]

            if not weight_data:
                st.warning(f"No weight data available for strategy: {strategy}")
                continue

            weight_df = pd.DataFrame(weight_data)

            if weight_df.empty:
                st.warning(f"Weight DataFrame is empty for strategy: {strategy}")
                continue

            fig_weights_bar = px.bar(
                weight_df,
                x="Window",
                y="Weight",
                color="Company",
                title=f"Weights of Each Company Across Windows for Strategy: {strategy}",
                barmode='stack'
            )
            fig_weights_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_weights_bar)

            results_df["Start Date"] = pd.to_datetime(results_df["Start Date"])
            model_return_traces.append(go.Scatter(
                x=results_df["Start Date"], y=results_df["Model Return (%)"], mode='lines+markers', name=f'{strategy} - Model Return (%)'
            ))
            invested_cash_traces.append(go.Scatter(
                x=results_df["Start Date"], y=results_df["Invested Cash"], mode='lines+markers', name=f'{strategy} - Invested Cash'
            ))
            accuracy_traces.append(go.Scatter(
                x=results_df["Start Date"], y=results_df["Accuracy (%)"], mode='lines+markers', name=f'{strategy} - Accuracy (%)'
            ))

        # Plot Model Returns
        fig_model_return = go.Figure(data=model_return_traces)
        fig_model_return.update_layout(
            title='Model Return (%) Over Time for All Selected Strategies',
            xaxis_title="Start Date", yaxis_title="Model Return (%)", template="plotly_white"
        )
        st.plotly_chart(fig_model_return)

        # Plot Invested Cash
        fig_invested_cash = go.Figure(data=invested_cash_traces)
        fig_invested_cash.update_layout(
            title='Invested Cash Over Time for All Selected Strategies',
            xaxis_title="Start Date", yaxis_title="Invested Cash", template="plotly_white"
        )
        st.plotly_chart(fig_invested_cash)

        # Plot Accuracy
        fig_accuracy = go.Figure(data=accuracy_traces)
        fig_accuracy.update_layout(
            title='Accuracy (%) Over Time for All Selected Strategies',
            xaxis_title="Start Date", yaxis_title="Accuracy (%)", template="plotly_white"
        )
        st.plotly_chart(fig_accuracy)

