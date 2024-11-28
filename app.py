import streamlit as st
from datetime import datetime
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from utils.model_utils import predict_with_model_in_memory
from utils.data_utils import divide_results_and_save_to_memory
from utils.portfolio_utils import get_next_dates, window_weight

def run_app():
    st.title("NLP Portfolio App")

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
                    "Weights": weights,
                    "Filtered Returns": filtered_returns,
                })
                
                initial_invested_cash = total_cash
                current_start_date = datetime.strptime(end_date_str, '%Y-%m-%d') + pd.DateOffset(days=1)

            results_df = pd.DataFrame(strategy_results)
            st.subheader(f"Details for Strategy: {strategy}")
            st.dataframe(results_df)


            # Convert 'Start Date' and 'End Date' to datetime
            results_df["Start Date"] = pd.to_datetime(results_df["Start Date"], errors='coerce')
            results_df["End Date"] = pd.to_datetime(results_df["End Date"], errors='coerce')



            # Identify peaks and bottoms
            results_df["Peaks"] = results_df["Model Return (%)"] > peak_threshold_ratio
            results_df["Bottoms"] = results_df["Model Return (%)"] < bottom_threshold_ratio
            #print(results_df.columns)           

            # Add Period column based on Start and End Dates
            results_df["Period"] = results_df["Start Date"].dt.to_period("Q").astype(str)
            

            # Display tables for Peaks and Bottoms
            peaks_table = results_df.loc[results_df["Peaks"], ["Period", "Model Return (%)" ,"Weights","Filtered Returns", "Window"]]
            bottoms_table = results_df.loc[results_df["Bottoms"], ["Period", "Model Return (%)" ,"Weights","Filtered Returns", "Window"]]

            
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

            # Check if Filtered Returns data is available
            filtered_return_data = [
                {"Window": result["Window"], "Company": company, "Filtered Return": filtered_return}
                for result in strategy_results
                if "Filtered Returns" in result and isinstance(result["Filtered Returns"], dict)
                for company, filtered_return in result["Filtered Returns"].items()
            ]

            if not filtered_return_data:
                st.warning(f"No weight data available for strategy: {strategy}")
                continue

            filtered_return_df = pd.DataFrame(filtered_return_data)

            if filtered_return_df.empty:
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


            # Create the plot for peaks and bottoms
            fig_strategy = go.Figure()

            # Line for model return
            fig_strategy.add_trace(go.Scatter(
                x=results_df["Start Date"], y=results_df["Model Return (%)"],
                mode='lines+markers', name=f'{strategy} - Model Return (%)',
                line=dict(color='blue')
            ))

            # Mark peaks
            fig_strategy.add_trace(go.Scatter(
                x=results_df.loc[results_df["Peaks"], "Start Date"],
                y=results_df.loc[results_df["Peaks"], "Model Return (%)"],
                mode='markers', name='Peaks',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))

            # Mark bottoms
            fig_strategy.add_trace(go.Scatter(
                x=results_df.loc[results_df["Bottoms"], "Start Date"],
                y=results_df.loc[results_df["Bottoms"], "Model Return (%)"],
                mode='markers', name='Bottoms',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))

            # Update layout
            fig_strategy.update_layout(
                title=f'Model Return with Peaks and Bottoms for Strategy: {strategy}',
                xaxis_title="Start Date", yaxis_title="Model Return (%)",
                template="plotly_white"
            )

            # Display plot
            st.plotly_chart(fig_strategy)





            st.subheader(f"Peaks Table for Strategy: {strategy}")
            st.table(peaks_table[["Period", "Model Return (%)" ]])


            # Iterate over periods in the peaks table
            st.subheader(f"Bar Charts for Peaks - Strategy: {strategy}")
            for period in peaks_table["Period"].unique():
                period_weights = weight_df[weight_df["Window"].isin(
                    peaks_table[peaks_table["Period"] == period]["Window"])]
                if period_weights.empty:
                    st.warning(f"No weight data available for period {period} in peaks.")
                    continue

                period_Filtered_Returns = filtered_return_df[filtered_return_df["Window"].isin(
                    peaks_table[peaks_table["Period"] == period]["Window"])]
                if period_Filtered_Returns.empty:
                    st.warning(f"No weight data available for period {period} in peaks.")
                    continue

                # Create grouped bar chart with both 'Weight' and 'Filtered Returns'
                fig_peaks_bar = go.Figure()

                # Add bar for Weight
                fig_peaks_bar.add_trace(go.Bar(
                    x=period_weights["Company"],
                    y=period_weights["Weight"] * 100  ,
                    name="Weight",
                    marker=dict(color='blue'),
                ))

                # Add bar for Filtered Returns
                fig_peaks_bar.add_trace(go.Bar(
                    x=period_Filtered_Returns["Company"],
                    y=period_Filtered_Returns["Filtered Return"],
                    name="Filtered Return",
                    marker=dict(color='orange'),
                ))

                fig_peaks_bar.update_layout(
                    title=f"Weight and Filtered Returns Distribution for Period {period} - Peaks",
                    barmode='group',  # Group the bars together
                    xaxis_title="Company",
                    yaxis_title="Value",
                    template="plotly_white"
                )

                # Display the plot
                st.plotly_chart(fig_peaks_bar, key=f"peaks_bar_{strategy}_{period}")




            st.subheader(f"Bottoms Table for Strategy: {strategy}")
            st.table(bottoms_table[["Period", "Model Return (%)"]])


            
            # Iterate over periods in the bottoms table
            st.subheader(f"Bar Charts for Bottoms - Strategy: {strategy}")
            for period in bottoms_table["Period"].unique():
                period_weights = weight_df[weight_df["Window"].isin(
                    bottoms_table[bottoms_table["Period"] == period]["Window"])]
                if period_weights.empty:
                    st.warning(f"No weight data available for period {period} in peaks.")
                    continue

                period_Filtered_Returns = filtered_return_df[filtered_return_df["Window"].isin(
                    bottoms_table[bottoms_table["Period"] == period]["Window"])]
                if period_Filtered_Returns.empty:
                    st.warning(f"No weight data available for period {period} in peaks.")
                    continue

                # Create grouped bar chart with both 'Weight' and 'Filtered Returns'
                fig_peaks_bar = go.Figure()

                # Add bar for Weight
                fig_peaks_bar.add_trace(go.Bar(
                    x=period_weights["Company"],
                    y=period_weights["Weight"] * 100  ,
                    name="Weight",
                    marker=dict(color='blue'),
                ))

                # Add bar for Filtered Returns
                fig_peaks_bar.add_trace(go.Bar(
                    x=period_Filtered_Returns["Company"],
                    y=period_Filtered_Returns["Filtered Return"],
                    name="Return",
                    marker=dict(color='orange'),
                ))

                fig_peaks_bar.update_layout(
                    title=f"Weight and Filtered Returns Distribution for Period {period} - Bottoms",
                    barmode='group',  # Group the bars together
                    xaxis_title="Company",
                    yaxis_title="Value",
                    template="plotly_white"
                )

                # Display the plot
                st.plotly_chart(fig_peaks_bar, key=f"Bottoms_bar_{strategy}_{period}")






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