import os
import pandas as pd
import numpy as np

# Assuming the design_portfolio function is defined somewhere in your code

def calculate_stock_weights(folder_path, stock_symbols, start_date, end_date):
    returns = {}
    # Iterate over all CSV files in the folder
    for stock_symbol in stock_symbols:
        filename = f'{stock_symbol}.csv'
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Ensure 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter dataframe to include data within the specified date range
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Skip if no data within the specified range
        if df_filtered.empty:
            returns[stock_symbol] = np.nan
            continue
        
        # Sort dataframe by date
        df_filtered = df_filtered.sort_values(by='Date')
        
        # Find nearest dates for start and end prices
        start_date_index = df_filtered['Date'].idxmin()
        end_date_index = df_filtered['Date'].idxmax()
        
        # Extract start and end prices
        start_price = df_filtered.loc[start_date_index, 'Close']
        end_price = df_filtered.loc[end_date_index, 'Close']
        
        # Calculate stock returns within the specified date range
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            returns[stock_symbol] = np.nan
        else:
            returns[stock_symbol] = (end_price - start_price) / start_price * 100

    # Filter out NaN, negative or zero returns
    filtered_returns = {stock: ret for stock, ret in returns.items() if not np.isnan(ret) and ret >= 0}

    if filtered_returns:
        # Calculate total return to normalize weights
        total_return = sum(filtered_returns.values())

        # Calculate weights
        weights = {stock: ret / total_return for stock, ret in filtered_returns.items()}

        # Normalize weights to sum to 1
        weights = {stock: weight / sum(weights.values()) for stock, weight in weights.items()}
    else:
        weights = {}

    return weights, filtered_returns


