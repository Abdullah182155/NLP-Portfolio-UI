import pandas as pd

def divide_results_and_save_to_memory(results_df, stock_symbol):
    
    if 'Date' not in results_df.columns:
        raise KeyError("The 'Date' column is missing from the results_df dataframe.")

    results_df['Date'] = pd.to_datetime(results_df['Date'])

    start_date = results_df['Date'].min()
    end_date = results_df['Date'].max()
    
    index = 1
    rolling_windows_data = {}

    while start_date <= end_date:
        end_period = start_date + pd.DateOffset(months=3)
        mask = (results_df['Date'] >= start_date) & (results_df['Date'] < end_period)
        current_df = results_df.loc[mask].copy()
        
        if not current_df.empty:
            window_key = f'{stock_symbol}_window_{index}'
            rolling_windows_data[window_key] = current_df

        start_date = end_period
        index += 1

    return rolling_windows_data

# Example usage:
# stock_data = divide_results_and_save_to_memory(results_df, 'AAPL')