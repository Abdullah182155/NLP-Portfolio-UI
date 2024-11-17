import os
import pandas as pd


# Assuming the design_portfolio function is defined somewhere in your code


def calculate_risk_adjusted_weights(folder_path, stock_symbols, start_date, end_date):
    filtered_returns = {}
    
    for stock_symbol in stock_symbols:
        filename = f'{stock_symbol}.csv'
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        
        if df_filtered.empty or 'Close' not in df_filtered.columns:
            continue
        
        df_filtered['Daily_Return'] = df_filtered['Close'].pct_change()
        avg_return = df_filtered['Daily_Return'].mean()
        risk = df_filtered['Daily_Return'].std()
        
        # Skip stocks with missing or zero risk
        if pd.isna(avg_return) or pd.isna(risk) or risk == 0:
            continue
        
        risk_adjusted_return = avg_return / risk
        
        # Only consider stocks with risk-adjusted return > 0
        if risk_adjusted_return > 0:
            filtered_returns[stock_symbol] = risk_adjusted_return
    
    if filtered_returns:
        total_risk_adjusted = sum(filtered_returns.values())
        
        weights = {stock: ret / total_risk_adjusted for stock, ret in filtered_returns.items()}
        
        # Normalize weights to sum to 1
        weights = {stock: weight / sum(weights.values()) for stock, weight in weights.items()}
    else:
        weights = {}
    
    return weights, filtered_returns