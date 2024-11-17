import os
import pandas as pd

# Define a function to calculate quarterly returns for a DataFrame
def calculate_quarterly_returns(df, column_name='Close'):
    df = df.resample('Q').ffill()  # Resample data to quarterly frequency
    df['Quarterly_Return'] = df[column_name].pct_change() * 100  # Calculate quarterly returns
    df.dropna(subset=['Quarterly_Return'], inplace=True)  # Drop rows with NaN values
    return df

# Define a function to calculate expected returns using CAPM and quarterly returns
def calculate_expected_returns(prices_folder, stock_symbols, start_date, end_date):
    expected_returns = {}
    betas = {}
    sp500_file = 'sp500.csv'
    treasury_file = 'Treasury_10y.csv'

    # Load and process S&P 500 data
    sp500_df = pd.read_csv(sp500_file, parse_dates=['Date'], index_col='Date')
    sp500_df = calculate_quarterly_returns(sp500_df)
    sp500_df = sp500_df.loc[:end_date]

    # Load and process treasury data for the risk-free rate
    treasury_data = pd.read_csv(treasury_file, parse_dates=['Date'], index_col='Date')
    treasury_data = treasury_data.resample('Q').ffill()  # Resample to quarterly
    treasury_data = treasury_data.loc[:end_date]


    # Iterate over each stock to calculate beta and expected return
    for stock in stock_symbols:
        stock_path = os.path.join(prices_folder, f"{stock}.csv")
        
        # Check if the stock file exists
        if not os.path.isfile(stock_path):
            print(f"File not found for {stock}")
            continue

        # Load and process stock data
        stock_df = pd.read_csv(stock_path, parse_dates=['Date'], index_col='Date')
        stock_df = calculate_quarterly_returns(stock_df)
        stock_df = stock_df.loc[:end_date]  # Filter by date range

        
        # Calculate beta
        
        if 'Quarterly_Return' in stock_df.columns and 'Quarterly_Return' in sp500_df.columns:
            covariance = stock_df['Quarterly_Return'].cov(sp500_df['Quarterly_Return'])


            #print(covariance)
            market_variance = sp500_df['Quarterly_Return'].var()
           
            
            beta = covariance / market_variance 
        else:
            print(f"Missing data to calculate beta for {stock}")
            continue

        # Store beta and expected returns if beta is valid
        if beta is not None:
            betas[stock] = beta
            expected_returns[stock] = {}

            # Calculate expected returns using CAPM formula
            for date in stock_df.index:
                if date in treasury_data.index and date in sp500_df.index:
                    risk_free_rate = treasury_data.loc[date, 'Rate']
                    #print(risk_free_rate)
                    market_return = sp500_df.loc[date, 'Quarterly_Return']
                    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
                    expected_returns[stock][date] = expected_return

    return expected_returns

# Define a function to calculate expected weights
def calculate_expected_weights(prices_folder, stock_symbols, start_date, end_date):
    expected_returns = calculate_expected_returns(prices_folder, stock_symbols, start_date, end_date)

    # Get the latest return for each stock, excluding negative returns
    latest_returns = {stock: list(returns.values())[-1] for stock, returns in expected_returns.items() if returns}
    filtered_returns = {stock: ret for stock, ret in latest_returns.items() if ret > 0}

    # Calculate weights based on the filtered returns
    total_expected_return = sum(filtered_returns.values())
    weights = {stock: ret / total_expected_return for stock, ret in filtered_returns.items()} if total_expected_return != 0 else {}

    return weights, filtered_returns

