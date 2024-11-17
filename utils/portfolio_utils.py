import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from Strategies.risk_adjusted import calculate_risk_adjusted_weights
from Strategies.return_based import calculate_stock_weights
from Strategies.expected_return import calculate_expected_weights
from Strategies.positive_news import calculate_positive_news_weights


def design_portfolio(resulted_df, stock_symbol, weight, total_cash):
    df = resulted_df.copy()
    Current_point = list(df['Close'])
    predicted_trend = list(df['Predicted_Future_Trend'])
    Actual_trend = list(df['Actual_Future_Trend'])

    model_decision = []
    invested_money = []
    total_money = []
    rest_money = []
    N_shares = []
    model_corrector = []
    sp_portfolio = []

    trade_yet = 0
    starting = True
    starting_cash = total_cash * weight

    wrong_decision = 0
    right_decision = 0
    total_fees = 0

    for c, p, a in zip(Current_point, predicted_trend, Actual_trend):
        if starting:
            starting_shares = starting_cash // c
            traded_money = starting_shares * c
            sp_portfolio.append(traded_money)
            starting = False
        else:
            sp_portfolio.append(starting_shares * c)

        if trade_yet > 0:
            last_decision = model_decision[-1]
            if p == 1:  # Predicted trend is 'Up'
                if last_decision == 'BUY' or last_decision == 'HOLD':
                    model_decision.append('HOLD')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_money[-1]
                    total_money.append(prev_shares * c + prev_rest)
                    invested_money.append(prev_shares * c)
                    rest_money.append(prev_rest)
                    N_shares.append(prev_shares)
                else:
                    model_decision.append('BUY')
                    prev_total_money = total_money[-1]
                    n_shares = prev_total_money // c
                    traded_money = n_shares * c
                    rest = prev_total_money % c
                    fee = max(3.5, n_shares * 0.01)
                    total_fees += fee
                    total_money.append(traded_money + rest - fee)
                    invested_money.append(traded_money)
                    rest_money.append(rest - fee)
                    N_shares.append(n_shares)
            elif p == -1:  # Predicted trend is 'Down'
                if last_decision == 'SELL' or last_decision == 'SKIP':
                    model_decision.append('SKIP')
                    total_money.append(total_money[-1])
                    invested_money.append(invested_money[-1])
                    rest_money.append(rest_money[-1])
                    N_shares.append(N_shares[-1])
                else:
                    model_decision.append('SELL')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_money[-1]
                    fee = max(3.5, prev_shares * 0.01)
                    total_fees += fee
                    total_money.append(prev_shares * c + prev_rest - fee)
                    invested_money.append(0)
                    rest_money.append(prev_shares * c + prev_rest - fee)
                    N_shares.append(0)
            elif p == 0:  # Predicted trend is 'Neutral'
                if last_decision == 'SELL' or last_decision == 'SKIP':
                    model_decision.append('SKIP')
                    total_money.append(total_money[-1])
                    invested_money.append(invested_money[-1])
                    rest_money.append(rest_money[-1])
                    N_shares.append(N_shares[-1])
                else:
                    model_decision.append('HOLD')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_money[-1]
                    total_money.append(prev_shares * c + prev_rest)
                    invested_money.append(prev_shares * c)
                    rest_money.append(prev_rest)
                    N_shares.append(prev_shares)
        else:
            if p == 1:  # Predicted trend is 'Up'
                trade_yet = True
                model_decision.append('BUY')
                n_shares = starting_cash // c
                traded_money = n_shares * c
                rest = starting_cash % c
                fee = max(3.5, n_shares * 0.01)
                total_fees += fee
                total_money.append(traded_money + rest - fee)
                invested_money.append(traded_money)
                rest_money.append(rest - fee)
                N_shares.append(n_shares)
            else:  # Predicted trend is 'Down' or 'Neutral'
                model_decision.append('SKIP')
                total_money.append(starting_cash)
                invested_money.append(0)
                rest_money.append(starting_cash)
                N_shares.append(0)

        last_decision = model_decision[-1]
        if (p == 1 and a == 1) or (p == -1 and a == -1) or (p == 0 and a == 0) or (p == 0 and a == 1):
            model_corrector.append('Right--' + last_decision)
            right_decision += 1
        else:
            model_corrector.append('Wrong--' + last_decision)
            wrong_decision += 1

    data = {
        'Date': df['Date'],
        'Day': pd.to_datetime(df['Date']).dt.day_name(),
        'Today Price': Current_point,
        'Predicted Trend': predicted_trend,
        'Actual Trend': Actual_trend,
        'Model Decision': model_decision,
        'Model Corrector': model_corrector,
        'Invested Money': invested_money,
        'Rest Money': rest_money,
        'Total Money': np.round(total_money, 2),
        'N_shares': N_shares,
        'Actual Stock Return': sp_portfolio
    }

    portfolio_df = pd.DataFrame(data)
    if starting_cash > 0:
        model_return = (total_money[-1] - starting_cash) / starting_cash * 100
        sp_return = (sp_portfolio[-1] - starting_cash) / starting_cash * 100
    else:
        model_return = 0
        sp_return = 0

    """print()
    print('------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------')
    print(f"Stock: {stock_symbol}")
    print(f"weight: {weight * 100}")
    print(f"Model Return: {model_return}")
    print('Actual Stock Return: ', sp_return)
    rw_ratio = (right_decision) / (right_decision + wrong_decision) * 100
    print('Accuracy: ', rw_ratio)
    print(f"Total Fees for {stock_symbol}: {total_fees}")"""

    return portfolio_df, starting_cash, right_decision, wrong_decision


def window_weight(window_num, start_date, end_date, total_cash, strategy, rolling_windows_data):
    prices_folder = 'Stocks Updates/'
    total_model_return_cash = 0
    total_sp_return_cash = 0
    quarter_right_decision = 0
    quarter_wrong_decision = 0
    stock_symbols = []
    weights = {}
    portfolio_results = {}  # To store portfolio results in memory
    print(rolling_windows_data.items())
    # Extract stock symbols from rolling_windows_data
    for stock_symbol, data in rolling_windows_data.items():
        if f'{stock_symbol}_window_{window_num}' in data:
            stock_symbols.append(stock_symbol)

    # Calculate weights based on the selected strategy
    if strategy == 'return_based':
        weights, filtered_returns = calculate_stock_weights(prices_folder, stock_symbols, start_date, end_date)
    elif strategy == 'risk_adjusted':
        weights, filtered_returns = calculate_risk_adjusted_weights(prices_folder, stock_symbols, start_date, end_date)
    elif strategy == 'expected_return':
        weights, filtered_returns = calculate_expected_weights(prices_folder, stock_symbols, start_date, end_date)
    elif strategy == 'positive_news':
        weights, filtered_returns = calculate_positive_news_weights('Processed_Datasets/', stock_symbols, start_date, end_date)

    # Portfolio processing
    for stock_symbol in stock_symbols:
        if stock_symbol in weights and f'{stock_symbol}_window_{window_num}' in rolling_windows_data[stock_symbol]:
            df = rolling_windows_data[stock_symbol][f'{stock_symbol}_window_{window_num}']
            #print(df.shape)
            if df.empty:
                print('df is empty')
                continue

            weight = weights[stock_symbol]
            portfolio_df, starting_cash, right_decision, wrong_decision = design_portfolio(df, stock_symbol, weight, total_cash)
            #print(portfolio_df)
            # Store results in memory
            portfolio_results[stock_symbol] = portfolio_df

            # Accumulate returns
            total_model_return_cash += (portfolio_df['Total Money'].iloc[-1] - starting_cash)
            total_sp_return_cash += (portfolio_df['Actual Stock Return'].iloc[-1] - starting_cash)
            quarter_right_decision += right_decision
            quarter_wrong_decision += wrong_decision

    total_model_return = (total_model_return_cash / total_cash) * 100
    total_sp_return = (total_sp_return_cash / total_cash) * 100
    accuracy = (quarter_right_decision / (quarter_right_decision + quarter_wrong_decision) * 100) if (quarter_right_decision + quarter_wrong_decision) > 0 else 0

    return total_model_return_cash, total_model_return, accuracy, weights, filtered_returns, portfolio_results







def get_next_dates(start_date, months=3):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = start_date + pd.DateOffset(months=months) - timedelta(days=1)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


