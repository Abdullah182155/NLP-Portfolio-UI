# Calculate positive news weights
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
def calculate_positive_news_weights(folder_path, stock_symbols, start_date, end_date):
    positive_news_counts = {}

    for stock in stock_symbols:
        # قراءة بيانات الأخبار الخاصة بكل شركة
        filename = f"{stock}.csv"
        file_path = os.path.join(folder_path, filename)
        df_news = pd.read_csv(file_path)
        
        # التأكد من وجود الأعمدة المطلوبة
        if 'Date' not in df_news.columns or 'Label' not in df_news.columns:
            print(f"'Date' or 'Trend' column not found in {filename}")
            continue
        
        # تحويل عمود التاريخ لنوع datetime
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        # فلترة الأخبار بناءً على الفترة الزمنية المحددة
        df_filtered = df_news[(df_news['Date'] >= start_date) & (df_news['Date'] <= end_date)]
        
        # حساب عدد الأخبار الموجبة (Trend = 1)
        positive_news_count = df_filtered[df_filtered['Label'] == 1].shape[0]
        
        if positive_news_count > 0:
            positive_news_counts[stock] = positive_news_count

    # حساب الأوزان بناءً على عدد الأخبار الموجبة
    if positive_news_counts:
        total_positive_news = sum(positive_news_counts.values())
        weights = {stock: count / total_positive_news for stock, count in positive_news_counts.items()}
    else:
        weights = {}
        
    return weights ,positive_news_counts
