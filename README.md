# NLP Portfolio App

The **NLP Portfolio App** is a data-driven financial portfolio management tool that leverages NLP (Natural Language Processing) techniques and stock prediction models to assist users in managing and optimizing their investment portfolios. Built with **Streamlit** for a user-friendly interface and **Plotly** for interactive visualizations, this app allows users to analyze and simulate portfolio strategies based on various stock analysis models.

## Features

- **Stock Price Prediction**: Uses pre-trained models to predict stock price trends over specified date ranges.
- **Rolling Windows**: Automatically creates rolling windows for time-series analysis to assess portfolio performance across multiple time frames.
- **Multiple Investment Strategies**:
  - **Return-Based Strategy**
  - **Risk-Adjusted Strategy**
  - **Expected Return Strategy**
  - **Positive News Strategy**
- **Dynamic Visualization**: Generates interactive charts of "Model Return (%)" over time for each strategy, using Plotly.
- **User-Defined Parameters**: Select stocks, strategies, start/end dates, and investment amounts to customize portfolio simulations.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **Streamlit**: For app deployment and interface
- **Pandas**: For data handling and analysis
- **Plotly**: For creating interactive data visualizations

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/NLP-Portfolio-App.git
   cd NLP-Portfolio-App
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have a `best_models/` folder with pre-trained model files (`.pkl`) and a `Processed_Datasets/` folder with stock data in `.csv` format.

### Folder Structure

- **`Processed_Datasets/`**: Contains CSV files of stock data, with each file named after the stock symbol (e.g., `AAPL.csv`).
- **`best_models/`**: Contains the pre-trained model files for each stock, named according to the stock symbol (e.g., `AAPL.pkl`).
- **`Stock Window/`**: Folder where processed stock predictions are saved.
- **`rolling_windows/`**: Folder to store rolling window data for time-series analysis.

### Running the App

Run the app using Streamlit:

```bash
streamlit run app.py
```
