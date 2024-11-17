import pickle
import pandas as pd
import os
def load_model(model_file):
    """Load the model object using pickle."""
    try:
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")
        return None



def predict_with_model_in_memory(model_file_path=None, csv_data=None, stock_symbol="", start_date="", end_date=""):
    model = load_model(model_file_path)
    
    
    if model is None:
        print(f"Model for {stock_symbol} could not be loaded.")
        return {}

    if isinstance(csv_data, str):
        if not os.path.isfile(csv_data):
            print(f"Error: The file {csv_data} does not exist.")
            return {}
        csv_data = pd.read_csv(csv_data)
    
    csv_data['Date'] = pd.to_datetime(csv_data['Date'])
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    test = csv_data[(csv_data['Date'] >= start_date) & (csv_data['Date'] <= end_date)]
    
    X = test[['Title', 'Content', 'day', 'month', 'year', 'Close Before']]
    y = test['Label']

    predictions = model.predict(X)

    results_dic = {
        'Date': test['Date'],
        'Close': test['Close Before'],
        'Actual_Future_Trend': y,
        'Predicted_Future_Trend': predictions
    }
    results_df = pd.DataFrame(results_dic)

    # Return DataFrame instead of saving
    return results_df
#results_df = predict_with_model_in_memory(model_file_path='best_models/DT.pkl', csv_data='Processed_Datasets/DT.csv', stock_symbol="DT", start_date="10/01/2021", end_date="10/01/2023")
#print(results_df)