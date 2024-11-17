from utils.data_utils import divide_results_and_save_to_memory
import pandas as pd 
import os
def merge_folders_and_divide_in_memory(folder_names, source_base_folder, num_windows=42):
    rolling_windows_data = {}

    for folder_name in folder_names:
        folder_path = os.path.join(source_base_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                source_file = os.path.join(folder_path, file_name)
                if os.path.isfile(source_file):
                    # Load data into DataFrame
                    df = pd.read_csv(source_file)
                    stock_symbol = file_name.split('_')[0]
                    windows_data = divide_results_and_save_to_memory(df, stock_symbol)
                    rolling_windows_data.update(windows_data)

    return rolling_windows_data

# Usage:
# rolling_data = merge_folders_and_divide_in_memory(['folder1', 'folder2'], 'source_folder')