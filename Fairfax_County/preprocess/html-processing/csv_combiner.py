import os
import glob
import pandas as pd

def combine_csvs_from_folder(input_dir, output_file, sort_columns=None):
    """
    Combine all CSV files in the specified folder into a single CSV file.

    Parameters:
    - input_dir (str): Folder containing the CSV files.
    - output_file (str): Path to the output combined CSV file.
    - sort_columns (list of str, optional): Columns to sort by before saving.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found at {input_dir}")
        return

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    if not csv_files:
        print("No CSV files found. Exiting.")
        return

    combined_df = pd.DataFrame()

    for file in csv_files:
        print(f"Reading {file}")
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        print("No data to write. Combined DataFrame is empty.")
        return

    if sort_columns:
        missing_cols = [col for col in sort_columns if col not in combined_df.columns]
        if missing_cols:
            print(f"Warning: Some sort columns not found in DataFrame: {missing_cols}")
        else:
            combined_df = combined_df.sort_values(sort_columns)

    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


# ============
# Breakfast
# ============
input_dir = '../../preprocess/html-processing/preprocessed-data/Breakfast production'
output_file = '../../preprocess/html-processing/preprocessed-data/Breakfast production/breakfast_combined.csv'
sort_columns = ['School_Name', 'Date', 'Identifier']

combine_csvs_from_folder(input_dir, output_file, sort_columns)


# ============
# Lunch
# ============
input_dir = '../../preprocess/html-processing/preprocessed-data/Lunch production'
output_file = '../../preprocess/html-processing/preprocessed-data/Lunch production/lunch_combined.csv'
sort_columns = ['School_Name', 'Date', 'Identifier']

combine_csvs_from_folder(input_dir, output_file, sort_columns)