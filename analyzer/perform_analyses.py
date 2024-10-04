import os
import pandas as pd
import glob
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import OAuth2Credentials
import json
import logging

# Logging to file/console
logger = logging.getLogger('backmarket_analytics')

# Get the current date and time for json/logs
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(f'logs/log_{current_time}.txt')
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG) # Handler for writing to file
console_handler.setLevel(logging.DEBUG) # Handler for writing to stdout

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def save_to_drive(csv_filepath: str) -> None:
    """
    Take in a CSV file and upload it to Google Drive using pydrive, oauth2credentials, and a credentials.json file.

    It also automatically updates the credentials file.

    Args:
        csv_file_name (str): Name of the input CSV file.
        csv_directory (str): Name of the output CSV directory.
    """
    try:
        # Load credentials from the JSON file
        with open('credentials.json', 'r') as file:
            credentials_data = json.load(file)

            # Create an OAuth2Credentials object
            credentials = OAuth2Credentials(
                access_token=credentials_data['access_token'],
                client_id=credentials_data['client_id'],
                client_secret=credentials_data['client_secret'],
                refresh_token=credentials_data['refresh_token'],
                token_expiry=credentials_data['token_expiry'],
                token_uri=credentials_data['token_uri'],
                user_agent=credentials_data['user_agent'],
                revoke_uri=credentials_data['revoke_uri'],
                id_token=credentials_data['id_token'],
                id_token_jwt=credentials_data['id_token_jwt'],
                token_response=credentials_data['token_response'],
                scopes=credentials_data['scopes'],
                token_info_uri=credentials_data['token_info_uri']
            )

        # Save credentials to a file (this is required by PyDrive)
        credentials_file = 'credentials.json'
        with open(credentials_file, 'w') as file:
            file.write(credentials.to_json())

        
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(credentials_file)
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()

        # Save the current credentials to a file
        gauth.SaveCredentialsFile(credentials_file)

        # Create Google Drive instance
        drive = GoogleDrive(gauth)

        google_file = drive.CreateFile(
            {
                "parents": [{"id": "1nz6zhvAti9caI_WcMyGc_ixYL220oKIi"}],
                "title": csv_filepath.replace("/analytics", "")
            }
        )

        google_file.SetContentFile(csv_filepath)
        google_file.Upload()
        google_file = None

    except Exception as e:
        logger.log(level=logging.ERROR, msg=f"An error occured while uploading the CSV to the Drive")
        logger.log(level=logging.ERROR, msg=(e, "message", str(e)))

def load_and_preprocess_csvs(path):
    """Load all CSVs from a directory and combine them into one DataFrame."""
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []

    for file in all_files:
        # Extract date from filename
        filename = os.path.basename(file).split('.')[0]
        date_str = filename.split('_')[1]
        scraping_date = pd.to_datetime(date_str, format='%Y%m%d')

        # Load the CSV file
        df = pd.read_csv(file)
        df['scraping_date'] = scraping_date

        # Convert all price columns to numeric, errors coerced to NaN
        price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_list.append(df)

    # Combine all files into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    return df

def clean_outliers(df, price_columns, threshold=2):
    """Replace outliers more than 2 standard deviations from the mean with the historical mean."""
    for col in price_columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = (df[col] - mean).abs() > (threshold * std)
        df.loc[outliers, col] = mean  # Replace outliers with the mean
    return df

def format_to_two_decimals(df, price_columns):
    """Format all price columns to two decimal places."""
    # Since DataFrame.map() applies the function to individual elements, the lambda function should handle scalar values.
    df[price_columns] = df[price_columns].map(lambda x: f'{x:.2f}' if pd.notnull(x) else x)
    return df


def get_average_over_period(df, days, price_columns):
    """Get average prices over the last 'days' days."""
    # Assume df has a 'scraping_date' column
    max_date = df['scraping_date'].max()
    start_date = max_date - pd.Timedelta(days=days - 1)
    df_period = df[(df['scraping_date'] >= start_date) & (df['scraping_date'] <= max_date)]
    # Group by phone category and compute mean
    df_avg = df_period.groupby(['Model', 'Storage', 'Colour', 'Grade'])[price_columns].mean().reset_index()
    return df_avg

def save_to_directory(output_dir, file_name, df):
    """Ensure the directory exists and save the DataFrame to a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)
    return output_path

def one_day_average(path):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = load_and_preprocess_csvs(path)

    # Define price columns to be cleaned and formatted
    price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']

    # Clean outliers
    df_clean = clean_outliers(df, price_columns)

    # Get 1-day average prices
    df_1d_avg = get_average_over_period(df_clean, 1, price_columns)

    # Format to two decimal places
    df_1d_avg = format_to_two_decimals(df_1d_avg, price_columns)

    # Save to the analytics/1_day_avg with timestamp
    output_dir = f"./analytics"
    output_file = f"1_day_avg_cleaned_{timestamp}.csv"
    csv_output_path = save_to_directory(output_dir, output_file, df_1d_avg)

    logger.info(f"1-day averaged data saved as '{csv_output_path}'")
    return csv_output_path  # Return the path for downstream use


def seven_day_average(path):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = load_and_preprocess_csvs(path)

    # Define price columns to be cleaned and formatted
    price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']

    # Clean outliers
    df_clean = clean_outliers(df, price_columns)

    # Get 7-day average prices
    df_7d_avg = get_average_over_period(df_clean, 7, price_columns)

    # Format to two decimal places
    df_7d_avg = format_to_two_decimals(df_7d_avg, price_columns)

    # Save to the analytics/7_day_avg with timestamp
    output_dir = f"./analytics"
    output_file = f"7_day_avg_cleaned_{timestamp}.csv"
    csv_output_path = save_to_directory(output_dir, output_file, df_7d_avg)

    logger.info(f"7-day averaged data saved as '{csv_output_path}'")
    return csv_output_path  # Return the path for downstream use


def thirty_day_average(path):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = load_and_preprocess_csvs(path)

    # Define price columns to be cleaned and formatted
    price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']

    # Clean outliers
    df_clean = clean_outliers(df, price_columns)

    # Get 30-day average prices
    df_30d_avg = get_average_over_period(df_clean, 30, price_columns)

    # Format to two decimal places
    df_30d_avg = format_to_two_decimals(df_30d_avg, price_columns)

    # Save to the analytics/30_day_avg with timestamp
    output_dir = f"./analytics"
    output_file = f"30_day_avg_cleaned_{timestamp}.csv"
    csv_output_path = save_to_directory(output_dir, output_file, df_30d_avg)

    logger.info(f"30-day averaged data saved as '{csv_output_path}'")
    return csv_output_path  # Return the path for downstream use


def forecast_prices(path, forecast_days=28):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = load_and_preprocess_csvs(path)

    # Define price columns to be cleaned and formatted
    price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']

    # Clean outliers
    df_clean = clean_outliers(df, price_columns)

    # Ensure date is sorted
    df_clean = df_clean.sort_values('scraping_date')

    forecast_results = []

    # Group by Model, Storage, Colour, and Grade
    grouped = df_clean.groupby(['Model', 'Storage', 'Colour', 'Grade'])

    for (model_name, storage, colour, grade), group in grouped:
        future_predictions = {}

        # Prepare the data for each group
        group = group.reset_index(drop=True)
        group['day_num'] = (group['scraping_date'] - group['scraping_date'].min()).dt.days

        for col in price_columns:
            X = group[['day_num']]
            y = group[col]

            # Handle NaN values in y
            if y.isna().sum() > 0:
                y = y.fillna(y.mean())  # Fill NaNs with mean of the column

            # Check if there are still NaN or insufficient data
            if y.isna().any() or len(y) < 2:
                logger.warning(f"Skipping forecast for {model_name}, {storage}, {colour}, {grade} in column {col} due to insufficient data.")
                continue

            # Train a linear regression model
            lr_model = LinearRegression()
            lr_model.fit(X, y)

            # Forecast for the next `forecast_days`
            last_day_num = group['day_num'].max()
            future_day_nums = np.arange(last_day_num + 1, last_day_num + forecast_days + 1)
            future_days = pd.DataFrame({'day_num': future_day_nums})
            predictions = lr_model.predict(future_days)

            future_predictions[col] = predictions

        if future_predictions:
            future_dates = pd.date_range(group['scraping_date'].max() + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame(future_predictions, index=future_dates)

            # **Ensure all price columns are present in forecast_df**
            for col in price_columns:
                if col not in forecast_df.columns:
                    forecast_df[col] = np.nan

            # Add the phone category columns
            forecast_df['Model'] = model_name
            forecast_df['Storage'] = storage
            forecast_df['Colour'] = colour
            forecast_df['Grade'] = grade

            # Reset index to get 'Forecast Date' column
            forecast_df = forecast_df.reset_index().rename(columns={'index': 'Forecast Date'})

            # Reorder columns
            forecast_df = forecast_df[['Forecast Date', 'Model', 'Storage', 'Colour', 'Grade'] + price_columns]

            forecast_results.append(forecast_df)

    # Concatenate all forecasted results
    if forecast_results:
        forecast_df = pd.concat(forecast_results, ignore_index=True)

        # Format to two decimal places
        forecast_df = format_to_two_decimals(forecast_df, price_columns)

        # Save to the analytics with timestamp
        output_dir = f"./analytics"
        output_file = f"{forecast_days}_day_forecast_{timestamp}.csv"
        csv_output_path = save_to_directory(output_dir, output_file, forecast_df)

        logger.info(f"{forecast_days}-day forecast saved as '{csv_output_path}'")
        return csv_output_path  # Return the path for downstream use
    else:
        logger.warning("No valid data available for forecasting.")
        return None


def perform_analyses(directory_path: str = "./backmarket_csv/"):
    file_paths = []
    file_paths.append(one_day_average(directory_path))
    file_paths.append(seven_day_average(directory_path))
    file_paths.append(thirty_day_average(directory_path))
    file_paths.append(forecast_prices(directory_path, forecast_days=28))

    for file_path in file_paths:
        save_to_drive(file_path)
