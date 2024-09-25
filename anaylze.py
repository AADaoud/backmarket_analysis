import pandas as pd
import glob

# Path to your folder with CSVs
path = "backmarket_csv/"
all_files = glob.glob(path + "*.csv")

df_list = []

# Load each file and extract date and time
for file in all_files:
    # Extract date and time from filename
    filename = file.split('\\')[-1].split('.')[0]
    date_str = filename.split('_')[1]
    time_str = filename.split('_')[2]
    
    # Convert date and time strings to proper datetime
    scraping_date = pd.to_datetime(date_str, format='%Y%m%d')
    
    # Load the CSV file
    df = pd.read_csv(file)
    
    # Add the scraping date as a new column
    df['scraping_date'] = scraping_date
    
    # Convert all price columns to numeric (assuming the country columns are numeric)
    price_columns = ['France', 'Germany', 'Italy', 'Spain', 'UK', 'USA']
    
    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors to NaN
    
    # Append the DataFrame to the list
    df_list.append(df)

# Concatenate all dataframes into one DataFrame
df = pd.concat(df_list, ignore_index=True)

# Now, group the data by Model, Storage, Colour, Grade, and Date, and take the mean of the price columns
grouped = df.groupby(['Model', 'Storage', 'Colour', 'Grade', 'scraping_date'], as_index=False)[price_columns].mean()

# Reshape the data so that you can see price changes over time for all countries
df_pivot = grouped.pivot_table(index=['Model', 'Storage', 'Colour', 'Grade'], 
                               columns='scraping_date', 
                               values=price_columns)

# Flatten the multi-level columns (scraping_date and countries) for better readability
df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns.values]

# Reset the index so that Model, Storage, Colour, and Grade become columns again
df_pivot.reset_index(inplace=True)
# Forward fill to propagate the last valid observation
df_pivot.ffill()

# Output the result to a CSV file
output_file = "iphone_price_averages.csv"
df_pivot.to_csv(output_file, index=False)

print(f"Data has been successfully saved to {output_file}")