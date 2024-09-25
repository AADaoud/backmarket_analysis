import pandas as pd
import streamlit as st

# Load the cleaned data
df = pd.read_csv("iphone_price_averages.csv")

# Streamlit app layout
st.title("iPhone Price Trend Analysis")

# Extract unique values for the model dropdown
models = df['Model'].unique()

# Dropdown for selecting the phone model
selected_model = st.selectbox("Select Model", models)

# Filter the data based on the selected model
filtered_model_df = df[df['Model'] == selected_model]

# Dynamically extract options for Storage, Colour, Grade, and Country based on the filtered data
storages = filtered_model_df['Storage'].unique()
colours = filtered_model_df['Colour'].unique()
grades = filtered_model_df['Grade'].unique()

# Extract country names from the column headers
country_columns = [col.split('_')[0] for col in df.columns if '_' in col]
countries = list(set(country_columns))

# Update options for other dropdowns based on the selected model
selected_storage = st.selectbox("Select Storage", storages)
selected_colour = st.selectbox("Select Colour", colours)
selected_grade = st.selectbox("Select Grade", grades)
selected_country = st.selectbox("Select Country", countries)

# Further filter the data based on the selected attributes
filtered_df = filtered_model_df[(filtered_model_df['Storage'] == selected_storage) &
                                (filtered_model_df['Colour'] == selected_colour) &
                                (filtered_model_df['Grade'] == selected_grade)]

# Check if there is any data for the selected phone configuration
if filtered_df.empty:
    st.warning("No data available for the selected phone configuration.")
else:
    # Filter for the selected country's price data columns (e.g., 'France_2024-04-13')
    country_columns = [col for col in filtered_df.columns if col.startswith(selected_country)]
    
    if not country_columns:
        st.warning(f"No data available for the selected country: {selected_country}.")
    else:
        # Remove the non-price columns to focus on the price data over time
        price_data = filtered_df[country_columns].T
        price_data.columns = ['Price']

        # Extract the date portion from the index (e.g., from 'France_2024-04-13' extract '2024-04-13')
        price_data.index = price_data.index.str.split('_').str[1]

        # Convert the index to datetime
        price_data.index = pd.to_datetime(price_data.index)

        # Plot the price trend over time
        st.line_chart(price_data)

        # Optionally, display the filtered data
        st.write("Filtered Data", filtered_df)
