# This file is used to reformat the available dataset

import pandas as pd
import os

# Use pandas to read each csv file:
data_path = os.path.join(os.getcwd(), "weathermuse", "hourly_weather_data", "original_data")
city_attributes = pd.read_csv(f'{data_path}city_attributes.csv')
weather_description = pd.read_csv(f'{data_path}weather_description.csv')
temperature = pd.read_csv(f'{data_path}temperature.csv')
humidity = pd.read_csv(f'{data_path}humidity.csv')
pressure = pd.read_csv(f'{data_path}pressure.csv')
wind_direction = pd.read_csv(f'{data_path}wind_direction.csv')
wind_speed = pd.read_csv(f'{data_path}wind_speed.csv')

for index, row in city_attributes.iterrows():
    # Generate a new CSV file for all cities:
    city_name = row[0]
    file_name = f'{city_name.lower().replace(" ", "_")}.csv'

    # Define the file paths and the columns to extract
    file_columns = {
        f'{data_path}weather_description.csv': [city_name],
        f'{data_path}temperature.csv': [city_name],
        f'{data_path}humidity.csv': [city_name],
        f'{data_path}pressure.csv': [city_name],
        f'{data_path}wind_direction.csv': [city_name],
        f'{data_path}wind_speed.csv': [city_name]
    }

    # Create an empty DataFrame to hold the merged data
    merged_data = pd.DataFrame()

    # Read the first CSV file and extract the datetime column
    data = pd.read_csv(f'{data_path}weather_description.csv', usecols=['datetime'])
    # Concatenate the extracted data to the merged DataFrame
    merged_data = pd.concat([merged_data, data], axis=1)

    # Iterate through each file and the specified columns
    for file, columns in file_columns.items():
        # Read the CSV file and extract the required columns
        data = pd.read_csv(file, usecols=columns)
        # Concatenate the extracted data to the merged DataFrame
        merged_data = pd.concat([merged_data, data], axis=1)
    
    column_names = ['datetime', 'weather_description', 'temperature', 'humidity', 'pressure', 'wind_direction', 'wind_speed']
    merged_data.columns = [name for name in column_names]

    # Save the merged data to a new CSV file
    merged_data.to_csv(f'{data_path}{file_name}', index=False)