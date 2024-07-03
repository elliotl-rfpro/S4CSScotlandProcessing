import pandas as pd
import numpy as np
from datetime import datetime

raw_data = pd.ExcelFile('C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/weather_data.xlsx')
dis_data = pd.read_excel(raw_data, 'Disdrometer and PWS')
weather_data = pd.read_excel(raw_data, 'Weather station')

# ----- DISDROMETER DATA -----
for i in range(len(dis_data['mmPerHour(MMH)'])):
    # Forward values which were taken every 5 mins (replace nulls with previous values)
    if dis_data['mmPerHour(MMH)'].isnull().values[i]:
        dis_data['mmPerHour(MMH)'].values[i] = dis_data['mmPerHour(MMH)'].values[i - 1]
    if dis_data['SYNOP Code()'].isnull().values[i]:
        dis_data['SYNOP Code()'].values[i] = dis_data['SYNOP Code()'].values[i - 1]
    if dis_data['Total mm(mm)'].isnull().values[i]:
        dis_data['Total mm(mm)'].values[i] = dis_data['Total mm(mm)'].values[i - 1]

    # Replace negative values with previous (measurement issue)
    if dis_data['mmPerHour(MMH)'].values[i] < 0:
        dis_data['mmPerHour(MMH)'].values[i] = dis_data['mmPerHour(MMH)'].values[i - 1]
    if dis_data['SYNOP Code()'].values[i] < 0:
        dis_data['SYNOP Code()'].values[i] = dis_data['SYNOP Code()'].values[i - 1]
    if dis_data['Total mm(mm)'].values[i] < 0:
        dis_data['Total mm(mm)'].values[i] = dis_data['Total mm(mm)'].values[i - 1]

# Filters for easy access
clear_df = dis_data[dis_data['mmPerHour(MMH)'] == 0]                # No weather event
synop_df = dis_data[dis_data['mmPerHour(MMH)'] > 0]                 # Weather event detected
drizzle_df = dis_data[dis_data['SYNOP Code()'].between(50, 59)]     # Drizzle
rain_df = dis_data[dis_data['SYNOP Code()'].between(60, 69)]        # Rain
sleet_df = dis_data[dis_data['SYNOP Code()'].between(80, 89)]       # Sleet

# Low visibility but no rain (likely fog)
lv_df = clear_df[(clear_df['MORVisibility(m)'] < 1500) | (clear_df['Visibility(m)'] < 20000)]

# Eliminate scenarios with rain
rainless_lv_df = lv_df[lv_df['mmPerHour(MMH)'] == 0]

# ----- WEATHER DATA -----
# Find max of solar data and row (do like this because it's str coded). Remove erroneous/str entries from dataframe.
max_val = 0
max_ind = 0
for i in range(len(weather_data['Solar Energy'])):
    if isinstance(weather_data['Solar Energy'][i], str):
        weather_data['Solar Energy'][i] = 0.0
    if weather_data['Solar Energy'][i] > max_val:
        max_val = weather_data['Solar Energy'][i]
        max_ind = i

print("All done!")
