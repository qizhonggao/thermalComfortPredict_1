import pandas as pd


df2 = pd.read_csv('ashrae_db2.csv', usecols=['Thermal preference', 'Season', 'Thermal comfort', 'Air temperature', 'Relative humidity', 'Outdoor air temperature'])

df2_cleaned = df2.dropna()


preference_mapping = {'cooler': -1, 'no change': 0, 'warmer': 1}
season_mapping = {'Spring': 3, 'Summer': 4, 'Autumn': 2, 'Winter': 1}
# use the map function to change the 'Season' column
df2_cleaned.loc[:, 'Season'] = df2_cleaned['Season'].map(season_mapping)
df2_cleaned.loc[:, 'Thermal preference'] = df2_cleaned['Thermal preference'].map(preference_mapping)

df2_cleaned['Thermal comfort'] = pd.to_numeric(df2_cleaned['Thermal comfort'], errors='coerce')

df2_cleaned = df2_cleaned.dropna()

# Select the rows where Air temperature is greater than 28 and Thermal comfort is less than 4
condition1 = (df2_cleaned['Air temperature'] > 29) & (df2_cleaned['Thermal comfort'] < 5)

# Select the rows where Air temperature is less than 18 and Thermal comfort is more than 4
condition2 = (df2_cleaned['Air temperature'] < 20) & (df2_cleaned['Thermal comfort'] > 4)

condition3 = (df2_cleaned['Air temperature'] > 24) & (df2_cleaned['Thermal comfort'] < 3) #& (df2_cleaned['Season'] == 2)

condition4 = (df2_cleaned['Air temperature'] < 25) & (df2_cleaned['Thermal comfort'] > 5) #& (df2_cleaned['Season'] == 2)


# Combine the conditions
combined_condition = condition1 | condition2 | condition3

# Select the rows that don't meet the conditions
df2_cleaned = df2_cleaned[~combined_condition]
df2_cleaned = df2_cleaned[df2_cleaned['Thermal comfort'].apply(lambda x: x.is_integer())]
df2_cleaned.to_csv('cleaned_asds.csv', index=False)
