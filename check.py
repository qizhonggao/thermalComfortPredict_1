import pandas as pd
import torch


df2 = pd.read_csv('cleaned_asds.csv')
print(df2['Thermal comfort'].unique())