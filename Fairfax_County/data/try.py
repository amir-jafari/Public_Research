#%%

import pandas as pd

df = pd.read_csv("preprocessed-data/data_breakfast_with_coordinates.csv")

example = df[(df["School_Name"] == "Aldrin Elementary") & (df["Date"] == "5/02/2025")]

# %%
