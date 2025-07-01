#%%
import tabula
import pandas as pd
import re 

pdf_file = "../data/Item Sales Reports - Mar May 2025/first_page_only.pdf"
csv_file = "output_data.csv"

tabula.convert_into(pdf_file, csv_file, output_format="csv", pages="all")

df = pd.read_csv("output_data.csv")
df
#%%

site = df[df.iloc[:, 0].astype(str).str.startswith("Site")].iloc[0,0]
match = re.match(r"Site:\s*(\d{3})(.+)", site)
if match:
    school_code = match.group(1)
    school_name = '_'.join(match.group(2).strip().split())

date = df[df.iloc[:, 0].astype(str).str.startswith("Session Date")].iloc[0,0]
match = re.match(r"Session Date:(.+)", date)
if match:
    session_date = match.group(1).strip()

#%%

indx_split = df[df.iloc[:, 0].astype(str).str.startswith("Item")].index[0]
sales_data = df.iloc[indx_split:, :].copy()
sales_data.columns = sales_data.iloc[0]
sales_data = sales_data.drop(indx_split).reset_index(drop=True)

# %%
