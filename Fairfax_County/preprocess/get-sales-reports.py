#%%
import tabula
import pandas as pd
from tqdm import tqdm
import re
import os
from PyPDF2 import PdfReader

pdf_file = "../data/Item Sales Reports - Mar May 2025/apr 2025 breakfast item sales daily all sites.pdf"
fallback_month = "April" 

reader = PdfReader(pdf_file)
num_pages = len(reader.pages)

base_folder = "breakfast"
os.makedirs(base_folder, exist_ok=True)

for page_num in tqdm(range(1, num_pages + 1)):
    print(f"📄 Processing page {page_num}/{num_pages}...")

    page_csv = f"temp_page_{page_num}.csv"
    tabula.convert_into(pdf_file, page_csv, output_format="csv", pages=page_num)

    try:
        df = pd.read_csv(page_csv)
    except Exception as e:
        print(f"⚠️ Could not read page {page_num}: {e}")
        continue

    try:
        site_line = df[df.iloc[:, 0].astype(str).str.startswith("Site")].iloc[0, 0]
        match = re.match(r"Site:\s*(\d{3})(.+)", site_line)
        if match:
            school_code = match.group(1)
            school_name = '_'.join(match.group(2).strip().split())
        else:
            raise ValueError("No match for site line")
    except Exception as e:
        print(f"⚠️ Could not extract school name on page {page_num}: {e}")
        continue

    session_date_filename = None
    try:
        date_line = df[df.iloc[:, 0].astype(str).str.startswith("Session Date")].iloc[0, 0]
        match = re.match(r"Session Date:(.+)", date_line)
        if match:
            session_date = match.group(1).strip()
            session_date_filename = session_date.replace("/", "-")
    except Exception:
        pass  

    if not session_date_filename:
        session_date_filename = f"GrandTotal-{fallback_month}"
        print(f"⚠️ No session date found on page {page_num}, using fallback: {session_date_filename}")

    try:
        indx_split = df[df.iloc[:, 0].astype(str).str.startswith("Item")].index[0]
        sales_data = df.iloc[indx_split:, :].copy()
        sales_data.columns = sales_data.iloc[0]
        sales_data = sales_data.drop(indx_split).reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Could not process sales data on page {page_num}: {e}")
        continue

    school_folder = os.path.join(base_folder, school_name)
    os.makedirs(school_folder, exist_ok=True)
    
    output_csv_path = os.path.join(school_folder, f"{session_date_filename}.csv")
    sales_data.to_csv(output_csv_path, index=False)
    print(f"✅ Saved: {output_csv_path}")

    try:
        os.remove(page_csv)
    except Exception as e:
        print(f"⚠️ Could not delete temp file {page_csv}: {e}")
# %%
