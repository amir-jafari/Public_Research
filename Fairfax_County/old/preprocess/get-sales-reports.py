#%%
from utils import process_sales_data_from_pdf_text
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
import pdfplumber

pdf_file = "../data/Item Sales Reports - Mar May 2025/may 2025 lunch item sales daily all sites.pdf"
reader = PdfReader(pdf_file)
num_pages = len(reader.pages)
time_of_day = "lunch"

all_clean_dfs = []
processed_count = 0
gt_count = 0

print(f"Processing {num_pages} pages...")

for page_num in tqdm(range(1, num_pages + 1)):
    
    with pdfplumber.open(pdf_file) as pdf:
        
        page_idx = page_num - 1
        
        if page_idx >= len(pdf.pages):
            print(f"⚠️ Page {page_num} does not exist in the PDF")
            continue
        
        page = pdf.pages[page_idx]
        text = page.extract_text()
        
        if 'GRAND TOTALS' in text:
            gt_count += 1
            continue
        else:    
            clean_df = process_sales_data_from_pdf_text(text, time_of_day)
            
            if not clean_df.empty:
                all_clean_dfs.append(clean_df)
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"✅ Processed {processed_count} pages (excluding grand total pages) so far...")
                    
if all_clean_dfs:
    final_df = pd.concat(all_clean_dfs, ignore_index=True)
    
    output_file = f"../data/{time_of_day}_sales_may_2025.csv"
    final_df.to_csv(output_file, index=False)
    print(f"✅ Final DataFrame saved to {output_file}")
    print(f"📊 Processing Summary:")
    print(f"   - Successfully processed: {processed_count} pages")
    print(f"   - Grand Total: {gt_count} pages")
    print(f"   - Total records: {len(final_df)}")
    
    print("\nSample of processed data:")
    print(final_df.head())
    
else:
    print("⚠️ No data was processed successfully")
    print(f"📊 Processing Summary:")
    print(f"   - Successfully processed: {processed_count} pages")
    print(f"   - Grand Total: {gt_count} pages")
# %%
