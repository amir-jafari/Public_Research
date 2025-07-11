# going to generate a new folder `preprocessed-data-folder`
#%%
import os
from utils import process_sales_data_from_pdf_text
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
import pdfplumber

os.chdir('../../data')

pdf_folder = "FairfaxCounty/Item Sales Reports - Mar May 2025/Item Sales Reports - Mar May 2025"

# Get all PDF files in the folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

print(f"📁 Found {len(pdf_files)} PDF files to process")

# Initialize overall tracking variables
all_clean_dfs = []
total_processed_count = 0
total_gt_count = 0

# Process each PDF file
for pdf_filename in pdf_files:
    pdf_file = os.path.join(pdf_folder, pdf_filename)
    
    print(f"\n🔄 Processing: {pdf_filename}")
    
    reader = PdfReader(pdf_file)
    num_pages = len(reader.pages)
    
    # Determine time of day from filename
    filename_lower = pdf_filename.lower()
    if 'breakfast' in filename_lower:
        time_of_day = "breakfast"
    elif 'lunch' in filename_lower:
        time_of_day = "lunch"
    else:
        # Default fallback - you might want to handle this differently
        time_of_day = "unknown"
        print(f"   ⚠️ Could not determine time of day from filename: {pdf_filename}")
        print(f"   📝 Using 'unknown' as time_of_day")
    
    # File-specific counters
    file_processed_count = 0
    file_gt_count = 0
    
    print(f"   📄 Processing {num_pages} pages...")
    
    for page_num in tqdm(range(1, num_pages + 1), desc=f"Processing {pdf_filename}"):
        
        with pdfplumber.open(pdf_file) as pdf:
            
            page_idx = page_num - 1
            
            if page_idx >= len(pdf.pages):
                print(f"   ⚠️ Page {page_num} does not exist in the PDF")
                continue
            
            page = pdf.pages[page_idx]
            text = page.extract_text()
            
            if 'GRAND TOTALS' in text:
                file_gt_count += 1
                total_gt_count += 1
                continue
            else:    
                clean_df = process_sales_data_from_pdf_text(text, time_of_day)
                
                if not clean_df.empty:
                    all_clean_dfs.append(clean_df)
                    file_processed_count += 1
                    total_processed_count += 1
                    
                    if total_processed_count % 50 == 0:
                        print(f"   ✅ Processed {total_processed_count} pages total (excluding grand total pages) so far...")
    
    print(f"   ✅ Completed {pdf_filename} ({time_of_day}): {file_processed_count} pages processed, {file_gt_count} grand total pages")

# Combine all data and save
if all_clean_dfs:
    final_df = pd.concat(all_clean_dfs, ignore_index=True)
    
    # Create preprocessed-data folder if it doesn't exist
    preprocessed_folder = "preprocessed-data"
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    
    output_file = os.path.join(preprocessed_folder, f"sales.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Final DataFrame saved to {output_file}")
    print(f"📊 Overall Processing Summary:")
    print(f"   - PDF files processed: {len(pdf_files)}")
    print(f"   - Successfully processed pages: {total_processed_count}")
    print(f"   - Grand Total pages: {total_gt_count}")
    print(f"   - Total records: {len(final_df)}")
    
    print("\nSample of processed data:")
    print(final_df.head())
    
else:
    print("⚠️ No data was processed successfully from any PDF files")
    print(f"📊 Overall Processing Summary:")
    print(f"   - PDF files found: {len(pdf_files)}")
    print(f"   - Successfully processed pages: {total_processed_count}")
    print(f"   - Grand Total pages: {total_gt_count}")

#%%