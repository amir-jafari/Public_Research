# HTML to CSV Parser for School Lunch Production Records

This Python script parses structured HTML reports of school lunch production data and converts them into clean CSV files for analysis.

---

## What It Does

- Reads all `.html` files from a specified folder.
- Extracts tables for each school from the HTML using BeautifulSoup.
- Parses planned, offered, served, discarded, and cost-related metrics.
- Automatically detects the reporting date from the file or content.
- Saves parsed results as individual `.csv` files.

---

## Requirements

Make sure you have Python 3 and the required libraries:

```bash
pip install pandas beautifulsoup4
```

---

## How to Use

1. **Place your HTML files** into a folder on your local machine.

2. **Download the script** into a Python file (e.g., `html_to_csv_parser.py`).

3. **Set your input and output paths**:  
   At the bottom of the script, modify this block:

   ```python
   if __name__ == "__main__":
       folder_path = "your/input/folder/path"
       output_dir = "your/output/folder/path"
       generate_csvs_from_folder(folder_path, output_dir)
   
 ⚠️ Important: Use valid local paths. These are not auto-detected and must match your system's folder structure.

4. **Run the script**:  
   Open a terminal, navigate to the script’s directory, and run:

5. **Check the output**:  
   The script will create a .csv file for each .html file and save it in the output directory you specified.


---

## Combine All CSVs Into One File

After generating individual CSV files from each HTML report, you can combine them into a single master CSV using the companion script `csv_combiner.py`.

### How to Run

1. Make sure your individual CSVs are saved in the output directory you specified earlier.

2. Open the `csv_combiner.py` file and set the following variables at the bottom of the script:

```python
input_dir = "your/output/folder/path"
output_file = "your/desired/final/combined_file.csv"
sort_columns = ['School_Name', 'Date', 'Identifier']
```

This will generate a single combined CSV file containing all records sorted by school name, date, and identifier.

⚠️ Note: Ensure input_dir matches the path where your individual CSVs were saved
