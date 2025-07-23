# School Lunch Production Data Parser & Combiner

This pipeline processes structured HTML reports of school lunch and breakfast production data and converts them into clean CSV files for analysis.

---

## 🔁 Overview: Step-by-Step Workflow

- ✅ **1. Run the HTML to CSV Transformer**  
  Use `data_transformer(HTML).py` to extract structured tables from `.html` reports and generate individual `.csv` files per report.

- ✅ **2. Run the CSV Combiner Script**  
  Use `csv_combiner.py` to merge the individual `.csv` files into two master datasets: one for lunch and one for breakfast.

---

## 🔍 What It Does

### `data_transformer(HTML).py`
- Reads all `.html` files from a specified folder.
- Extracts tables for each school using BeautifulSoup.
- Parses key metrics: planned, offered, served, discarded, and cost-related values.
- Automatically detects the reporting date from the content or file name.
- Saves parsed results as individual `.csv` files in the target directory.

### `csv_combiner.py`
- Collects all `.csv` files in an output folder.
- Merges them into one combined dataset.
- Optionally sorts by columns like `School_Name`, `Date`, and `Identifier`.
- Saves final merged CSVs for both **Breakfast** and **Lunch** datasets.

---

## 🛠️ Requirements

Make sure you have Python 3 and the required libraries installed:

```bash
pip install pandas beautifulsoup4
```

---

## ▶️ How to Run

### 1. Place Your HTML Files
Put your `.html` reports into a local folder for either lunch or breakfast data.

### 2. Run the HTML Parser

Open and modify the paths in `data_transformer(HTML).py`:

```python
folder_path = "your/html/folder/path"
output_dir = "your/csv/output/folder"
generate_csvs_from_folder(folder_path, output_dir)
```

Then run:

```bash
python data_transformer(HTML).py
```

Each `.html` file will produce one `.csv` file.

---

### 3. Combine the CSV Files

Edit the bottom of `csv_combiner.py`:

```python
input_dir = "your/csv/output/folder"
output_file = "combined_file.csv"
sort_columns = ['School_Name', 'Date', 'Identifier']
```

Run:

```bash
python csv_combiner.py
```

This will produce a combined, clean CSV file sorted by key dimensions.

---

