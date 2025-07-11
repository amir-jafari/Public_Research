import pandas as pd
import re
import os
import glob
from bs4 import BeautifulSoup


def parse_school_table(school_name, table, date):
    import re
    import pandas as pd

    print(f"Parsing table for school: {school_name}")

    # Define column names
    columns = [
        'School_Name', 'Date', 'Identifier', 'Name',
        'Planned_Reimbursable', 'Planned_Non-Reimbursable', 'Planned_Total',
        'Offered_Reimbursable', 'Offered_Non-Reimbursable', 'Offered_Total',
        'Served_Reimbursable', 'Served_Non-Reimbursable', 'Served_Total',
        'Discarded_Total', 'Discarded_Cost', 'Subtotal_Cost',
        'Left_Over_Total', 'Left_Over_Percent_of_Offered', 'Left_Over_Cost',
        'Production_Cost_Total'
    ]

    data = []

    rows = table.find('tbody').find_all('tr')

    for idx, row in enumerate(rows):
        # Skip footer rows
        if row.get('class') and 'footer' in row.get('class'):
            continue

        cells = row.find_all('td')
        if len(cells) >= 18:
            try:
                # More robust name extraction
                identifier = cells[0].get_text(strip=True)
                name = cells[1].get_text(separator=' ', strip=True)

                row_data = [
                    school_name, date,
                    identifier, name,
                    cells[2].get_text(strip=True), cells[3].get_text(strip=True), cells[4].get_text(strip=True),
                    cells[5].get_text(strip=True), cells[6].get_text(strip=True), cells[7].get_text(strip=True),
                    cells[8].get_text(strip=True), cells[9].get_text(strip=True), cells[10].get_text(strip=True),
                    cells[11].get_text(strip=True), cells[12].get_text(strip=True), cells[13].get_text(strip=True),
                    cells[14].get_text(strip=True), cells[15].get_text(strip=True), cells[16].get_text(strip=True),
                    cells[17].get_text(strip=True)
                ]
                data.append(row_data)
            except Exception as e:
                print(f"Error parsing row {idx} in {school_name}: {e}")
        else:
            print(
                f"Skipping malformed row ({len(cells)} cells) in {school_name}: {[c.get_text(strip=True) for c in cells]}")

    if not data:
        print(f"No valid data rows found for school: {school_name}")
        return None

    df = pd.DataFrame(data, columns=columns)
    print(f"Created DataFrame for {school_name} with {len(df)} rows")
    return df


def parse_html_file(file_path):
    print(f"Processing file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as f:
        html_data = f.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_data, 'html.parser')

    # Try to find the filters section
    filters_section = soup.find(string=re.compile(r'Date Range', re.I))
    date = 'Unknown'

    if filters_section:
        print(f"Filters section found: {filters_section[:100]}...")
        # Try multiple date patterns
        date_patterns = [
            r'Date Range\s*\(Start = (\d+/\d+/\d+), End = \d+/\d+/\d+\)',  # MM/DD/YYYY
            r'Date Range\s*\(Start = (\d+-\d+-\d+), End = \d+-\d+-\d+\)',  # MM-DD-YYYY
            r'Date Range\s*\(Start = ([A-Za-z]+ \d+, \d{4}), End = [A-Za-z]+ \d+, \d{4}\)',  # Month DD, YYYY
            r'Date Range\s*:\s*(\d+/\d+/\d+)',  # Single date MM/DD/YYYY
            r'Date Range\s*:\s*(\d+-\d+-\d+)'  # Single date MM-DD-YYYY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, html_data, re.I)
            if match:
                date = match.group(1)
                break
    else:
        print(f"No filters section found in {file_path}")

    # Fallback: Try to infer date from file name (e.g., 5.01.25 breakfast.html → 5/1/2025)
    if date == 'Unknown':
        file_name = os.path.basename(file_path)
        date_match = re.search(r'(\d+)\.(\d+)\.(\d+)', file_name)
        if date_match:
            month, day, year = date_match.groups()
            year = f"20{year}" if len(year) == 2 else year  # Convert YY to YYYY
            date = f"{month}/{day}/{year}"
            print(f"Inferred date from file name: {date}")

    print(f"Using date: {date}")

    # Find all page-break divs (each contains a school)
    page_breaks = soup.find_all('div', class_='page-break')
    print(f"Found {len(page_breaks)} school sections in {file_path}")

    # Initialize list to store DataFrames for this file
    file_dfs = []

    # Process each school section
    for page_break in page_breaks:
        # Extract school name
        school_name_elem = page_break.find('div', class_='sub-heading').find('li')
        school_name = school_name_elem.text.strip() if school_name_elem else 'Unknown School'
        print(f"Processing school: {school_name}")

        # Extract table
        table = page_break.find('table', class_='striped')
        if table:
            df = parse_school_table(school_name, table, date)
            if df is not None:
                file_dfs.append(df)
        else:
            print(f"No table found for school: {school_name}")

    return file_dfs


def generate_csvs_from_folder(folder_path, output_dir='output_csvs'):
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all HTML files in the folder
    html_files = glob.glob(os.path.join(folder_path, '*.html'))
    print(f"Found {len(html_files)} HTML files in {folder_path}")

    if not html_files:
        print("Error: No HTML files found in the folder.")
        return

    # Process each HTML file
    for file_path in html_files:
        # Get the base name of the file and create CSV name
        file_name = os.path.basename(file_path)
        csv_name = os.path.splitext(file_name)[0] + '.csv'
        output_file = os.path.join(output_dir, csv_name)
        print(f"Generating CSV: {output_file}")

        # Parse the HTML file
        file_dfs = parse_html_file(file_path)

        if not file_dfs:
            print(f"No valid data found for {file_name}. Skipping CSV generation.")
            continue

        # Combine DataFrames for this file
        final_df = pd.concat(file_dfs, ignore_index=True)

        # Sort by School_Name, Date, Identifier
        final_df = final_df.sort_values(['School_Name', 'Date', 'Identifier'])

        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f"CSV file generated: {output_file}")


if __name__ == "__main__":
    folder_path = "../Fairfax_County/Data/FairfaxCounty/May 2025 Lunch production records/May 2025 Lunch production records"
    output_dir = "../Fairfax_County/Tim_test/lunch"
    generate_csvs_from_folder(folder_path, output_dir)