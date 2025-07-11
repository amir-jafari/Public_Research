#%%
import pandas as pd
import re

def process_sales_data_from_pdf_text(text, time_of_day):
    """
    Process sales data from pdfplumber text extraction to:
    1. Extract metadata (site, date, etc.)
    2. Parse item sales data
    3. Split item codes and descriptions into separate columns
    
    Parameters:
    text (str): Raw text extracted from PDF using pdfplumber
    time_of_day (str): Time period identifier
    
    Returns:
    pd.DataFrame: Processed dataframe with structured columns
    """
    
    # Extract metadata from text
    school_code = ""
    school_name = ""
    session_date = ""
    
    # Extract site information
    site_match = re.search(r"Site:\s*(\d{3})\s+(.+?)(?=\n|Session Date:)", text, re.DOTALL)
    if site_match:
        school_code = site_match.group(1)
        school_name = '_'.join(site_match.group(2).strip().split())
    
    # Extract session date
    date_match = re.search(r"Session Date:\s*(\d{2}/\d{2}/\d{4})", text)
    if date_match:
        session_date = date_match.group(1)
    
    # Find the table headers and data
    # Look for the header line that contains column names
    header_pattern = r"Item Description Total F R P A Stud\. Adult Stud\. Adult Stud\. Adult A la Carte Meal"
    header_match = re.search(header_pattern, text)
    
    if not header_match:
        # Try alternative header patterns
        header_pattern = r"Item Description.*?(?=\n\d+)"
        header_match = re.search(header_pattern, text)
    
    if not header_match:
        raise ValueError("Could not find table headers in the text")
    
    # Extract everything after the headers
    data_start = header_match.end()
    data_text = text[data_start:].strip()
    
    # Split into lines and process each item
    lines = data_text.split('\n')
    
    # Process lines to extract item data
    items_data = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number (item code)
        item_match = re.match(r'^(\d+)\s+(.+)', line)
        if item_match:
            # This is a new item
            if current_item:
                items_data.append(current_item)
            
            item_code = item_match.group(1)
            rest_of_line = item_match.group(2)
            
            # Parse the rest of the line to extract description and numeric values
            current_item = parse_item_line(item_code, rest_of_line)
        else:
            # This might be a continuation of the previous item's description
            if current_item and line:
                # Check if this line contains only text (continuation of description)
                if not re.search(r'\d+', line):
                    current_item['description'] += ' ' + line
    
    # Don't forget the last item
    if current_item:
        items_data.append(current_item)
    
    # Convert to DataFrame
    df = pd.DataFrame(items_data)
    
    # Add metadata columns
    df.insert(0, 'time_of_day', time_of_day)
    df.insert(1, 'school_code', school_code)
    df.insert(2, 'school_name', school_name)
    df.insert(3, 'date', session_date)
    
    return df

def parse_item_line(item_code, line_text):
    """
    Parse a single item line to extract description and numeric values
    Expected format: DESCRIPTION numbers...
    """
    # Split the line into tokens
    tokens = line_text.split()
    
    # Find where the numeric values start
    numeric_start_idx = None
    for i, token in enumerate(tokens):
        if re.match(r'^\d+$', token):
            numeric_start_idx = i
            break
    
    if numeric_start_idx is None:
        # No numeric values found, entire line is description
        return {
            'item': item_code,
            'description': line_text.strip(),
            'total': 0,
            'free_meals': 0,
            'reduced_price_meals': 0,
            'full_price_meals': 0,
            'adults': 0,
            'alac_student': 0,
            'alac_adult': 0,
            'earned_student': 0,
            'earned_adult': 0,
            'earned_alac_student': 0,
            'earned_alac_adult': 0,
            'adj_alac': 0,
            'adj_meal': 0
        }
    
    # Extract description (everything before numeric values)
    description = ' '.join(tokens[:numeric_start_idx]).strip()
    
    # Extract numeric values
    numeric_values = []
    for token in tokens[numeric_start_idx:]:
        try:
            numeric_values.append(int(token))
        except ValueError:
            break
    
    # Pad with zeros if not enough values
    while len(numeric_values) < 13:
        numeric_values.append(0)
    
    # Map to expected columns based on the header structure
    return {
        'item': item_code,
        'description': description,
        'total': numeric_values[0] if len(numeric_values) > 0 else 0,
        'free_meals': numeric_values[1] if len(numeric_values) > 1 else 0,
        'reduced_price_meals': numeric_values[2] if len(numeric_values) > 2 else 0,
        'full_price_meals': numeric_values[3] if len(numeric_values) > 3 else 0,
        'adults': numeric_values[4] if len(numeric_values) > 4 else 0,
        'alac_student': numeric_values[5] if len(numeric_values) > 5 else 0,
        'alac_adult': numeric_values[6] if len(numeric_values) > 6 else 0,
        'earned_student': numeric_values[7] if len(numeric_values) > 7 else 0,
        'earned_adult': numeric_values[8] if len(numeric_values) > 8 else 0,
        'earned_alac_student': numeric_values[9] if len(numeric_values) > 9 else 0,
        'earned_alac_adult': numeric_values[10] if len(numeric_values) > 10 else 0,
        'adj_alac': numeric_values[11] if len(numeric_values) > 11 else 0,
        'adj_meal': numeric_values[12] if len(numeric_values) > 12 else 0
    }
