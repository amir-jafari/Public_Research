## Fairfax County Public Schools (FCPS) Research

This folder contains research regarding Fairfax County Public Schools (FCPS) data.

## Download Data

To download the data, run the following `wget` command:

```bash
  wget -O 'FairfaxCounty.zip' 'https://gwu.box.com/shared/static/wvgmx7n7tbrhuk140t8tm9zkb2z5top9.zip'
  d
```

## Requirements

Please make sure to install the following Python packages:

```bash
pip install pandas tqdm PyPDF2 pdfplumber
```

## Folder Structure

```bash
.
├── README.md
├── assets
├── data
├── preprocess
└── test-script-tyler.py
```

## Schema

Once you retrive the corresponding data using the `wget` command, you will notice that the schema is structured in the following manner:

<div style="text-align: center;">
  <table style="margin: 0 auto;">
    <tr>
      <td>
        <img src="assets/FCPS.drawio.svg" width="100%">
      </td>
    </tr>
  </table>
</div>

## Preprocess (In-progress)

This folder will focus on preprocessing data provided by Fairfax County Public Schools (FCPS) to convert it to tabular format for students.

### `get-menus.py`

The following script reads each corresponding Menu `.pdf` and converts it into a tabular format `.csv` of the same information.

$$
\text{.pdf} \to \text{.csv}
$$

### `get-production-records.py`

The following script reads each corresponding Production Record Breakfast/Lunch `.html` and converts it into a tabular format `.csv` of the same information.

$$
\text{.html} \to \text{.csv}
$$

### `get-sales-reports.py`

The following script reads each corresponding Item Sales Report Breakfast/Lunch `.pdf` and converts it into a tabular format `.csv` of the same information.

$$
\text{.pdf} \to \text{.csv}
$$
