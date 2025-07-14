import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming your dataframe is named 'df'
df = pd.read_csv("Fairfax_County/Tim_test/(Breakfast)merged_with_metadata.csv")

# Clean cost columns safely
cost_columns = ['Discarded_Cost', 'Subtotal_Cost', 'Left_Over_Cost', 'Production_Cost_Total']

for col in cost_columns:
    # If column is already numeric, skip
    if pd.api.types.is_numeric_dtype(df[col]):
        continue

    # If column contains strings, clean them
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = (
            df[col]
            .astype(str)  # Ensure everything is a string
            .str.replace('$', '', regex=False)  # Remove $
            .str.replace(',', '', regex=False)  # Remove commas
            .replace('nan', '0')  # Replace NaN strings with 0
            .astype(float)  # Convert to float
        )
    else:
        # If not string or numeric, force conversion
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#%%

# Set Plotly to render in browser
pio.renderers.default = "browser"

# --- FIGURE 1: TOTAL COST BY SCHOOL ---
school_costs = df.groupby('School_Name_Mapped')['Production_Cost_Total'].sum().reset_index()
fig1 = px.bar(
    school_costs,
    x='School_Name_Mapped',
    y='Production_Cost_Total',
    title='Total Cost by School',
    labels={'School_Name_Mapped': 'School', 'Production_Cost_Total': 'Total Cost ($)'}
)
fig1.update_layout(xaxis={'categoryorder':'total descending'})


fig1.show()

#%%
# --- FIGURE 2: COST OVER TIME BY MENU ITEM ---
time_item_costs = df.groupby(['Date', 'Name'])['Production_Cost_Total'].sum().reset_index()
fig2 = px.line(
    time_item_costs,
    x='Date',
    y='Production_Cost_Total',
    color='Name',
    title='Cost Over Time by Menu Item',
    labels={'Date': 'Date', 'Production_Cost_Total': 'Total Cost ($)', 'Name': 'Menu Item'}
)

fig2.show()

#%%
# --- FIGURE 3: COST BY SCHOOL AND MENU ITEM ---
school_item_costs = df.groupby(['School_Name_Mapped', 'Name'])['Production_Cost_Total'].sum().reset_index()
fig3 = px.sunburst(
    school_item_costs,
    path=['School_Name_Mapped', 'Name'],
    values='Production_Cost_Total',
    title='Cost Distribution: Schools and Menu Items'
)

fig3.show()

#%%
# =====================
# Food Waste Analysis
# =====================

# Goal: Identify which schools/menu items have the highest waste (leftover + discarded).

# Calculate waste (Left_Over_Cost + Discarded_Cost)
df['Total_Waste_Cost'] = df['Left_Over_Cost'] + df['Discarded_Cost']

# Top 10 wasteful schools
waste_by_school = df.groupby('School_Name_Mapped')['Total_Waste_Cost'].sum().nlargest(10)
px.bar(waste_by_school, title='Top 10 Schools by Food Waste Cost').show()

# Top 10 wasteful menu items
waste_by_item = df.groupby('Name')['Total_Waste_Cost'].sum().nlargest(10)
px.bar(waste_by_item, title='Top 10 Wasted Menu Items').show()

#%%
# =====================
# Cost Efficiency Analysis
# =====================

# Goal: Compare planned vs. actual costs to identify inefficiencies.

# Calculate cost deviation (Actual - Planned)
df['Planned_Cost'] = df['Planned_Total'] * (df['Subtotal_Cost'] / df['Served_Total'].replace(0, 1))  # Approximate
df['Cost_Deviation'] = df['Production_Cost_Total'] - df['Planned_Cost']

# Schools with highest cost overruns
px.box(df, x='School_Name_Mapped', y='Cost_Deviation', title='Cost Deviation by School').show()


#%%
# =====================
# Popularity vs. Waste Heatmap
# =====================

# Goal: Visualize which items are both popular and wasteful.

# Aggregate data by menu item
item_stats = df.groupby('Name').agg({
    'Served_Total': 'sum',
    'Total_Waste_Cost': 'sum'
}).reset_index()

# Create a scatter plot with size = waste
px.scatter(
    item_stats,
    x='Served_Total',
    y='Total_Waste_Cost',
    size='Total_Waste_Cost',
    hover_name='Name',
    title='Popularity vs. Waste by Menu Item'
).show()

#%%
# =====================
# Temporal Trends (Day of Week)
# =====================

# Goal: Check if waste/cost varies by day of week.

# Convert 'Date' to datetime (if not already done)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check for failed conversions (NaT values)
if df['Date'].isna().any():
    print(f"Warning: {df['Date'].isna().sum()} rows have invalid dates.")

# Extract day of week (with safety checks)
df['Day_of_Week'] = df['Date'].dt.day_name()

# Handle missing/incorrect dates
df['Day_of_Week'] = df['Day_of_Week'].fillna('Unknown')

# Plot average waste by day
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Unknown']
waste_by_day = df.groupby('Day_of_Week', observed=True)['Total_Waste_Cost'].mean().reindex(day_order, fill_value=0)

fig = px.bar(
    waste_by_day.reset_index(),
    x='Day_of_Week',
    y='Total_Waste_Cost',
    title='Average Food Waste by Day of Week',
    labels={'Total_Waste_Cost': 'Average Waste Cost ($)', 'Day_of_Week': 'Day'}
)
fig.update_xaxes(categoryorder='array', categoryarray=day_order)
fig.show()


#%%
# =====================
# Regional Comparison
# =====================

# Goal: Compare costs/waste across FCPS regions or CEP vs. non-CEP schools.

# Cost per student (assuming 'Served_Total' ≈ students)
region_cost = df.groupby('FCPS Region').agg({
    'Production_Cost_Total': 'sum',
    'Served_Total': 'sum'
}).reset_index()
region_cost['Cost_Per_Student'] = region_cost['Production_Cost_Total'] / region_cost['Served_Total']

px.bar(
    region_cost,
    x='FCPS Region',
    y='Cost_Per_Student',
    title='Cost per Student by Region'
).show()


