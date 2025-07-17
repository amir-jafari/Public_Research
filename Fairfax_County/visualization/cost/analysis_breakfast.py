import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import geojson
import random
from folium.plugins import FeatureGroupSubGroup
from pathlib import Path

# Assuming your dataframe is named 'df'
df = pd.read_csv("../../data/preprocessed-data/data_breakfast_with_coordinates.csv", low_memory=False)

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
school_costs = df.groupby('School Name')['Production_Cost_Total'].sum().reset_index()
fig1 = px.bar(
    school_costs,
    x='School Name',
    y='Production_Cost_Total',
    title='Total Cost by School',
    labels={'School Name': 'School', 'Production_Cost_Total': 'Total Cost ($)'}
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
school_item_costs = df.groupby(['School Name', 'Name'])['Production_Cost_Total'].sum().reset_index()
fig3 = px.sunburst(
    school_item_costs,
    path=['School Name', 'Name'],
    values='Production_Cost_Total',
    title='Cost Distribution: Schools and Menu Items'
)

fig3.show()

#%%
# Second verions of # --- FIGURE 3 ---

# Get top 5 most costly menu items overall
top_items = df.groupby('Name')['Production_Cost_Total'].sum().nlargest(5).index

# Filter the dataframe to only include these items
filtered_df = df[df['Name'].isin(top_items)]

# Generate sunburst chart
fig3 = px.sunburst(
    filtered_df,
    path=['School Name', 'Name'],
    values='Production_Cost_Total',
    title='Cost Distribution: Schools and Top 5 Menu Items (Overall)'
)
fig3.show()

#%%



#%%
# =====================
# Food Waste Analysis
# =====================

# Goal: Identify which schools/menu items have the highest waste (leftover + discarded).

# Calculate waste (Left_Over_Cost + Discarded_Cost)
df['Total_Waste_Cost'] = df['Left_Over_Cost'] + df['Discarded_Cost']

# Top 10 wasteful schools
waste_by_school = df.groupby('School Name')['Total_Waste_Cost'].sum().nlargest(10)
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

# Schools with the highest cost overruns
px.box(df, x='School Name', y='Cost_Deviation', title='Cost Deviation by School').show()


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
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
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

#%%
# ========================================
# Map of Total Production Costs by School
# ========================================

# Goal: Visualize the geographic distribution of breakfast program costs across schools,
# helping identify high-cost areas that may need budget review or operational adjustments.

# Aggregate data
school_geo = df.groupby(['School Name', 'latitude', 'longitude']).agg({
    'Production_Cost_Total': 'sum',
    'Total_Waste_Cost': 'sum'
}).reset_index()

# Create the map
fig_map = px.scatter_map(
    school_geo,
    lat='latitude',
    lon='longitude',
    size='Production_Cost_Total',
    color='Total_Waste_Cost',
    hover_name='School Name',
    hover_data={
        'Production_Cost_Total': True,
        'Total_Waste_Cost': True,
        'latitude': False,
        'longitude': False
    },
    zoom=10,
    title='School Breakfast Program: Costs and Waste'
)

# Use open-street-map styling (free, no token needed)
fig_map.update_layout(mapbox_style="open-street-map")
fig_map.show()

#%%
# Create an interactive map with multiple layers

# Goal: Compare cost and waste patterns side-by-side to identify schools where high
# costs correlate with high waste (potential inefficiency hotspots).

fig = go.Figure()

# Add cost layer
fig.add_trace(go.Scattermap(
    lat=school_geo['latitude'],
    lon=school_geo['longitude'],
    mode='markers',
    marker=dict(
        size=school_geo['Production_Cost_Total'] / 100,
        color=school_geo['Production_Cost_Total'],
        colorscale='Viridis',
        showscale=True
    ),
    text=school_geo['School Name'] + '<br>Cost: $' + school_geo['Production_Cost_Total'].round(2).astype(str),
    name='Production Costs'
))

# Add waste layer
fig.add_trace(go.Scattermap(
    lat=school_geo['latitude'],
    lon=school_geo['longitude'],
    mode='markers',
    marker=dict(
        size=school_geo['Total_Waste_Cost'] / 50,
        color=school_geo['Total_Waste_Cost'],
        colorscale='Hot',
        showscale=True
    ),
    text=school_geo['School Name'] + '<br>Waste: $' + school_geo['Total_Waste_Cost'].round(2).astype(str),
    name='Food Waste',
    visible=False
))

fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        center={"lat": 38.853, "lon": -77.299},
        zoom=10
    ),
    mapbox_layers=[],
    title="Interactive School Breakfast Program Map",
    updatemenus=[{
        "buttons": [
            {"method": "update", "args": [{"visible": [True, False]}], "label": "Show Costs"},
            {"method": "update", "args": [{"visible": [False, True]}], "label": "Show Waste"},
            {"method": "update", "args": [{"visible": [True, True]}], "label": "Show Both"}
        ],
        "direction": "down"
    }]
)

fig.show()


#%%
# --- Load Data ---
with open('Fairfax_County/Data/School_Regions.geojson', 'r') as f:
    geojson_data = geojson.load(f)

school_stats = df.groupby('School Name').agg({
    'latitude': 'first',
    'longitude': 'first',
    'Production_Cost_Total': 'sum',
    'Total_Waste_Cost': 'sum'
}).reset_index()

# --- Map Setup ---
map_center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")


# --- Color Utilities ---
def get_random_color():
    return "#{:02x}{:02x}{:02x}".format(
        random.randint(100, 255),  # Avoid dark colors
        random.randint(100, 255),
        random.randint(100, 255)
    )


region_colors = {
    feature['properties']['REGION']: get_random_color()
    for feature in geojson_data['features']
}

# --- Add Regions ---
folium.GeoJson(
    geojson_data,
    name='School Regions',
    style_function=lambda feature: {
        'fillColor': region_colors.get(feature['properties']['REGION'], '#808080'),  # Grey fallback
        'color': '#000000',
        'weight': 1,
        'fillOpacity': 0.5
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['REGION'],  # Confirm property name matches your GeoJSON
        aliases=['Region:'],
        sticky=True
    )
).add_to(m)

# --- Add School Data Layers ---
cost_group = folium.FeatureGroup(name='Cost by School', show=True)
waste_group = folium.FeatureGroup(name='Waste by School', show=False)

# Create subgroups
cost_subgroup = FeatureGroupSubGroup(cost_group)
waste_subgroup = FeatureGroupSubGroup(waste_group)

m.add_child(cost_group)
m.add_child(waste_group)
m.add_child(cost_subgroup)
m.add_child(waste_subgroup)

# Cost markers (blue)
for _, row in school_stats.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=max(3, 5 + (row['Production_Cost_Total'] / 1000)),  # Minimum radius
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f"Cost: ${row['Production_Cost_Total']:,.2f}",
    ).add_to(cost_subgroup)  # Add to subgroup directly

# Waste markers (red)
for _, row in school_stats.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=max(3, 5 + (row['Total_Waste_Cost'] / 500)),
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Waste: ${row['Total_Waste_Cost']:,.2f}",
    ).add_to(waste_subgroup)

# --- Finalize ---
folium.LayerControl(collapsed=False).add_to(m)  # Add LayerControl LAST
m.save("improved_breakfast_map.html")

