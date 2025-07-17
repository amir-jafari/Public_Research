import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import geojson
import random
from folium.plugins import FeatureGroupSubGroup
from streamlit_folium import folium_static
import numpy as np
# import os

# os.chdir("../../data/preprocessed-data")

# Set page config
st.set_page_config(
    page_title="Breakfast Program Analytics Dashboard",
    page_icon="🍳",
    layout="wide"
)

# Title
st.title("🍳 School Breakfast Program Analytics Dashboard")
st.markdown("---")

# Load and preprocess data
@st.cache_data
def load_data():
    # Load your data here - adjust path as needed
    df = pd.read_csv("../data/preprocessed-data/data_breakfast_with_coordinates.csv", low_memory=False)
    
    # Clean cost columns safely
    cost_columns = ['Discarded_Cost', 'Subtotal_Cost', 'Left_Over_Cost', 'Production_Cost_Total']
    
    for col in cost_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .replace('nan', '0')
                .astype(float)
            )
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate additional metrics
    df['Total_Waste_Cost'] = df['Left_Over_Cost'] + df['Discarded_Cost']
    df['Planned_Cost'] = df['Planned_Total'] * (df['Subtotal_Cost'] / df['Served_Total'].replace(0, 1))
    df['Cost_Deviation'] = df['Production_Cost_Total'] - df['Planned_Cost']
    
    # Process dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Day_of_Week'] = df['Date'].dt.day_name().fillna('Unknown')
    
    return df

# Load data
try:
    df = load_data()
    st.success(f"Data loaded successfully! {len(df)} records found.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar for visualization selection
st.sidebar.header("📊 Select Visualization")

visualization_options = [
    "Total Cost by School",
    "Cost Over Time by Menu Item", 
    "Cost Distribution: Schools and Menu Items",
    "Cost Distribution: Schools and Top 5 Menu Items",
    "Top 10 Schools by Food Waste Cost",
    "Top 10 Wasted Menu Items",
    "Cost Deviation by School",
    "Popularity vs. Waste by Menu Item",
    "Average Food Waste by Day of Week",
    "Cost per Student by Region",
    "Geographic Distribution of Costs and Waste",
    "Interactive School Map with Layers"
]

selected_viz = st.sidebar.selectbox(
    "Choose a visualization:",
    visualization_options
)

# Display selected visualization
if selected_viz == "Total Cost by School":
    st.header("📈 Total Cost by School")
    school_costs = df.groupby('School Name')['Production_Cost_Total'].sum().reset_index()
    fig = px.bar(
        school_costs,
        x='School Name',
        y='Production_Cost_Total',
        title='Total Cost by School',
        labels={'School Name': 'School', 'Production_Cost_Total': 'Total Cost ($)'}
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Cost Over Time by Menu Item":
    st.header("📅 Cost Over Time by Menu Item")
    time_item_costs = df.groupby(['Date', 'Name'])['Production_Cost_Total'].sum().reset_index()
    fig = px.line(
        time_item_costs,
        x='Date',
        y='Production_Cost_Total',
        color='Name',
        title='Cost Over Time by Menu Item',
        labels={'Date': 'Date', 'Production_Cost_Total': 'Total Cost ($)', 'Name': 'Menu Item'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Cost Distribution: Schools and Menu Items":
    st.header("🌞 Cost Distribution: Schools and Menu Items")
    school_item_costs = df.groupby(['School Name', 'Name'])['Production_Cost_Total'].sum().reset_index()
    fig = px.sunburst(
        school_item_costs,
        path=['School Name', 'Name'],
        values='Production_Cost_Total',
        title='Cost Distribution: Schools and Menu Items'
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Cost Distribution: Schools and Top 5 Menu Items":
    st.header("🏆 Cost Distribution: Schools and Top 5 Menu Items")
    top_items = df.groupby('Name')['Production_Cost_Total'].sum().nlargest(5).index
    filtered_df = df[df['Name'].isin(top_items)]
    fig = px.sunburst(
        filtered_df,
        path=['School Name', 'Name'],
        values='Production_Cost_Total',
        title='Cost Distribution: Schools and Top 5 Menu Items (Overall)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Top 10 Schools by Food Waste Cost":
    st.header("🗑️ Top 10 Schools by Food Waste Cost")
    waste_by_school = df.groupby('School Name')['Total_Waste_Cost'].sum().nlargest(10)
    fig = px.bar(waste_by_school, title='Top 10 Schools by Food Waste Cost')
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Top 10 Wasted Menu Items":
    st.header("🍽️ Top 10 Wasted Menu Items")
    waste_by_item = df.groupby('Name')['Total_Waste_Cost'].sum().nlargest(10)
    fig = px.bar(waste_by_item, title='Top 10 Wasted Menu Items')
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Cost Deviation by School":
    st.header("💰 Cost Deviation by School")
    fig = px.box(df, x='School Name', y='Cost_Deviation', title='Cost Deviation by School')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Popularity vs. Waste by Menu Item":
    st.header("📊 Popularity vs. Waste by Menu Item")
    item_stats = df.groupby('Name').agg({
        'Served_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()
    
    fig = px.scatter(
        item_stats,
        x='Served_Total',
        y='Total_Waste_Cost',
        size='Total_Waste_Cost',
        hover_name='Name',
        title='Popularity vs. Waste by Menu Item'
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Average Food Waste by Day of Week":
    st.header("📅 Average Food Waste by Day of Week")
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
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_viz == "Cost per Student by Region":
    st.header("🏫 Cost per Student by Region")
    if 'FCPS Region' in df.columns:
        region_cost = df.groupby('FCPS Region').agg({
            'Production_Cost_Total': 'sum',
            'Served_Total': 'sum'
        }).reset_index()
        region_cost['Cost_Per_Student'] = region_cost['Production_Cost_Total'] / region_cost['Served_Total']
        
        fig = px.bar(
            region_cost,
            x='FCPS Region',
            y='Cost_Per_Student',
            title='Cost per Student by Region'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("FCPS Region column not found in data")
        
elif selected_viz == "Geographic Distribution of Costs and Waste":
    st.header("🗺️ Geographic Distribution of Costs and Waste")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        school_geo = df.groupby(['School Name', 'latitude', 'longitude']).agg({
            'Production_Cost_Total': 'sum',
            'Total_Waste_Cost': 'sum'
        }).reset_index()
        
        fig = px.scatter_map(
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
        fig.update_geos(fitbounds="locations")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Latitude/Longitude columns not found in data")
        
elif selected_viz == "Interactive School Map with Layers":
    st.header("🗺️ Interactive School Map with Layers")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        school_geo = df.groupby(['School Name', 'latitude', 'longitude']).agg({
            'Production_Cost_Total': 'sum',
            'Total_Waste_Cost': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        # Add cost layer
        fig.add_trace(go.Scattermapbox(
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
        fig.add_trace(go.Scattermapbox(
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
                center={"lat": school_geo['latitude'].mean(), "lon": school_geo['longitude'].mean()},
                zoom=10
            ),
            title="Interactive School Breakfast Program Map",
            updatemenus=[{
                "buttons": [
                    {"method": "update", "args": [{"visible": [True, False]}], "label": "Show Costs"},
                    {"method": "update", "args": [{"visible": [False, True]}], "label": "Show Waste"},
                    {"method": "update", "args": [{"visible": [True, True]}], "label": "Show Both"}
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.02,
                "yanchor": "top"
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Latitude/Longitude columns not found in data")

# Add summary statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.header("📋 Summary Statistics")

if not df.empty:
    st.sidebar.metric("Total Schools", df['School Name'].nunique())
    st.sidebar.metric("Total Menu Items", df['Name'].nunique())
    st.sidebar.metric("Total Production Cost", f"${df['Production_Cost_Total'].sum():,.2f}")
    st.sidebar.metric("Total Waste Cost", f"${df['Total_Waste_Cost'].sum():,.2f}")
    st.sidebar.metric("Average Daily Servings", f"{df['Served_Total'].mean():,.0f}")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for School Breakfast Program Analysis*")

# os.chdir("../../demo")