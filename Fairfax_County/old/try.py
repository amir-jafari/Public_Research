import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="School Meal Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# Load data from local path - replace with your actual file path
df = pd.read_csv('data/breakfast_sales_apr_2025.csv')  # Replace with your actual file path

# App title
st.title("🍽️ School Meal Distribution Dashboard")
st.markdown("---")

# Aggregate meal data by school
school_meal_data = df.groupby('school_name')[['free_meals', 'reduced_price_meals', 'full_price_meals', 'adults']].sum().reset_index()

# Melt the data to long format for plotting
melted_data = school_meal_data.melt(
    id_vars=['school_name'], 
    value_vars=['free_meals', 'reduced_price_meals', 'full_price_meals', 'adults'],
    var_name='meal_category', 
    value_name='count'
)

# Create bar chart with meal categories on x-axis and schools as legend
fig = px.bar(
    melted_data,
    x='meal_category',
    y='count',
    color='school_name',
    title="Meal Distribution by Category and School",
    labels={'count': 'Number of Meals', 'meal_category': 'Meal Category', 'school_name': 'School Name'},
    barmode='group'
)

fig.update_layout(
    xaxis_title="Meal Category",
    yaxis_title="Number of Meals",
    height=600,
    showlegend=True,
    legend=dict(
        title="School Name (Click to toggle)",
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

# Update x-axis labels to be more readable
fig.update_xaxes(tickangle=0)

st.plotly_chart(fig, use_container_width=True)

#%%

import pandas as pd

# List of dataframe names (these will be the CSV filenames)
df_names = ['breakfast_sales_mar_2025', 'breakfast_sales_apr_2025', 'breakfast_sales_may_2025', 
            'lunch_sales_mar_2025', 'lunch_sales_apr_2025', 'lunch_sales_may_2025']

# Load dataframes from CSV files
dfs = [pd.read_csv(f'data/{name}.csv') for name in df_names]

# Vertically stack all dataframes
stacked_df = pd.concat(dfs, ignore_index=True)

# Save to CSV
stacked_df.to_csv('sales.csv', index=False)

print(f"Successfully stacked {len(dfs)} dataframes and saved to sales.csv")
print(f"Final dataframe shape: {stacked_df.shape}")
# %%
import pandas as pd

df = pd.read_csv('sales.csv')
# %%
