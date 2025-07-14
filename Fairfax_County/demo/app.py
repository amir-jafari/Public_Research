# streamlit run app.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Meal Data Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🍽️ School Meal Data Dashboard")
st.markdown("Interactive visualization of daily meal counts by category and school")

df = pd.read_csv('../../data/preprocessed-data/sales.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Define meal columns
meal_columns = ['free_meals', 'reduced_price_meals', 'full_price_meals', 'adults']

# Sidebar filters
st.sidebar.header("📊 Filters")

# School filter
schools_list = ['All Schools'] + sorted(df['school_name'].unique().tolist())
selected_school = st.sidebar.selectbox(
    "🏫 Select School:",
    options=schools_list,
    index=0
)

# Meal category filter
st.sidebar.subheader("🍽️ Meal Categories")
meal_category_options = {
    'Free Meals': 'free_meals',
    'Reduced Price Meals': 'reduced_price_meals', 
    'Full Price Meals': 'full_price_meals',
    'Adults': 'adults'
}

selected_categories = []
for display_name, column_name in meal_category_options.items():
    if st.sidebar.checkbox(display_name, value=True):
        selected_categories.append(column_name)

# Date range filter
st.sidebar.subheader("📅 Date Range")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Time aggregation filter
st.sidebar.subheader("📅 Time Aggregation")
time_aggregation = st.sidebar.selectbox(
    "Group Data By:",
    options=['Date', 'Day of Week', 'Month'],
    index=0
)

# Chart type selector
chart_type = st.sidebar.radio(
    "📈 Chart Type:",
    options=['Grouped Bars', 'Stacked Bars'],
    index=0
)

# Apply filters
def filter_data(df, school, categories, date_range, time_agg):
    # Filter by school
    if school != 'All Schools':
        df_filtered = df[df['school_name'] == school].copy()
    else:
        df_filtered = df.copy()
    
    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['date'].dt.date >= start_date) & 
            (df_filtered['date'].dt.date <= end_date)
        ]
    
    # Group by date and sum meal categories
    if categories:
        daily_totals = df_filtered.groupby('date')[categories].sum().reset_index()
        
        # Apply time aggregation
        if time_agg == 'Day of Week':
            daily_totals['day_of_week'] = daily_totals['date'].dt.day_name()
            # Group by day of week and sum
            aggregated = daily_totals.groupby('day_of_week')[categories].sum().reset_index()
            # Order by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            aggregated['day_of_week'] = pd.Categorical(aggregated['day_of_week'], categories=day_order, ordered=True)
            aggregated = aggregated.sort_values('day_of_week')
            time_column = 'day_of_week'
            
        elif time_agg == 'Month':
            daily_totals['month'] = daily_totals['date'].dt.to_period('M').astype(str)
            # Group by month and sum
            aggregated = daily_totals.groupby('month')[categories].sum().reset_index()
            aggregated = aggregated.sort_values('month')
            time_column = 'month'
            
        else:  # Date
            aggregated = daily_totals
            time_column = 'date'
        
        # Melt the data
        df_melted = pd.melt(aggregated, 
                           id_vars=[time_column],
                           value_vars=categories,
                           var_name='meal_category',
                           value_name='count')
        
        # Clean up category names for display
        df_melted['meal_category'] = df_melted['meal_category'].map({
            'free_meals': 'Free Meals',
            'reduced_price_meals': 'Reduced Price Meals',
            'full_price_meals': 'Full Price Meals',
            'adults': 'Adults'
        })
        
        # Rename time column for consistency
        df_melted = df_melted.rename(columns={time_column: 'time_period'})
        
        return df_melted, time_column
    else:
        return pd.DataFrame(), 'date'

# Filter the data
filtered_data, time_col = filter_data(df, selected_school, selected_categories, date_range, time_aggregation)

# Main content area
if not filtered_data.empty:
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_meals = filtered_data['count'].sum()
        st.metric("Total Meals", f"{total_meals:,}")
    
    with col2:
        if time_aggregation == 'Date':
            avg_daily = filtered_data.groupby('time_period')['count'].sum().mean()
            st.metric("Avg Daily Meals", f"{avg_daily:.1f}")
        elif time_aggregation == 'Day of Week':
            avg_daily = filtered_data.groupby('time_period')['count'].sum().mean()
            st.metric("Avg per Day of Week", f"{avg_daily:.1f}")
        else:  # Month
            avg_daily = filtered_data.groupby('time_period')['count'].sum().mean()
            st.metric("Avg per Month", f"{avg_daily:.1f}")
    
    with col3:
        if time_aggregation == 'Date':
            max_daily = filtered_data.groupby('time_period')['count'].sum().max()
            st.metric("Max Daily Meals", f"{max_daily:,}")
        elif time_aggregation == 'Day of Week':
            max_daily = filtered_data.groupby('time_period')['count'].sum().max()
            st.metric("Max Day of Week", f"{max_daily:,}")
        else:  # Month
            max_daily = filtered_data.groupby('time_period')['count'].sum().max()
            st.metric("Max Monthly", f"{max_daily:,}")
    
    with col4:
        if time_aggregation == 'Date':
            unique_periods = filtered_data['time_period'].nunique()
            st.metric("Days with Data", f"{unique_periods}")
        elif time_aggregation == 'Day of Week':
            unique_periods = filtered_data['time_period'].nunique()
            st.metric("Days of Week", f"{unique_periods}")
        else:  # Month
            unique_periods = filtered_data['time_period'].nunique()
            st.metric("Months", f"{unique_periods}")
    
    # Create the chart
    st.subheader(f"📊 {time_aggregation} Meal Counts")
    
    # Determine chart mode
    bar_mode = 'group' if chart_type == 'Grouped Bars' else 'stack'
    
    # Set appropriate x-axis title
    x_axis_title = time_aggregation
    if time_aggregation == 'Day of Week':
        x_axis_title = 'Day of Week'
    elif time_aggregation == 'Month':
        x_axis_title = 'Month'
    
    # Create the plot
    fig = px.bar(
        filtered_data,
        x='time_period',
        y='count',
        color='meal_category',
        title=f"{time_aggregation} Meal Counts - {selected_school}",
        labels={'count': 'Number of Meals', 'time_period': x_axis_title},
        hover_data={'time_period': True, 'meal_category': True, 'count': True},
        barmode=bar_mode,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title="Number of Meals",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Special formatting for day of week
    if time_aggregation == 'Day of Week':
        fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        # Show aggregated data by time period
        summary_data = filtered_data.pivot(index='time_period', columns='meal_category', values='count').fillna(0)
        summary_data['Total'] = summary_data.sum(axis=1)
        
        # Format the display based on time aggregation
        if time_aggregation == 'Day of Week':
            summary_data = summary_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        st.dataframe(summary_data, use_container_width=True)
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"meal_data_{selected_school}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit and Plotly*")