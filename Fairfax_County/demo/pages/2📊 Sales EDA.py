# pages/1_📊_Sales_EDA.py
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Sales EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Sales EDA")
st.markdown("Exploratory Data Analysis of FCPS meal sales data.")

df = pd.read_csv('../data/preprocessed-data/sales.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

df['school_name'] = df['school_name'].str.replace('_', ' ').str.title()

# Define meal columns
meal_columns = ['free_meals', 'reduced_price_meals', 'full_price_meals', 'adults']

# Calculate school statistics for sorting
def calculate_school_stats(df):
    # Group by school and calculate metrics
    school_stats = df.groupby('school_name')[meal_columns].sum().reset_index()
    
    # Calculate total meals per school
    school_stats['total_meals'] = school_stats[meal_columns].sum(axis=1)
    
    # Calculate proportion of total sales
    total_system_meals = school_stats['total_meals'].sum()
    school_stats['sales_proportion'] = (school_stats['total_meals'] / total_system_meals) * 100
    
    # Calculate daily averages and variation
    daily_totals = df.groupby(['school_name', 'date'])[meal_columns].sum().reset_index()
    daily_totals['daily_total'] = daily_totals[meal_columns].sum(axis=1)
    
    # Calculate standard deviation for each school (replacing CV)
    daily_variation = daily_totals.groupby('school_name')['daily_total'].agg(['mean', 'std']).reset_index()
    daily_variation['std_dev'] = daily_variation['std'].fillna(0)  # Handle cases where std is 0
    
    # Merge variation data
    school_stats = school_stats.merge(daily_variation[['school_name', 'std_dev']], on='school_name')
    
    # Set school_name as index for easier lookup
    school_stats = school_stats.set_index('school_name')
    
    return school_stats

# Calculate school statistics
school_stats = calculate_school_stats(df)

# School sorting options (updated to use standard deviation)
sort_options = {
    'Alphabetical': 'alphabetical',
    'Highest Sales Volume': 'volume_desc',
    'Lowest Sales Volume': 'volume_asc',
    'Highest Sales Proportion %': 'proportion_desc',
    'Lowest Sales Proportion %': 'proportion_asc',
    'Highest Variation (Std Dev)': 'variation_desc',
    'Lowest Variation (Std Dev)': 'variation_asc'
}

selected_sort = st.sidebar.selectbox(
    "🔢 Sort Schools By:",
    options=list(sort_options.keys()),
    index=0
)

# Apply sorting to schools list
def sort_schools(school_stats, sort_method):
    if sort_method == 'alphabetical':
        return sorted(school_stats.index.tolist())
    elif sort_method == 'volume_desc':
        return school_stats.sort_values('total_meals', ascending=False).index.tolist()
    elif sort_method == 'volume_asc':
        return school_stats.sort_values('total_meals', ascending=True).index.tolist()
    elif sort_method == 'proportion_desc':
        return school_stats.sort_values('sales_proportion', ascending=False).index.tolist()
    elif sort_method == 'proportion_asc':
        return school_stats.sort_values('sales_proportion', ascending=True).index.tolist()
    elif sort_method == 'variation_desc':
        return school_stats.sort_values('std_dev', ascending=False).index.tolist()
    elif sort_method == 'variation_asc':
        return school_stats.sort_values('std_dev', ascending=True).index.tolist()
    else:
        return sorted(school_stats.index.tolist())

# Get sorted schools list
sorted_schools = sort_schools(school_stats, sort_options[selected_sort])
schools_list = ['All Schools'] + sorted_schools

# School filter with enhanced display (updated to show standard deviation)
selected_school = st.sidebar.selectbox(
    "🏫 Select School:",
    options=schools_list,
    index=0,
    format_func=lambda x: x if x == 'All Schools' else f"{x} ({school_stats.loc[x, 'sales_proportion']:.1f}% | Std Dev: {school_stats.loc[x, 'std_dev']:.1f})" if x in school_stats.index else x
)

# Show school statistics in sidebar (updated to show standard deviation)
if selected_school != 'All Schools' and selected_school in school_stats.index:
    st.sidebar.markdown("### 📈 School Statistics")
    stats = school_stats.loc[selected_school]
    st.sidebar.metric("Total Meals", f"{int(stats['total_meals']):,}")
    st.sidebar.metric("Sales Proportion", f"{stats['sales_proportion']:.2f}%")
    st.sidebar.metric("Daily Variation (Std Dev)", f"{stats['std_dev']:.1f}")

# Time of day filter
time_of_day_options = ['All Times'] + sorted(df['time_of_day'].unique().tolist())
selected_time_of_day = st.sidebar.selectbox(
    "🍽️ Select Meal Time:",
    options=time_of_day_options,
    index=0
)

# Meal category filter (PRESET: All except adults)
st.sidebar.markdown("🍽️ Select Meal Categories:")
meal_category_options = {
    'Free Meals': 'free_meals',
    'Reduced Price Meals': 'reduced_price_meals', 
    'Full Price Meals': 'full_price_meals',
    'Adults': 'adults'
}

selected_categories = []
for display_name, column_name in meal_category_options.items():
    # Preset: All except adults selected
    default_value = column_name != 'adults'
    if st.sidebar.checkbox(display_name, value=default_value):
        selected_categories.append(column_name)

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "📅 Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Time aggregation filter (PRESET: Day of Week)
time_aggregation = st.sidebar.selectbox(
    "📅 Group Dates By:",
    options=['Date', 'Day of Week', 'Month'],
    index=1  # Default to 'Day of Week'
)

# Chart type selector (PRESET: Stacked Bars)
chart_type = st.sidebar.radio(
    "📈 Chart Type:",
    options=['Grouped Bars', 'Stacked Bars'],
    index=1  # Default to 'Stacked Bars'
)

# Apply filters
def filter_data(df, school, time_of_day, categories, date_range, time_agg):
    # Filter by school
    if school != 'All Schools':
        df_filtered = df[df['school_name'] == school].copy()
    else:
        df_filtered = df.copy()
    
    # Filter by time of day
    if time_of_day != 'All Times':
        df_filtered = df_filtered[df_filtered['time_of_day'] == time_of_day]
    
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
            daily_totals['month'] = daily_totals['date'].dt.to_period('M')
            daily_totals['month_name'] = daily_totals['date'].dt.strftime('%B %Y')
            # Group by month and sum
            aggregated = daily_totals.groupby(['month', 'month_name'])[categories].sum().reset_index()
            aggregated = aggregated.sort_values('month')
            # Use month_name for display
            aggregated = aggregated.drop('month', axis=1)
            time_column = 'month_name'
            
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
        
        return df_melted, time_column, aggregated
    else:
        return pd.DataFrame(), 'date', pd.DataFrame()

# Filter the data
filtered_data, time_col, aggregated_data = filter_data(df, selected_school, selected_time_of_day, selected_categories, date_range, time_aggregation)

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
    time_filter_text = f" - {selected_time_of_day}" if selected_time_of_day != 'All Times' else ""
    st.subheader(f"📊 {time_aggregation} Meal Counts{time_filter_text}")
    
    # Determine chart mode
    bar_mode = 'group' if chart_type == 'Grouped Bars' else 'stack'
    
    # Set appropriate x-axis title
    x_axis_title = time_aggregation
    if time_aggregation == 'Day of Week':
        x_axis_title = 'Day of Week'
    elif time_aggregation == 'Month':
        x_axis_title = 'Month'
    
    # Calculate averages for each meal category to include in legend
    category_averages = {}
    for category in selected_categories:
        # Map back to original column names for calculation
        original_col = category
        if not aggregated_data.empty:
            avg_value = aggregated_data[original_col].mean()
            category_averages[category] = avg_value
    
    # Create a copy of filtered_data with updated meal category names that include averages
    filtered_data_with_avg = filtered_data.copy()
    
    # Update meal category names to include averages
    def update_category_name(row):
        category = row['meal_category']
        # Map display names back to original column names
        reverse_mapping = {
            'Free Meals': 'free_meals',
            'Reduced Price Meals': 'reduced_price_meals',
            'Full Price Meals': 'full_price_meals',
            'Adults': 'adults'
        }
        
        original_col = reverse_mapping.get(category, category)
        if original_col in category_averages:
            avg = category_averages[original_col]
            return f"{category} (Avg: {avg:.1f})"
        return category
    
    filtered_data_with_avg['meal_category'] = filtered_data_with_avg.apply(update_category_name, axis=1)
    
    # Create the plot
    chart_title = f"{time_aggregation} Meal Counts - {selected_school}{time_filter_text}"
    fig = px.bar(
        filtered_data_with_avg,
        x='time_period',
        y='count',
        color='meal_category',
        title=chart_title,
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
    
    # Special formatting for day of week and month
    if time_aggregation == 'Day of Week':
        fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    elif time_aggregation == 'Month':
        # Sort months chronologically by extracting the date from month names
        month_order = sorted(filtered_data['time_period'].unique(), 
                           key=lambda x: pd.to_datetime(x, format='%B %Y'))
        fig.update_xaxes(categoryorder='array', categoryarray=month_order)
    
    # Fix x-axis labels for Date visualization
    if time_aggregation == 'Date':
        # Format dates for better readability - remove year and show only dates with data
        fig.update_xaxes(
            tickformat='%m-%d',
            tickangle=45,
            tickmode='array',
            tickvals=filtered_data['time_period'].unique()
        )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # FIXED: Show top 5 schools info - now properly updates based on selected sort
    with st.expander("🏆 Top 5 Schools by Selected Metric"):
        # Get the current sort method
        current_sort_method = sort_options[selected_sort]
        
        # Determine which schools to show based on the sorting method
        if current_sort_method in ['volume_desc', 'proportion_desc', 'variation_desc']:
            # For descending sorts, show the first 5 (highest values)
            top_schools = sorted_schools[:5]
        elif current_sort_method in ['volume_asc', 'proportion_asc', 'variation_asc']:
            # For ascending sorts, show the first 5 (lowest values)
            top_schools = sorted_schools[:5]
        else:  # alphabetical
            # For alphabetical, show first 5 alphabetically
            top_schools = sorted_schools[:5]
        
        # Display the appropriate metric based on selected sort
        for i, school in enumerate(top_schools, 1):
            if school in school_stats.index:
                stats = school_stats.loc[school]
                
                # Determine what value to show based on the selected sort
                if 'Volume' in selected_sort:
                    value = f"{int(stats['total_meals']):,} meals"
                elif 'Proportion' in selected_sort:
                    value = f"{stats['sales_proportion']:.1f}%"
                elif 'Variation' in selected_sort:
                    value = f"{stats['std_dev']:.1f} std dev"
                else:  # Alphabetical
                    value = f"{int(stats['total_meals']):,} meals"
                
                st.write(f"{i}. **{school}** - {value}")
    
    # Additional expandable section for school comparison (updated column names)
    if selected_school == 'All Schools':
        with st.expander("🏫 School Comparison Table"):
            comparison_df = school_stats[['total_meals', 'sales_proportion', 'std_dev']].copy()
            comparison_df.columns = ['Total Meals', 'Sales Proportion (%)', 'Daily Variation (Std Dev)']
            comparison_df = comparison_df.sort_values('Total Meals', ascending=False)
            
            # Format the dataframe for better display
            comparison_df['Total Meals'] = comparison_df['Total Meals'].apply(lambda x: f"{int(x):,}")
            comparison_df['Sales Proportion (%)'] = comparison_df['Sales Proportion (%)'].apply(lambda x: f"{x:.2f}%")
            comparison_df['Daily Variation (Std Dev)'] = comparison_df['Daily Variation (Std Dev)'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(comparison_df, use_container_width=True)
    
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
    filename_time_filter = f"_{selected_time_of_day.lower()}" if selected_time_of_day != 'All Times' else ""
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"meal_data_{selected_school}{filename_time_filter}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created using Streamlit and Plotly*")
st.markdown("Author: Tyler Wallett.")