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
    page_title="Production EDA Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Production EDA Dashboard")
st.markdown("Exploratory Data Analysis of FCPS meal production data.")

# Meal type selection at the top
meal_type = st.sidebar.radio(
    "🍽️ Select Meal Program",
    options=["Breakfast 🍳", "Lunch 🥪"],
    index=0,
    horizontal=True
)


# Load data function with meal type selection
@st.cache_data
def load_data(meal_program):
    # Determine file path based on meal type
    file_path = {
        "Breakfast 🍳": "../data/preprocessed-data/data_breakfast_with_coordinates.csv",
        "Lunch 🥪": "../data/preprocessed-data/data_lunch_with_coordinates.csv"
    }[meal_program]

    try:
        df = pd.read_csv(file_path, low_memory=False)

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
    except Exception as e:
        st.error(f"Error loading {meal_program} data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# Load data based on selected meal type
df = load_data(meal_type)

if df.empty:
    st.error("No data available. Please check your data files.")
    st.stop()

# Display success message with meal type
# st.success(f"{meal_type} data loaded successfully! {len(df)} records found.")

visualization_options = [
    "Total Cost by School",
    "Cost Over Time by Menu Item",
    "Cost Distribution: Schools and Menu Items",
    "Top Schools by Food Waste Cost",
    "Top Wasted Menu Items",
    "Cost Deviation by School",
    "Popularity vs. Waste by Menu Item",
    "Average Food Waste by Day of Week",
    "Cost per Student by Region",
    "Geographic Distribution of Costs and Waste",
    "Interactive School Map with Layers",
    "Enhanced School Region Map"
]

selected_viz = st.sidebar.selectbox(
    "📈 Choose visualization:",
    options=visualization_options
)

# School filter
all_schools = ['All Schools'] + sorted(df['School Name'].unique().tolist())
selected_school = st.sidebar.selectbox(
    "🏫 Select School",
    options=all_schools,
    index=0
)

# Menu item filter (multi-select)
menu_items = st.sidebar.multiselect(
    "🍽️ Menu Items",
    options=sorted(df['Name'].unique()),
    default=[]
)

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Main content area
st.header(selected_viz)  # Dynamic title based on selection

# Apply filters to data
def apply_filters(df, school, date_range, items):
    filtered = df.copy()

    if school != 'All Schools':
        filtered = filtered[filtered['School Name'] == school]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['Date'] >= pd.to_datetime(start_date)) &
            (filtered['Date'] <= pd.to_datetime(end_date))
            ]

    if items:
        filtered = filtered[filtered['Name'].isin(items)]

    return filtered


filtered_df = apply_filters(df, selected_school, date_range, menu_items)

# ---------------------------------------------------------------------------------------------------------

if selected_viz == "Total Cost by School":
    # Comparison toggle
    compare_option = st.sidebar.radio(
        "Meal View",
        options=["Current Meal Only", "Compare Breakfast vs. Lunch"],
        index=0,
        horizontal=True,
        key="cost_school_compare_toggle"
    )
    compare_meals = (compare_option == "Compare Breakfast vs. Lunch")

    # Load data accordingly
    if compare_meals:
        df_breakfast = load_data("Breakfast 🍳")
        df_lunch = load_data("Lunch 🥪")
        df_breakfast["Meal"] = "Breakfast"
        df_lunch["Meal"] = "Lunch"
        combined_df = pd.concat([df_breakfast, df_lunch], ignore_index=True)
        summary_df = apply_filters(combined_df, selected_school, date_range, menu_items)
    else:
        summary_df = apply_filters(df, selected_school, date_range, menu_items)
        summary_df["Meal"] = meal_type.split()[0]  # 'Breakfast' or 'Lunch'

    school_summary = summary_df.groupby(['School Name', 'Meal'], observed=True)['Production_Cost_Total'].sum().reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Displayed Schools", school_summary['School Name'].nunique())
    with col2:
        st.metric("Menu Items Used", summary_df['Name'].nunique())
    with col3:
        total_cost_shown = school_summary['Production_Cost_Total'].sum()
        st.metric("Total Production Cost (Shown)", f"${total_cost_shown:,.2f}")

    # Only show individual meal totals when comparing
    if compare_meals:
        meal_totals = school_summary.groupby("Meal")["Production_Cost_Total"].sum().to_dict()

        st.markdown("**Breakdown by Meal:**")
        colA, colB = st.columns(len(meal_totals))
        for idx, (meal, cost) in enumerate(meal_totals.items()):
            with [colA, colB][idx]:
                st.metric(f"{meal} Total", f"${cost:,.2f}")

    # Sidebar display settings
    total_costs_by_school = (
        school_summary.groupby('School Name')['Production_Cost_Total']
        .sum()
        .reset_index()
    )
    absolute_max = total_costs_by_school['Production_Cost_Total'].max()

    max_schools = st.sidebar.selectbox(
        "Number of schools to display",
        options=[10, 15, 20, 30, 50, 100, "All"],
        index=1,
        key="school_count_selector"
    )

    cost_range = st.sidebar.slider(
        "Cost range ($)",
        min_value=0,
        max_value=int(absolute_max),
        value=(0, int(absolute_max)),
        step=1000,
        key="cost_range_selector"
    )

    # Filter by range
    display_df = school_summary.groupby(['School Name', 'Meal'], observed=True)['Production_Cost_Total'].sum().reset_index()

    total_costs = display_df.groupby('School Name')['Production_Cost_Total'].sum()
    top_schools = total_costs.sort_values(ascending=False)

    if max_schools != "All":
        top_schools = top_schools.head(int(max_schools))

    display_df = display_df[display_df['School Name'].isin(top_schools.index)]

    # Filter by cost range (total per school)
    school_totals = display_df.groupby('School Name')['Production_Cost_Total'].sum()
    schools_in_range = school_totals[(school_totals >= cost_range[0]) & (school_totals <= cost_range[1])].index
    display_df = display_df[display_df['School Name'].isin(schools_in_range)]

    # Plot
    if not display_df.empty:
        fig = px.bar(
            display_df,
            x='School Name',
            y='Production_Cost_Total',
            color='Meal',
            barmode='group',
            labels={'Production_Cost_Total': 'Total Cost ($)'},
            title=f"Schools by Cost (${cost_range[0]:,}-${cost_range[1]:,})" +
                  (f" | Top {max_schools}" if max_schools != "All" else "")
        )

        fig.update_layout(
            height=600,
            width=1200,
            xaxis={'categoryorder': 'total descending'},
            coloraxis_showscale=False
        )

        avg_cost = display_df.groupby('School Name')['Production_Cost_Total'].sum().mean()
        fig.add_hline(
            y=avg_cost,
            line_dash="dot",
            annotation_text=f"Avg: ${avg_cost:,.0f}",
            annotation_position="bottom right"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No schools match the current filters")


# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Cost Over Time by Menu Item":
    # st.header("\U0001F4C5 Cost Over Time by Menu Item")

    # Get all unique menu items sorted by total cost (descending)
    top_menu_items = df.groupby('Name')['Production_Cost_Total'].sum().nlargest(20).index.tolist()

    # Sidebar selection
    st.sidebar.subheader("Menu Item Selection")
    selected_items = st.sidebar.multiselect(
        "Select Menu Items to Display",
        options=top_menu_items,
        default=top_menu_items[:2],
        key="menu_item_selector"
    )

    # Filtered data based on menu item selection
    filtered_data = df[df['Name'].isin(selected_items)] if selected_items else df

    # Summary Metrics (dynamic)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Menu Items Displayed", filtered_data['Name'].nunique())
    with col2:
        st.metric("Total Records", f"{len(filtered_data):,}")
    with col3:
        st.metric("Total Cost (Selected Items)", f"${filtered_data['Production_Cost_Total'].sum():,.2f}")

    # Time-series plot
    time_item_costs = filtered_data.groupby(['Date', 'Name'])['Production_Cost_Total'].sum().reset_index()
    fig = px.line(
        time_item_costs,
        x='Date',
        y='Production_Cost_Total',
        color='Name',
        labels={'Date': 'Date', 'Production_Cost_Total': 'Total Cost ($)', 'Name': 'Menu Item'},
        height=600,
        width=1200
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=50, b=100, t=50, pad=4)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Cost Distribution: Schools and Menu Items":
    # st.header("\U0001F31E Cost Distribution: Schools and Menu Items")

    # 1. GET BASE METRICS FROM FULL DATA
    full_stats = df.groupby(['School Name', 'Name']).agg(
        total_servings=('Served_Total', 'sum'),
        total_cost=('Production_Cost_Total', 'sum'),
        cost_per_serving=('Production_Cost_Total', 'mean')
    ).reset_index()

    # 2. SET UP FILTERS
    st.sidebar.subheader("Primary Filters")

    school_options = ['All Schools', 'Top 10 Schools'] + sorted(df['School Name'].unique().tolist())
    selected_schools = st.sidebar.multiselect(
        "Select Schools",
        options=school_options,
        default=['Top 10 Schools'],
        key="school_selector"
    )

    all_items = sorted(df['Name'].unique())
    excluded_items = st.sidebar.multiselect(
        "Exclude Items",
        options=all_items,
        default=[],
        key="item_excluder"
    )

    st.sidebar.subheader("Cost Filters")

    cost_per_serving = st.sidebar.slider(
        "Cost Per Serving ($)",
        min_value=0.0,
        max_value=float(df['Production_Cost_Total'].max()),
        value=(3.0, 10.0),
        step=0.5,
        key="cost_per_serving"
    )

    # 3. APPLY FILTERS
    viz_df = df.copy()

    if 'date_range' in locals() and len(date_range) == 2:
        start_date, end_date = date_range
        viz_df = viz_df[
            (viz_df['Date'] >= pd.to_datetime(start_date)) &
            (viz_df['Date'] <= pd.to_datetime(end_date))
        ]

    if 'All Schools' not in selected_schools:
        if 'Top 10 Schools' in selected_schools:
            top_schools = viz_df.groupby('School Name')['Production_Cost_Total'].sum().nlargest(10).index
            selected_schools = [s for s in selected_schools if s != 'Top 10 Schools'] + list(top_schools)
        viz_df = viz_df[viz_df['School Name'].isin(selected_schools)]

    if excluded_items:
        viz_df = viz_df[~viz_df['Name'].isin(excluded_items)]

    viz_df = viz_df[
        (viz_df['Production_Cost_Total'] >= cost_per_serving[0]) &
        (viz_df['Production_Cost_Total'] <= cost_per_serving[1])
    ]

    # 4. CREATE VISUALIZATION
    if not viz_df.empty:
        agg_data = viz_df.groupby(['School Name', 'Name']).agg(
            total_cost=('Production_Cost_Total', 'sum'),
            avg_cost=('Production_Cost_Total', 'mean'),
            servings=('Served_Total', 'sum')
        ).reset_index()

        # Summary Metrics (dynamic)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Schools", agg_data['School Name'].nunique())
            st.metric("Total Items", agg_data['Name'].nunique())
        with col2:
            st.metric("Total Cost", f"${agg_data['total_cost'].sum():,.2f}")
            st.metric("Avg Cost/Serving", f"${agg_data['avg_cost'].mean():.2f}")
        with col3:
            st.metric("Total Servings", f"{agg_data['servings'].sum():,}")
            if excluded_items:
                st.metric("Excluded Items", len(excluded_items))

        fig = px.sunburst(
            agg_data,
            path=['School Name', 'Name'],
            values='total_cost',
            color='avg_cost',
            color_continuous_scale='Viridis',
            hover_data=['servings'],
            title=f"Cost Distribution by School and Menu Item<br>" +
                 f"<sup>Cost/Serving: ${cost_per_serving[0]:.2f}-${cost_per_serving[1]:.2f} | " +
                 f"{len(selected_schools)} {'school' if len(selected_schools)==1 else 'schools'} selected</sup>"
        )

        fig.update_layout(
            margin=dict(t=80, l=0, r=0, b=0),
            height=800,
            coloraxis_colorbar=dict(
                title="Avg Cost/Serving",
                tickprefix="$"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    #     with st.expander("\U0001F4CA Detailed View", expanded=False):
    #         st.caption("**Active Filters:**")
    #         filter_text = []
    #         if 'date_range' in locals() and len(date_range) == 2:
    #             filter_text.append(f"Date range: {date_range[0]} to {date_range[1]}")
    #         if 'Top 10 Schools' in selected_schools:
    #             filter_text.append("Showing top 10 schools by cost")
    #         elif len(selected_schools) > 0 and 'All Schools' not in selected_schools:
    #             filter_text.append(f"{len(selected_schools)} schools selected")
    #         if excluded_items:
    #             filter_text.append(f"{len(excluded_items)} items excluded")
    #         filter_text.append(f"Cost/serving: ${cost_per_serving[0]:.2f}-${cost_per_serving[1]:.2f}")
    #         st.write(" | ".join(filter_text))
    # else:
    #     st.warning("No data matches all filter criteria")



# ---------------------------------------------------------------------------------------------------------
    
elif selected_viz == "Top Schools by Food Waste Cost":
    # st.header("\U0001F5D1\ufe0f Top Schools by Food Waste Cost")

    # Add comparison toggle
    compare_option = st.sidebar.radio(
        "Meal View",
        options=["Current Meal Only", "Compare Breakfast vs. Lunch"],
        index=0,
        horizontal=True,
        key="waste_compare_toggle"
    )
    compare_meals = (compare_option == "Compare Breakfast vs. Lunch")

    # Load data accordingly
    if compare_meals:
        df_breakfast = load_data("Breakfast 🍳")
        df_lunch = load_data("Lunch 🥪")
        df_breakfast["Meal"] = "Breakfast"
        df_lunch["Meal"] = "Lunch"
        combined_df = pd.concat([df_breakfast, df_lunch], ignore_index=True)
        filtered_df = apply_filters(combined_df, selected_school, date_range, menu_items)
    else:
        filtered_df = apply_filters(df, selected_school, date_range, menu_items)
        filtered_df["Meal"] = meal_type.split()[0]  # 'Breakfast' or 'Lunch'

    with st.sidebar:
        st.subheader("Display Options")

        waste_min = float(filtered_df['Total_Waste_Cost'].min())
        waste_max = float(
            filtered_df.groupby(['School Name'])['Total_Waste_Cost'].sum().max()
        )

        waste_range = st.slider(
            "Filter by Total Waste Cost ($)",
            min_value=0.0,
            max_value=waste_max,
            value=(0.0, waste_max),
            step=100.0,
            key="waste_cost_range"
        )

        num_schools = st.selectbox(
            "Number of schools to display",
            options=[10, 20, 30, 50, "All"],
            index=0,
            key="waste_school_count"
        )

    # Aggregate food waste cost by school and meal
    waste_by_school = (
        filtered_df.groupby(['School Name', 'Meal'], observed=True)['Total_Waste_Cost']
        .sum()
        .reset_index()
    )

    # Filter by range
    waste_by_school = waste_by_school[
        (waste_by_school['Total_Waste_Cost'] >= waste_range[0]) &
        (waste_by_school['Total_Waste_Cost'] <= waste_range[1])
    ]

    # Determine top N schools overall (summing across meals)
    top_schools = (
        waste_by_school.groupby('School Name')['Total_Waste_Cost']
        .sum()
        .sort_values(ascending=False)
    )

    if num_schools != "All":
        top_schools = top_schools.head(int(num_schools))

    display_waste = waste_by_school[
        waste_by_school['School Name'].isin(top_schools.index)
    ]

    if display_waste.empty:
        st.warning("No schools match the selected filters.")
        st.stop()

    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Displayed Schools", display_waste['School Name'].nunique())
    with col2:
        st.metric("Total Waste Cost (Shown)", f"${display_waste['Total_Waste_Cost'].sum():,.2f}")
    with col3:
        st.metric("Meals Compared", display_waste['Meal'].nunique())

    # Plot grouped bars by school and meal
    fig = px.bar(
        display_waste,
        x="School Name",
        y="Total_Waste_Cost",
        color="Meal",
        barmode="group",
        labels={'Total_Waste_Cost': 'Total Waste Cost ($)'},
        title="Top Schools by Food Waste Cost" + (f" (Top {num_schools})" if num_schools != "All" else "")
    )

    fig.update_layout(
        xaxis_title="School Name",
        yaxis_title="Total Waste Cost ($)",
        xaxis_tickangle=45,
        height=600,
        width=1200,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)



# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Top Wasted Menu Items":
    # st.header("🍽️ Top Wasted Menu Items")

    # Sidebar toggle for comparison
    compare_option = st.sidebar.radio(
        "Meal View",
        options=["Current Meal Only", "Compare Breakfast vs. Lunch"],
        index=0,
        horizontal=True,
        key="waste_meal_toggle"
    )
    compare_meals = (compare_option == "Compare Breakfast vs. Lunch")

    # Load and filter data
    if compare_meals:
        df_breakfast = load_data("Breakfast 🍳")
        df_lunch = load_data("Lunch 🥪")
        df_breakfast["Meal"] = "Breakfast"
        df_lunch["Meal"] = "Lunch"
        combined_df = pd.concat([df_breakfast, df_lunch], ignore_index=True)
        filtered_df = apply_filters(combined_df, selected_school, date_range, menu_items)
    else:
        filtered_df = apply_filters(df, selected_school, date_range, menu_items)
        filtered_df["Meal"] = meal_type.split()[0]  # 'Breakfast' or 'Lunch'

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    with st.sidebar:
        st.subheader("Display Options")

        # Max possible items BEFORE grouping
        max_items = filtered_df['Name'].nunique()

        item_waste_max = float(
            filtered_df.groupby('Name')['Total_Waste_Cost'].sum().max()
        )
        item_waste_range = st.slider(
            "Filter by Total Waste Cost ($)",
            min_value=0.0,
            max_value=item_waste_max,
            value=(0.0, item_waste_max),
            step=50.0,
            key="item_waste_cost_range"
        )

        num_items = st.slider(
            "Number of menu items to display",
            min_value=1,
            max_value=max_items,
            value=min(10, max_items),
            step=1,
            key="waste_item_count_slider"
        )

    # Now safe to use filtered_df and define grouped
    grouped = filtered_df.groupby(['Name', 'Meal'], observed=True)['Total_Waste_Cost'].sum().reset_index()
    grouped = grouped[(grouped['Total_Waste_Cost'] >= item_waste_range[0]) &
                      (grouped['Total_Waste_Cost'] <= item_waste_range[1])]

    top_items = (
        grouped.groupby('Name')['Total_Waste_Cost']
        .sum()
        .sort_values(ascending=False)
        .head(num_items)
        .index
    )
    grouped = grouped[grouped['Name'].isin(top_items)]

    # Aggregate and filter
    grouped = filtered_df.groupby(['Name', 'Meal'], observed=True)['Total_Waste_Cost'].sum().reset_index()
    grouped = grouped[(grouped['Total_Waste_Cost'] >= item_waste_range[0]) &
                      (grouped['Total_Waste_Cost'] <= item_waste_range[1])]

    # Determine top items overall
    top_items = (
        grouped.groupby('Name')['Total_Waste_Cost']
        .sum()
        .sort_values(ascending=False)
        .head(None if num_items == "All" else int(num_items))
        .index
    )
    grouped = grouped[grouped['Name'].isin(top_items)]

    if grouped.empty:
        st.warning("No menu items match the selected criteria.")
        st.stop()

    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Displayed Items", grouped['Name'].nunique())
    with col2:
        st.metric("Total Waste Cost (Shown)", f"${grouped['Total_Waste_Cost'].sum():,.2f}")
    with col3:
        st.metric("Meals Compared", grouped['Meal'].nunique())

    # Bar chart
    fig = px.bar(
        grouped,
        x='Name',
        y='Total_Waste_Cost',
        color='Meal',
        barmode='group',
        labels={'Name': 'Menu Item', 'Total_Waste_Cost': 'Total Waste Cost ($)'},
        title="Top Wasted Menu Items (Grouped by Meal)"
    )
    fig.update_layout(
        xaxis_tickangle=45,
        height=600,
        width=1200,
        yaxis_title="Total Waste Cost ($)",
        xaxis_title="Menu Item",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Cost Deviation by School":
    # st.header("💰 Cost Deviation by School")

    # Apply global filters first
    filtered_df = apply_filters(df, selected_school, date_range, menu_items)

    if filtered_df.empty:
        st.warning("No records found for the selected school, menu items, or date range.")
        st.stop()

    with st.sidebar:
        st.subheader("Deviation Filters")

        school_deviation = filtered_df.groupby('School Name')['Cost_Deviation'].mean()
        deviation_min = float(school_deviation.min())
        deviation_max = float(school_deviation.max())

        if deviation_min == deviation_max:
            st.info(f"All schools have the same average cost deviation: ${deviation_min:,.2f}")
            deviation_range = (deviation_min, deviation_max)
        else:
            deviation_range = st.slider(
                "Filter by Avg Cost Deviation ($)",
                min_value=round(deviation_min, 2),
                max_value=round(deviation_max, 2),
                value=(round(deviation_min, 2), round(deviation_max, 2)),
                step=10.0,
                key="deviation_range_selector"
            )

        num_schools = st.selectbox(
            "Number of schools to display",
            options=[10, 20, 30, 50, "All"],
            index=0,
            key="deviation_school_selector"
        )

    # Filter schools based on slider range
    valid_schools = school_deviation[
        (school_deviation >= deviation_range[0]) &
        (school_deviation <= deviation_range[1])
    ]

    if valid_schools.empty:
        st.warning("No schools fall within the selected deviation range.")
        st.stop()

    if num_schools != "All":
        top_schools = valid_schools.abs().sort_values(ascending=False).head(int(num_schools)).index
    else:
        top_schools = valid_schools.index

    plot_df = filtered_df[filtered_df['School Name'].isin(top_schools)]

    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Displayed Schools", plot_df['School Name'].nunique())
    with col2:
        total_positive = plot_df[plot_df['Cost_Deviation'] > 0]['Cost_Deviation'].sum()
        st.metric("Total Overruns", f"${total_positive:,.2f}")
    with col3:
        total_negative = plot_df[plot_df['Cost_Deviation'] < 0]['Cost_Deviation'].sum()
        st.metric("Total Underspends", f"${total_negative:,.2f}")

    # Plot
    fig = px.box(
        plot_df,
        x='School Name',
        y='Cost_Deviation',
        color_discrete_sequence=['#1f77b4'],
        title='Cost Deviation by School (Filtered)'
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        height=600,
        width=1200,
        yaxis_title="Cost Deviation ($)",
        xaxis_title="School Name"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Popularity vs. Waste by Menu Item":
    # st.header("📊 Popularity vs. Waste by Menu Item")

    # Apply global filters
    filtered_df = apply_filters(df, selected_school, date_range, menu_items)

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    # Aggregate stats by menu item
    item_stats = filtered_df.groupby('Name').agg({
        'Served_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()

    # Sidebar filters
    with st.sidebar:
        st.subheader("Item Filters")

        max_served = int(item_stats['Served_Total'].max())
        max_waste = float(item_stats['Total_Waste_Cost'].max())

        served_range = st.slider(
            "Servings Range",
            min_value=0,
            max_value=max_served,
            value=(0, max_served),
            step=10,
            key="served_range"
        )

        waste_range = st.slider(
            "Waste Cost Range ($)",
            min_value=0.0,
            max_value=round(max_waste + 1, 2),
            value=(0.0, round(max_waste + 1, 2)),
            step=10.0,
            key="waste_range"
        )

        item_limit = st.selectbox(
            "Number of items to display",
            options=[10, 20, 30, 50, "All"],
            index=0,
            key="item_limit"
        )

    # Filter data
    item_stats = item_stats[
        (item_stats['Served_Total'] >= served_range[0]) &
        (item_stats['Served_Total'] <= served_range[1]) &
        (item_stats['Total_Waste_Cost'] >= waste_range[0]) &
        (item_stats['Total_Waste_Cost'] <= waste_range[1])
    ]

    if item_stats.empty:
        st.warning("No menu items match the selected ranges.")
        st.stop()

    # Limit items if needed
    if item_limit != "All":
        item_stats = item_stats.sort_values('Total_Waste_Cost', ascending=False).head(int(item_limit))

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Items Displayed", item_stats['Name'].nunique())
    with col2:
        st.metric("Total Servings", f"{item_stats['Served_Total'].sum():,}")
    with col3:
        st.metric("Total Waste Cost", f"${item_stats['Total_Waste_Cost'].sum():,.2f}")

    # Plot
    fig = px.scatter(
        item_stats,
        x='Served_Total',
        y='Total_Waste_Cost',
        size='Total_Waste_Cost',
        hover_name='Name',
        title='Popularity vs. Waste by Menu Item',
        color='Total_Waste_Cost',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=600,
        width=1200,
        xaxis_title="Total Servings",
        yaxis_title="Total Waste Cost ($)",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Average Food Waste by Day of Week":
    # st.header("📅 Average Food Waste by Day of Week")

    # Sidebar toggle: radio instead of checkbox
    compare_option = st.sidebar.radio(
        "Meal View",
        options=["Current Meal Only", "Compare Breakfast vs. Lunch"],
        index=0,
        horizontal=True
    )
    compare_meals = (compare_option == "Compare Breakfast vs. Lunch")

    # Load and filter data
    if compare_meals:
        df_breakfast = load_data("Breakfast 🍳")
        df_lunch = load_data("Lunch 🥪")
        df_breakfast['Meal'] = 'Breakfast'
        df_lunch['Meal'] = 'Lunch'
        combined_df = pd.concat([df_breakfast, df_lunch], ignore_index=True)
        filtered_df = apply_filters(combined_df, selected_school, date_range, menu_items)
    else:
        filtered_df = apply_filters(df, selected_school, date_range, menu_items)
        filtered_df['Meal'] = meal_type.split()[0]  # 'Breakfast' or 'Lunch'

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    # Order of weekdays
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(day_order)]

    # Grouped data for plotting
    waste_by_day_meal = (
        filtered_df.groupby(['Day_of_Week', 'Meal'], observed=True)['Total_Waste_Cost']
        .mean()
        .reset_index()
    )
    waste_by_day_meal['Day_of_Week'] = pd.Categorical(
        waste_by_day_meal['Day_of_Week'],
        categories=day_order,
        ordered=True
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Waste/Day", f"${waste_by_day_meal['Total_Waste_Cost'].mean():,.2f}")
    with col2:
        max_row = waste_by_day_meal.loc[waste_by_day_meal['Total_Waste_Cost'].idxmax()]
        st.metric(
            label="Highest Avg Day",
            value=f"{max_row['Day_of_Week']} ({max_row['Meal']})",
            delta=f"${max_row['Total_Waste_Cost']:,.2f}"
        )

    with col3:
        min_row = waste_by_day_meal.loc[waste_by_day_meal['Total_Waste_Cost'].idxmin()]
        st.metric(
            label="Lowest Avg Day",
            value=f"{min_row['Day_of_Week']} ({min_row['Meal']})",
            delta=f"${min_row['Total_Waste_Cost']:,.2f}"
        )

    # Plot
    fig = px.bar(
        waste_by_day_meal,
        x='Day_of_Week',
        y='Total_Waste_Cost',
        color='Meal',
        barmode='group',
        title='Average Food Waste by Day of Week',
        labels={'Total_Waste_Cost': 'Avg Waste ($)', 'Day_of_Week': 'Day'},
        color_discrete_map={'Breakfast': '#1f77b4', 'Lunch': '#17becf'}
    )

    fig.update_layout(
        height=500,
        width=1100,
        xaxis_title="Day of Week",
        yaxis_title="Average Waste Cost ($)",
        xaxis=dict(categoryorder='array', categoryarray=day_order)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Cost per Student by Region":
    # st.header("🏫 Cost per Student by Region")

    # Sidebar toggle for comparison
    compare_option = st.sidebar.radio(
        "Meal View",
        options=["Current Meal Only", "Compare Breakfast vs. Lunch"],
        index=0,
        horizontal=True
    )
    compare_meals = (compare_option == "Compare Breakfast vs. Lunch")

    # Load and filter data
    if compare_meals:
        df_breakfast = load_data("Breakfast 🍳")
        df_lunch = load_data("Lunch 🥪")
        df_breakfast["Meal"] = "Breakfast"
        df_lunch["Meal"] = "Lunch"
        combined_df = pd.concat([df_breakfast, df_lunch], ignore_index=True)
        filtered_df = apply_filters(combined_df, selected_school, date_range, menu_items)
    else:
        filtered_df = apply_filters(df, selected_school, date_range, menu_items)
        filtered_df["Meal"] = meal_type.split()[0]  # 'Breakfast' or 'Lunch'

    if "FCPS Region" not in filtered_df.columns:
        st.warning("⚠️ 'FCPS Region' column not found in data.")
        st.stop()

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    # Group and calculate cost per student
    region_cost = (
        filtered_df.groupby(["FCPS Region", "Meal"])
        .agg({"Production_Cost_Total": "sum", "Served_Total": "sum"})
        .reset_index()
    )
    region_cost["Cost_Per_Student"] = region_cost["Production_Cost_Total"] / region_cost["Served_Total"]

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Regions Displayed", region_cost['FCPS Region'].nunique())
    with col2:
        avg_cost = region_cost['Cost_Per_Student'].mean()
        st.metric("Avg Cost/Student", f"${avg_cost:,.2f}")

    col3, col4 = st.columns(2)
    max_row = region_cost.loc[region_cost['Cost_Per_Student'].idxmax()]
    min_row = region_cost.loc[region_cost['Cost_Per_Student'].idxmin()]
    with col3:
        st.metric("Highest Region", f"{max_row['FCPS Region']} ({max_row['Meal']})", delta=f"${max_row['Cost_Per_Student']:,.2f}")
    with col4:
        st.metric("Lowest Region", f"{min_row['FCPS Region']} ({min_row['Meal']})", delta=f"${min_row['Cost_Per_Student']:,.2f}")

    # Plot
    fig = px.bar(
        region_cost,
        x="FCPS Region",
        y="Cost_Per_Student",
        color="Meal",
        barmode="group",  # You can change to 'stack' here if needed
        labels={
            "Cost_Per_Student": "Cost per Student ($)",
            "FCPS Region": "Region",
            "Meal": "Meal"
        },
        color_discrete_map={"Breakfast": "#1f77b4", "Lunch": "#17becf"},
        title="Cost per Student by Region"
    )

    fig.update_layout(
        height=600,
        width=1100,
        xaxis_title="Region",
        yaxis_title="Cost per Student ($)",
        xaxis_tickangle=0,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)



# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Geographic Distribution of Costs and Waste":
    # st.header("🗺️ Geographic Distribution of Costs and Waste")

    # Apply global filters
    filtered_df = apply_filters(df, selected_school, date_range, menu_items)

    required_columns = {'latitude', 'longitude', 'School Name', 'Production_Cost_Total', 'Total_Waste_Cost'}
    if not required_columns.issubset(filtered_df.columns):
        st.warning("⚠️ Required columns (latitude, longitude, costs) not found in data.")
        st.stop()

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    # Group and aggregate by school and location
    school_geo = filtered_df.groupby(['School Name', 'latitude', 'longitude']).agg({
        'Production_Cost_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()

    # Add sliders for filtering
    with st.sidebar:
        st.subheader("Map Filters")

        cost_range = st.slider(
            "Production Cost Range ($)",
            min_value=0,
            max_value=int(school_geo['Production_Cost_Total'].max()),
            value=(0, int(school_geo['Production_Cost_Total'].max())),
            step=100
        )

        waste_range = st.slider(
            "Waste Cost Range ($)",
            min_value=0,
            max_value=int(school_geo['Total_Waste_Cost'].max()),
            value=(0, int(school_geo['Total_Waste_Cost'].max())),
            step=50
        )

    # Apply the filters
    school_geo = school_geo[
        (school_geo['Production_Cost_Total'] >= cost_range[0]) &
        (school_geo['Production_Cost_Total'] <= cost_range[1]) &
        (school_geo['Total_Waste_Cost'] >= waste_range[0]) &
        (school_geo['Total_Waste_Cost'] <= waste_range[1])
        ]

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Schools Displayed", school_geo['School Name'].nunique())
    with col2:
        st.metric("Total Cost", f"${school_geo['Production_Cost_Total'].sum():,.2f}")
    with col3:
        st.metric("Total Waste", f"${school_geo['Total_Waste_Cost'].sum():,.2f}")

    # Create map
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
        color_continuous_scale='Blues',
        zoom=10,
        # title='School Breakfast Program: Geographic Distribution of Costs and Waste'
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 10, "t": 50, "l": 10, "b": 10},
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------------------------------------

elif selected_viz == "Interactive School Map with Layers":
    # st.header("🗺️ Interactive School Map with Layers")

    # Apply global filters
    filtered_df = apply_filters(df, selected_school, date_range, menu_items)

    if 'latitude' not in filtered_df.columns or 'longitude' not in filtered_df.columns:
        st.warning("Latitude/Longitude columns not found in data.")
        st.stop()

    # Aggregate school data
    school_geo = filtered_df.groupby(['School Name', 'latitude', 'longitude']).agg({
        'Production_Cost_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()

    if school_geo.empty:
        st.warning("No schools match the selected filters.")
        st.stop()

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.subheader("Map Display Filters")

        # Show which layer
        layer_option = st.radio(
            "Layer View",
            options=["Show Costs", "Show Waste", "Show Both"],
            index=2,
            horizontal=False,
            key="layer_selector"
        )

        cost_range = st.slider(
            "Production Cost Range ($)",
            min_value=0,
            max_value=int(school_geo['Production_Cost_Total'].max()),
            value=(0, int(school_geo['Production_Cost_Total'].max())),
            step=100,
            key="map_cost_range"
        )

        waste_range = st.slider(
            "Waste Cost Range ($)",
            min_value=0,
            max_value=int(school_geo['Total_Waste_Cost'].max()),
            value=(0, int(school_geo['Total_Waste_Cost'].max())),
            step=50,
            key="map_waste_range"
        )

        top_n = st.selectbox(
            "Number of Schools to Display",
            options=[10, 20, 30, 50, 100, "All"],
            index=2,
            key="map_school_count"
        )

    # ---------------- Filtering ----------------
    school_geo = school_geo[
        (school_geo['Production_Cost_Total'] >= cost_range[0]) &
        (school_geo['Production_Cost_Total'] <= cost_range[1]) &
        (school_geo['Total_Waste_Cost'] >= waste_range[0]) &
        (school_geo['Total_Waste_Cost'] <= waste_range[1])
    ]

    if top_n != "All":
        school_geo = school_geo.sort_values("Production_Cost_Total", ascending=False).head(int(top_n))

    if school_geo.empty:
        st.warning("No schools match the selected filter values.")
        st.stop()

    # ---------------- Summary ----------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Schools Displayed", school_geo['School Name'].nunique())
    with col2:
        st.metric("Total Cost", f"${school_geo['Production_Cost_Total'].sum():,.2f}")
    with col3:
        st.metric("Total Waste", f"${school_geo['Total_Waste_Cost'].sum():,.2f}")

    # ---------------- Map Layers ----------------
    fig = go.Figure()

    if layer_option in ["Show Costs", "Show Both"]:
        fig.add_trace(go.Scattermap(
            lon=school_geo['longitude'],
            lat=school_geo['latitude'],
            text=school_geo.apply(lambda row: f"{row['School Name']}<br>Cost: ${row['Production_Cost_Total']:,.2f}", axis=1),
            marker=dict(
                size=school_geo['Production_Cost_Total'] / 100,
                color=school_geo['Production_Cost_Total'],
                colorscale='Viridis',
                colorbar_title="Cost ($)",
                sizemode='area',
                sizemin=5
            ),
            name='Production Costs'
        ))

    if layer_option in ["Show Waste", "Show Both"]:
        fig.add_trace(go.Scattermap(
            lon=school_geo['longitude'],
            lat=school_geo['latitude'],
            text=school_geo.apply(lambda row: f"{row['School Name']}<br>Waste: ${row['Total_Waste_Cost']:,.2f}", axis=1),
            marker=dict(
                size=school_geo['Total_Waste_Cost'] / 50,
                color=school_geo['Total_Waste_Cost'],
                colorscale='Blues',
                colorbar_title="Waste ($)",
                sizemode='area',
                sizemin=5
            ),
            name='Food Waste'
        ))

    # ---------------- Layout ----------------
    fig.update_layout(
        # title="Interactive School Map: Costs and Waste",
        geo=dict(
            scope='usa',  # or 'world' if your schools are international
            projection_type='equirectangular',
            showland=True,
            landcolor="rgb(243, 243, 243)",
            subunitwidth=1,
            countrywidth=1,
            center=dict(lat=school_geo['latitude'].mean(), lon=school_geo['longitude'].mean()),
            lataxis_range=[school_geo['latitude'].min() - 0.1, school_geo['latitude'].max() + 0.1],
            lonaxis_range=[school_geo['longitude'].min() - 0.1, school_geo['longitude'].max() + 0.1],
        ),
        height=650,
        margin={"r": 10, "t": 50, "l": 10, "b": 10}
    )

    st.plotly_chart(fig, use_container_width=True)


elif selected_viz == "Enhanced School Region Map":
    # st.header("🗺️ Enhanced School Region Map")

    # Load GeoJSON file
    try:
        with open('../data/preprocessed-data/School_Regions.geojson', 'r') as f:
            geojson_data = geojson.load(f)
    except Exception as e:
        st.error(f"Failed to load GeoJSON file: {e}")
        st.stop()

    # Group and aggregate school data
    school_stats = df.groupby('School Name').agg({
        'latitude': 'first',
        'longitude': 'first',
        'Production_Cost_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()

    # Set map center
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")

    # --- Color utilities ---
    def get_random_color():
        return "#{:02x}{:02x}{:02x}".format(
            random.randint(100, 255),
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
            'fillColor': region_colors.get(feature['properties']['REGION'], '#808080'),
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['REGION'],
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
            radius=max(3, 5 + (row['Production_Cost_Total'] / 1000)),
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(f"{row['School Name']}<br>Cost: ${row['Production_Cost_Total']:,.2f}", max_width=250)
        ).add_to(cost_subgroup)

    # Waste markers (red)
    for _, row in school_stats.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=max(3, 5 + (row['Total_Waste_Cost'] / 500)),
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(f"{row['School Name']}<br>Waste: ${row['Total_Waste_Cost']:,.2f}", max_width=250)
        ).add_to(waste_subgroup)

    # Finalize map
    folium.LayerControl(collapsed=False).add_to(m)

    # Display map in Streamlit
    folium_static(m, width=1200, height=650)

# Footer
st.markdown("---")
st.markdown("*Dashboard created using Streamlit and Plotly*")
st.markdown("Author: Timur Abdygulov.")
