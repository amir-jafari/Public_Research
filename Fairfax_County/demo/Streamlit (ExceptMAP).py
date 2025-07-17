
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


# --- CONFIGURE PAGE ---
st.set_page_config(
    page_title="FCPS Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOGIN LOGIC ---
#if 'authenticated' not in st.session_state:
    #st.session_state.authenticated = False


#def login():
   # st.markdown("<h2 style='text-align:center;'>🔐 Login to Access Report </h2>", unsafe_allow_html=True)
   # username = st.text_input("Username")
    #password = st.text_input("Password", type="password")

   # if st.button("Login"):
      #  if username == "Amir" and password == "FCPS@123":
      #      st.session_state.authenticated = True
            #st.rerun()
        #else:
          #  st.error("Invalid credentials. Please try again.")


#if not st.session_state.authenticated:
    #login()
   # st.stop()

# --- LOGO URLs (update paths accordingly) ---
fcps_logo_url = "FCPS-logo.png"
gw_logo_url = "gw-logo.png"

# --- HEADER SECTION ---
header_col1, header_col2, header_col3 = st.columns([1, 5, 1])
with header_col1:
    st.image(fcps_logo_url, width=80)
with header_col2:
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>FCPS Food Court Management</h1>
        <p style='text-align: center; color: gray;'>In partnership with GW Data Science</p>
    """, unsafe_allow_html=True)
with header_col3:
    st.image(gw_logo_url, width=80)

st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# Initialize session state variables
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# --- STYLE & JS ---
def set_background(dark_mode):
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }
        button {
            color: white !important;
            background-color: #1e88e5 !important;
        }
        .css-1offfwp {
            background-color: #121212 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        button {
            color: black !important;
            background-color: #4A90E2 !important;
        }
        .css-1offfwp {
            background-color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)


# Call style setter
set_background(st.session_state.dark_mode)

st.markdown("""
<style>
.big-button-container {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    height: 7vh;
    gap: 6rem;
    padding-top: 0.10rem;
    padding-right: 10rem;
    padding-left: 10rem;
}
.small-button-container {
    font-size: 12px !important;
    padding: 0.25rem 0.75rem !important;
    height: auto !important;
    width: auto !important;
    background-color: #e0e0e0 !important;
    color: black !important;
    border-radius: 8px !important;
}

div.stButton > button {
    width: 300px;
    height: 300px;
    font-size: 36px;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    border-radius: 20px;
    border: none;
    color: white !important;
    background-color: #b71c1c !important;
    box-shadow: 4px 6px 10px rgba(0, 0, 0, 0.4);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background-color: #d32f2f !important;
    transform: translateY(-2px);
    box-shadow: 6px 8px 14px rgba(0, 0, 0, 0.5);
}

.dark-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background-color: rgba(0,0,0,0.6);
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    mode = st.radio("Select Background Mode:", ["Light (White)", "Dark"], index=1 if st.session_state.dark_mode else 0)
    new_mode = (mode == "Dark")
    if new_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_mode
        st.rerun()

if st.session_state.dark_mode:
    st.markdown('<div class="dark-overlay"></div>', unsafe_allow_html=True)

# Buttons centered on page
st.markdown("<div class='big-button-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Sales Overview", key="landing_sales"):
        st.session_state.selected_option = "Sales"
        st.rerun()

with col2:
    if st.button("Production Overview", key="landing_production"):
        st.session_state.selected_option = "Production"
        st.rerun()

with col3:
    if st.button("Forecast Next 7 Days (Coming Soon)", key="landing_forecast"):
        st.info("Forecast feature is coming soon. Stay tuned!")

st.markdown("</div>", unsafe_allow_html=True)

def show_sidebar():
    sidebar = st.sidebar
    sidebar.markdown(f"<h2 style='color:#333333; font-weight:bold;'>{st.session_state.selected_option} Panel</h2>", unsafe_allow_html=True)
    sidebar.markdown("<hr style='margin-top: 5px; margin-bottom: 7px;'>", unsafe_allow_html=True)

    sidebar.markdown('<div class="small-button-container">', unsafe_allow_html=True)
    if sidebar.button("Back to Main Menu"):
        st.session_state.selected_option = None
        st.rerun()
    sidebar.markdown('</div>', unsafe_allow_html=True)

    return sidebar


def sales_panel(sidebar):
    sidebar.markdown("<h4 style='margin-top: 20px;'>Upload Data</h4>", unsafe_allow_html=True)
    uploaded_file = sidebar.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv('../data/preprocessed-data/sales.csv')
        except FileNotFoundError:
            st.error("sales.csv not found.")
            st.stop()

    if 'date' not in df.columns:
        st.error("CSV must include a `date` column.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])

    section = sidebar.radio("Select Section", ["Single School View", "Compare Schools"])

    meal_types = {
        'Free Meals': 'free_meals',
        'Reduced Meals': 'reduced_price_meals',
        'Full Price Meals': 'full_price_meals',
        'Adults': 'adults'
    }

    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    if section == "Single School View":
        sidebar.markdown("<h4 style='margin-top: 20px;'>Filters</h4>", unsafe_allow_html=True)
        school_list = ['All Schools'] + sorted(df['school_name'].unique())
        selected_school = sidebar.selectbox("School", school_list)

        selected_meals = [col for label, col in meal_types.items() if sidebar.checkbox(label, value=True)]

        date_range = sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
        time_agg = sidebar.selectbox("Aggregate By", ['Date', 'Day of Week', 'Month'])
        chart_style = sidebar.selectbox("Chart Type", ['Grouped Bars', 'Stacked Bars', 'Line Chart', 'Pie Chart'])

        filtered = filter_data(df, selected_school, selected_meals, date_range, time_agg)

        if not filtered.empty:
            show_single_school_data(filtered, time_agg, chart_style)
        else:
            st.warning("No data for selected filters.")

    elif section == "Compare Schools":
        sidebar.markdown("<h4 style='margin-top: 20px;'>Comparison Filters</h4>", unsafe_allow_html=True)

        school_list = sorted(df['school_name'].unique())
        selected_schools = sidebar.multiselect("Select Schools to Compare", school_list, default=school_list[:2])

        selected_meals = [col for label, col in meal_types.items() if sidebar.checkbox(label, value=True)]

        date_range = sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

        time_agg = sidebar.selectbox("Aggregate By", ['Date', 'Day of Week', 'Month'], key="compare_time_agg")
        chart_style = sidebar.selectbox("Chart Type", ['Grouped Bars', 'Stacked Bars', 'Line Chart', 'Pie Chart'], key="compare_chart_style")

        if not selected_schools:
            st.warning("Please select at least one school to compare.")
        else:
            comp_df = prepare_comparison_data(df, selected_schools, selected_meals, date_range, time_agg)
            if comp_df.empty:
                st.warning("No data available for the selected schools and filters.")
            else:
                show_comparison_data(comp_df, time_agg, chart_style)

    import folium
    import geojson
    import random
    from folium.plugins import FeatureGroupSubGroup
    import traceback

    #def create_school_cost_waste_map(df, geojson_path, output_html="school_cost_waste_map.html"):
    #    try:
    #        print("Data columns:", df.columns.tolist())
    #        print("GeoJSON path:", geojson_path)

    #        with open(geojson_path, 'r') as f:
    #            geojson_data = geojson.load(f)
    #        print("Loaded GeoJSON features:", len(geojson_data['features']))

    #        school_stats = df.groupby('School Name').agg({
    #            'latitude': 'first',
    #            'longitude': 'first',
     #           'Production_Cost_Total': 'sum',
      #          'Total_Waste_Cost': 'sum'
      #      }).reset_index()

    #        map_center = [df['latitude'].mean(), df['longitude'].mean()]
    #       m = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")

         #   def get_random_color():
         #       return "#{:02x}{:02x}{:02x}".format(
          #          random.randint(100, 255),
          #          random.randint(100, 255),
          #          random.randint(100, 255)
         #       )

           # region_colors = {
           #     feature['properties']['REGION']: get_random_color()
          #      for feature in geojson_data['features']
          #  }

           # folium.GeoJson(
           #     geojson_data,
            #    name='School Regions',
            #    style_function=lambda feature: {
            #        'fillColor': region_colors.get(feature['properties']['REGION'], '#808080'),
            #        'color': '#000000',
            #        'weight': 1,
           #         'fillOpacity': 0.5
            #    },
          #      tooltip=folium.GeoJsonTooltip(fields=['REGION'], aliases=['Region:'], sticky=True)
           # ).add_to(m)

          #  cost_group = folium.FeatureGroup(name='Cost by School', show=True)
          #  waste_group = folium.FeatureGroup(name='Waste by School', show=False)

           # cost_subgroup = FeatureGroupSubGroup(cost_group, 'Cost Subgroup')
           # waste_subgroup = FeatureGroupSubGroup(waste_group, 'Waste Subgroup')

          #  m.add_child(cost_group)
          #  m.add_child(waste_group)
          #  m.add_child(cost_subgroup)
          #  m.add_child(waste_subgroup)

            #for _, row in school_stats.iterrows():
            #    folium.CircleMarker(
             #       location=[row['latitude'], row['longitude']],
           #         radius=max(3, 5 + (row['Production_Cost_Total'] / 1000)),
           #         color='blue',
           #         fill=True,
           #         fill_opacity=0.7,
           #         popup=f"{row['School Name']}<br>Cost: ${row['Production_Cost_Total']:,.2f}"
           #     ).add_to(cost_subgroup)

           # for _, row in school_stats.iterrows():
             #   folium.CircleMarker(
            #        location=[row['latitude'], row['longitude']],
            #        radius=max(3, 5 + (row['Total_Waste_Cost'] / 500)),
            #        color='red',
            #        fill=True,
             #       fill_opacity=0.7,
              #      popup=f"{row['School Name']}<br>Waste: ${row['Total_Waste_Cost']:,.2f}"
              #  ).add_to(waste_subgroup)

            #folium.LayerControl(collapsed=False).add_to(m)

         #   m.save(output_html)
         #   print(f"Map saved to {output_html}")

        #except Exception as e:
        #    print("Error in map generation:", e)
       #     print(traceback.format_exc())
        #    raise


def production_panel(sidebar):
    st.subheader("Production Overview")

    prod_type = sidebar.radio("Select Production Type", ["Breakfast", "Lunch", "Consolidated"])
    view_option = sidebar.radio(
        "Select View Option",
        ["Show Production Map", "Show Top 10 School Report"]
    )



    if prod_type == "Breakfast":
        data_path = "../data/preprocessed-data/data_breakfast_with_coordinates.csv"
        #geojson_path = "/Users/sayanpatra/Downloads/School_Regions.geojson"
       # map_path = "/Users/sayanpatra/Downloads/improved_breakfast_map.html"
        try:
            df_production = pd.read_csv(data_path, low_memory=False)
        except FileNotFoundError:
            st.error(f"{prod_type} production data not found.")
            return

    elif prod_type == "Lunch":
        data_path = "../data/preprocessed-data/data_lunch_with_coordinates.csv"
        #geojson_path = "/Users/sayanpatra/Downloads/School_Regions.geojson"
        #map_path = "/Users/sayanpatra/Downloads/improved_lunch_map.html"
        try:
            df_production = pd.read_csv(data_path, low_memory=False)
        except FileNotFoundError:
            st.error(f"{prod_type} production data not found.")
            return

    else:  # Consolidated
        breakfast_path = "../data/preprocessed-data/data_breakfast_with_coordinates.csv"
        lunch_path = "../data/preprocessed-data/data_lunch_with_coordinates.csv"
        #map_path = "improved_consolidated_map.html"

        try:
            df_breakfast = pd.read_csv(breakfast_path, low_memory=False)
            df_lunch = pd.read_csv(lunch_path, low_memory=False)
        except FileNotFoundError:
            st.error("Breakfast or Lunch production data not found.")
            return

        # Combine breakfast and lunch data
        df_production = pd.concat([df_breakfast, df_lunch], ignore_index=True)

    # Clean numeric columns
    cost_columns = ['Discarded_Cost', 'Subtotal_Cost', 'Left_Over_Cost', 'Production_Cost_Total']
    for col in cost_columns:
        if col in df_production.columns and not pd.api.types.is_numeric_dtype(df_production[col]):
            df_production[col] = (
                df_production[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .replace('nan', '0')
                .astype(float)
            )

    # Derived metrics
    df_production['Total_Waste_Cost'] = df_production.get('Left_Over_Cost', 0) + df_production.get('Discarded_Cost', 0)
    df_production['Date'] = pd.to_datetime(df_production['Date'], errors='coerce')
    df_production['Day_of_Week'] = df_production['Date'].dt.day_name()
    df_production['Planned_Cost'] = df_production.get('Planned_Total', 0) * (
        df_production.get('Subtotal_Cost', 0) / df_production.get('Served_Total', 1).replace(0, 1)
    )
    df_production['Cost_Deviation'] = df_production.get('Production_Cost_Total', 0) - df_production['Planned_Cost']

    # Show map or top 10 report
    #if view_option == "Show Production Map":
        #st.markdown(f"### Production Map ({prod_type})")
        #try:
            # Make sure df_production has 'latitude' and 'longitude' columns (lowercase) as expected by your function
           # if 'latitude' not in df_production.columns or 'longitude' not in df_production.columns:
            #    df_production['latitude'] = df_production['latitude']
            #    df_production['longitude'] = df_production['longitude']

            # Calculate total waste cost if missing
            #if 'Total_Waste_Cost' not in df_production.columns:
               # df_production['Total_Waste_Cost'] = df_production.get('Left_Over_Cost', 0) + df_production.get(
                    #'Discarded_Cost', 0)

            #create_school_cost_waste_map(df_production, geojson_path, output_html=map_path)

            #with open(map_path, 'r') as f:
             #   map_html = f.read()
            #st.components.v1.html(map_html, height=700)
        #except Exception as e:
        #    st.error(f"Failed to generate map: {e}")
    #else:
       # st.markdown(f"### Top 10 Schools by Production Cost ({prod_type})")
      #  top_10_schools = df_production.groupby("School Name")["Production_Cost_Total"].sum().nlargest(10).reset_index()
       # fig_top10 = px.bar(
       #     top_10_schools,
        #    x="School Name",
       #     y="Production_Cost_Total",
       #     title=f"Top 10 Schools by Total Production Cost ({prod_type})",
       #     labels={"Production_Cost_Total": "Total Cost ($)"},
     #       color_discrete_sequence=["#1E88E5"]
       # )
     #   st.plotly_chart(fig_top10, use_container_width=True)

    # Consolidated Production Summary
    st.markdown("### Consolidated Production Summary")
    total_cost = df_production['Production_Cost_Total'].sum()
    total_waste = df_production['Total_Waste_Cost'].sum()
    total_served = df_production['Served_Total'].sum() if 'Served_Total' in df_production.columns else None
    avg_cost_per_student = total_cost / total_served if total_served else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Production Cost", f"${total_cost:,.2f}")
    col2.metric("Total Waste Cost", f"${total_waste:,.2f}")
    if avg_cost_per_student:
        col3.metric("Avg Cost per Student", f"${avg_cost_per_student:,.2f}")
    else:
        col3.write("")

    # Total Production Cost by School Bar Chart
    school_costs = df_production.groupby('School Name')['Production_Cost_Total'].sum().reset_index().sort_values(by='Production_Cost_Total', ascending=False)
    fig1 = px.bar(
        school_costs,
        x='School Name',
        y='Production_Cost_Total',
        title=f'Total Production Cost by School ({prod_type})',
        labels={'Production_Cost_Total': 'Total Cost ($)', 'School Name': 'School'},
        color_discrete_sequence=['#4A90E2'],
        height=700
    )
    fig1.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig1, use_container_width=True)

    # Cost Over Time by Menu Item Line Chart
    all_menu_items = sorted(df_production['Name'].dropna().unique())

    selected_menu_items = sidebar.multiselect(
        "Select Menu Items to Display",
        options=all_menu_items,
        default=all_menu_items
    )

    filtered_df = df_production[df_production['Name'].isin(selected_menu_items)]

    time_item_costs = filtered_df.groupby(['Date', 'Name'])['Production_Cost_Total'].sum().reset_index()

    fig2 = px.line(
        time_item_costs, x='Date', y='Production_Cost_Total', color='Name',
        title=f'Cost Over Time by Menu Item ({prod_type})',
        labels={'Production_Cost_Total': 'Total Cost ($)', 'Name': 'Menu Item'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Cost Distribution Sunburst Chart
    school_item_costs = df_production.groupby(['School Name', 'Name'])['Production_Cost_Total'].sum().reset_index()
    fig3 = px.sunburst(
        school_item_costs,
        path=['School Name', 'Name'],
        values='Production_Cost_Total',
        title=f'Cost Distribution: Schools and Menu Items ({prod_type})'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Food Waste Overview
    st.markdown(f"### Food Waste Overview ({prod_type})")

    waste_by_school = df_production.groupby('School Name')['Total_Waste_Cost'].sum().nlargest(10)
    fig4 = px.bar(
        waste_by_school, title=f'Top 10 Schools by Food Waste Cost ({prod_type})',
        labels={'value': 'Waste Cost ($)', 'School Name': 'School'},
        color_discrete_sequence=['#D32F2F'] if prod_type == "Breakfast" else ['#E91E63']
    )
    st.plotly_chart(fig4, use_container_width=True)

    waste_by_item = df_production.groupby('Name')['Total_Waste_Cost'].sum().nlargest(10)
    fig5 = px.bar(
        waste_by_item, title=f'Top 10 Wasted Menu Items ({prod_type})',
        labels={'value': 'Waste Cost ($)', 'Name': 'Menu Item'},
        color_discrete_sequence=['#F4511E'] if prod_type == "Breakfast" else ['#9C27B0']
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Cost Deviation by School Box Plot
    fig6 = px.box(
        df_production, x='School Name', y='Cost_Deviation',
        title=f'Cost Deviation by School ({prod_type})',
        labels={'Cost_Deviation': 'Deviation ($)'},
        color_discrete_sequence=['#00897B'] if prod_type == "Breakfast" else ['#1976D2']
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Popularity vs Waste Scatter Plot
    st.markdown(f"### Popularity vs. Waste by Menu Item ({prod_type})")
    item_stats = df_production.groupby('Name').agg({
        'Served_Total': 'sum',
        'Total_Waste_Cost': 'sum'
    }).reset_index()

    fig7 = px.scatter(
        item_stats,
        x='Served_Total', y='Total_Waste_Cost',
        size='Total_Waste_Cost', hover_name='Name',
        title=f'Popularity vs. Waste by Menu Item ({prod_type})',
        labels={'Served_Total': 'Total Served', 'Total_Waste_Cost': 'Waste ($)'}
    )
    st.plotly_chart(fig7, use_container_width=True)

    # Average Food Waste by Day of Week Bar Chart
    waste_by_day = df_production.groupby('Day_of_Week')['Total_Waste_Cost'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    waste_by_day = waste_by_day.reindex(day_order, fill_value=0).reset_index()

    fig8 = px.bar(
        waste_by_day, x='Day_of_Week', y='Total_Waste_Cost',
        title=f'Average Food Waste by Day of Week ({prod_type})',
        labels={'Total_Waste_Cost': 'Avg Waste ($)', 'Day_of_Week': 'Day'}
    )
    st.plotly_chart(fig8, use_container_width=True)

    # Cost per Student by Region Bar Chart
    if 'FCPS Region' in df_production.columns:
        region_cost = df_production.groupby('FCPS Region').agg({
            'Production_Cost_Total': 'sum',
            'Served_Total': 'sum'
        }).reset_index()
        region_cost['Cost_Per_Student'] = region_cost['Production_Cost_Total'] / region_cost['Served_Total']

        fig9 = px.bar(
            region_cost, x='FCPS Region', y='Cost_Per_Student',
            title=f'Cost per Student by Region ({prod_type})',
            labels={'Cost_Per_Student': 'Cost/Student ($)'}
        )
        st.plotly_chart(fig9, use_container_width=True)


def filter_data(df, school, meals, date_range, agg):
    if school != 'All Schools':
        df = df[df['school_name'] == school].copy()
    if len(date_range) == 2:
        start, end = date_range
        df = df[(df['date'].dt.date >= start) & (df['date'].dt.date <= end)]

    daily = df.groupby('date')[meals].sum().reset_index()

    if agg == 'Day of Week':
        daily['agg'] = daily['date'].dt.day_name()
        agg_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily['agg'] = pd.Categorical(daily['agg'], categories=agg_order, ordered=True)
    elif agg == 'Month':
        daily['agg'] = daily['date'].dt.to_period('M').astype(str)
    else:
        daily['agg'] = daily['date'].dt.strftime('%Y-%m-%d')

    melt_df = pd.melt(daily, id_vars='agg', value_vars=meals,
                      var_name='Category', value_name='Count')

    melt_df['Category'] = melt_df['Category'].map({
        'free_meals': 'Free Meals',
        'reduced_price_meals': 'Reduced Meals',
        'full_price_meals': 'Full Price Meals',
        'adults': 'Adults'
    })

    melt_df.rename(columns={'agg': 'Time'}, inplace=True)
    return melt_df

def prepare_comparison_data(df, selected_schools, selected_meals, date_range, time_agg):
    comp_list = []
    for school in selected_schools:
        temp = df[df['school_name'] == school].copy()
        if len(date_range) == 2:
            start, end = date_range
            temp = temp[(temp['date'].dt.date >= start) & (temp['date'].dt.date <= end)]
        daily = temp.groupby('date')[selected_meals].sum().reset_index()

        if time_agg == 'Day of Week':
            daily['agg'] = daily['date'].dt.day_name()
            agg_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily['agg'] = pd.Categorical(daily['agg'], categories=agg_order, ordered=True)
        elif time_agg == 'Month':
            daily['agg'] = daily['date'].dt.to_period('M').astype(str)
        else:
            daily['agg'] = daily['date'].dt.strftime('%Y-%m-%d')

        melt_df = pd.melt(daily, id_vars='agg', value_vars=selected_meals,
                          var_name='Category', value_name='Count')

        melt_df['Category'] = melt_df['Category'].map({
            'free_meals': 'Free Meals',
            'reduced_price_meals': 'Reduced Meals',
            'full_price_meals': 'Full Price Meals',
            'adults': 'Adults'
        })
        melt_df['School'] = school
        melt_df.rename(columns={'agg': 'Time'}, inplace=True)
        comp_list.append(melt_df)

    comp_df = pd.concat(comp_list)
    return comp_df

def show_single_school_data(filtered, time_agg, chart_style):
    st.markdown("### Portfolio Snapshot")
    total = filtered['Count'].sum()
    avg = filtered.groupby('Time')['Count'].sum().mean()
    peak = filtered.groupby('Time')['Count'].sum().max()
    span = filtered['Time'].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Meals", f"{total:,}")
    m2.metric("Avg per Period", f"{avg:.2f}")
    m3.metric("Peak Meals", f"{peak:,}")
    m4.metric(f"{time_agg}s Tracked", span)

    st.subheader(f"{time_agg} Breakdown")

    if chart_style in ['Grouped Bars', 'Stacked Bars']:
        fig = px.bar(
            filtered,
            x='Time',
            y='Count',
            color='Category',
            title=f"{time_agg} Meal Distribution",
            barmode='group' if chart_style == 'Grouped Bars' else 'stack',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=500, xaxis_title=time_agg, yaxis_title="Meal Count")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_style == 'Line Chart':
        fig = px.line(
            filtered,
            x='Time',
            y='Count',
            color='Category',
            title=f"{time_agg} Meal Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
            markers=True
        )
        fig.update_layout(height=500, xaxis_title=time_agg, yaxis_title="Meal Count")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_style == 'Pie Chart':
        pie_data = filtered.groupby('Category')['Count'].sum().reset_index()
        fig = px.pie(
            pie_data,
            names='Category',
            values='Count',
            title="Total Meal Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Data"):
        summary = filtered.pivot_table(index='Time', columns='Category', values='Count', aggfunc='sum').fillna(0)
        summary['Total'] = summary.sum(axis=1)
        st.dataframe(summary, use_container_width=True)

    st.download_button(
        "Download CSV",
        filtered.to_csv(index=False),
        file_name=f"filtered_meal_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def show_comparison_data(comp_df, time_agg, chart_style):
    st.markdown("### School Comparison")

    if chart_style == 'Grouped Bars':
        fig = px.bar(
            comp_df,
            x='Time',
            y='Count',
            color='School',
            pattern_shape='Category',
            barmode='group',
            title=f"Meal Comparison by School ({time_agg} Aggregation)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=600, xaxis_title=time_agg, yaxis_title="Meal Count")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_style == 'Line Chart':
        fig = px.line(
            comp_df,
            x='Time',
            y='Count',
            color='School',
            line_dash='Category',
            title=f"Meal Comparison by School ({time_agg} Aggregation)",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=600, xaxis_title=time_agg, yaxis_title="Meal Count")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Comparison Raw Data"):
        comp_summary = comp_df.pivot_table(index=['Time', 'School'], columns='Category', values='Count', aggfunc='sum').fillna(0)
        comp_summary['Total'] = comp_summary.sum(axis=1)
        st.dataframe(comp_summary, use_container_width=True)

    st.download_button(
        "Download Comparison CSV",
        comp_df.to_csv(index=False),
        file_name=f"compare_schools_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# --- MAIN APP LOGIC ---
if st.session_state.selected_option == "Sales":
    sidebar = show_sidebar()
    sales_panel(sidebar)
elif st.session_state.selected_option == "Production":
    sidebar = show_sidebar()
    production_panel(sidebar)


