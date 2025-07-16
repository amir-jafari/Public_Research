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
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False


def login():
    st.markdown("<h2 style='text-align:center;'>🔐 Login to Access Report </h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Amir" and password == "FCPS@123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")


if not st.session_state.authenticated:
    login()
    st.stop()

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
    gap: 5rem;
    padding-top: 0.10rem;
    padding-right: 8rem;
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
            df = pd.read_csv('../../Data/preprocessed-data/sales.csv')
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
        chart_style = sidebar.selectbox("Chart Type", ['Grouped Bars', 'Line Chart'], key="compare_chart_style")

        if not selected_schools:
            st.warning("Please select at least one school to compare.")
        else:
            comp_df = prepare_comparison_data(df, selected_schools, selected_meals, date_range, time_agg)
            if comp_df.empty:
                st.warning("No data available for the selected schools and filters.")
            else:
                show_comparison_data(comp_df, time_agg, chart_style)

def production_panel(sidebar):
    st.subheader("Production Overview")
    st.info("This section is under development.")

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
