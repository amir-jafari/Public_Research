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

st.header("About This Dashboard")
st.write("Welcome to the School Meal Data Dashboard!")
st.write("This dashboard provides comprehensive analysis of meal data.")

st.markdown("---")
st.markdown("*Dashboard created with Streamlit and Plotly*")
