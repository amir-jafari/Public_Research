# streamlit run app.py
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Future Ideas",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🔮 Future Ideas")
st.markdown("What's next? Well, we have a few ideas that we would like to explore further:")

st.subheader("EDA Ideas")

st.markdown("🔮 Segment by High School, Middle School and Elementary School.")
st.markdown("🔮 Connect Sales and Production records to identify discrepancies, trends, and revenue/cost optimization opportunities.")
st.markdown("🔮 Connect to FCPS database system.")

st.subheader("AI Modeling Ideas")

st.markdown("🔮 Forecasting/Time Series Modeling.")
st.markdown("🔮 Machine Learning Model: Classification.")


