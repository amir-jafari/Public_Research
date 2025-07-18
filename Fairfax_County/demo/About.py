# streamlit run app.py
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Fairfax County Public Schools (FCPS) Meal Dashboard Demo",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🍽️ Fairfax County Public Schools (FCPS) Meal Dashboard Demo")
st.markdown("Interactive dashboard created by GWU Data Science alumni to explore insights from school meal data in Fairfax County.")

# Sections overview
st.header("Sections 📂")
st.markdown("""
- ⚡ **Quick Overview**: A high-level summary of our methodology and insights gathered.
- 🏭 **Production EDA**: Visual exploration of food production patterns and menu item usage across schools.
- 📊 **Sales EDA**: Deep dive into sales trends by `free_meals`, `reduced_price_meals`, `full_price_meals`, `adults` by order quantities.
- 🔮 **Future Ideas**: Suggestions for additional data integration and advanced modeling opportunities.
""")

# GitHub link
st.header("GitHub Repository 🚀")
st.markdown("""
You can view the full code and data files used to generate this dashboard on our [GitHub repository](https://github.com/amir-jafari/Public_Research).
""")

# Python libraries used
st.header("Python Libraries Used 📚")
st.markdown("""
- `pandas` – data manipulation and cleaning.
- `tqdm` – progress bars during preprocessing.  
- `PyPDF2`, `pdfplumber` – extracting text and tables from PDF reports.  
- `plotly` – interactive charts and graphs. 
- `streamlit` – building the dashboard UI.
- `streamlit_folium` – embedding maps (if used).
- `geojson` – geographic data handling for school boundaries or maps.
""")

# Footer
st.markdown("---")
st.markdown("Authors: **Tyler Wallett** and **Timur Abdygulov**")
