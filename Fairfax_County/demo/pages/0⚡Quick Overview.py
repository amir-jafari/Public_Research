import streamlit as st

# Set page config
st.set_page_config(
    page_title="Quick Overview",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Quick Overview")
st.markdown("Quick overview of what we have completed so far in this project:")

st.markdown("We began by preprocessing the data, converting Sales reports (PDFs) and Production reports (HTMLs) into structured CSV files. For extracting data from PDFs, we utilized the `PyPDF2` and `pdfplumber` Python libraries.")

col1, col2 = st.columns(2)
with col1:
    st.image("images/example-pdf-sales.png", caption="Sales PDF Example")
with col2:
    st.image("images/example-csv-sales.png", caption="Preprocessed Sales CSV Example")

col3, col4 = st.columns(2)
with col3:
    st.image("images/example-html-prod.png", caption="Production HTML Example")
with col4:
    st.image("images/example-csv-prod.png", caption="Preprocessed Production CSV Example")

st.markdown("Due to the limited time frame, we prioritized the development of an [Exploratory Data Analysis (EDA)](https://en.wikipedia.org/wiki/Exploratory_data_analysis) dashboard to explore and visualize key trends in the data. The dashboard was built using `streamlit` and `plotly` Python libraries.")

col5, col6 = st.columns(2)
with col5:
    st.image("images/sales-eda.png", caption="EDA Sales Dashboard")
with col6:
    st.image("images/prod-eda.png", caption="EDA Production Dashboard")

st.markdown("---")

st.header("🏭 Production EDA Insights")
st.write("Some insights that we have gathered from the Production dashboard:")

st.subheader("📌 Total Cost by School")
st.markdown("- **Insight:** Average cost of breakfast for all schools was 4,239\$ and the average cost of lunch for all schools was 21,037\$.")
st.markdown("- **Insight:** Top 3 schools with highest breakfast cost: Falls Church, Poe, Hybla Valley.")
st.markdown("- **Insight:** Top 3 schools with highest lunch cost: Lake Braddock, Robinson, Chantilly.")

st.subheader("📌 Cost Over Time by Menu Item")
st.markdown("- **Insight:** Top 3 breakfast menu items with the highest cost: Mini Maple Pancakes, 1% White Milk, Apple Juice.")
st.markdown("- **Insight:** Top 3 lunch menu items with the highest cost: Chicken Tenders, Fat Free Chocolate Milk, Asian Veggie Sub.")

st.subheader("📌 Cost Distribution: Schools and Menu Items")
st.markdown("- **Insight:** Top 3 schools with the highest cost breakfast menu items: Hybla Valley, Mount Vernon, Parklawn.")
st.markdown("- **Insight:** Top 3 schools with the highest cost lunch menu items: Annandale, Oakton, Woodson.")

st.subheader("📌 Top Schools by Food Waste Cost")
st.markdown("- **Insight:** Top 3 schools with the highest breakfast waste cost: Hybla Valley, Falls Church, Poe.")
st.markdown("- **Insight:** Top 3 schools with the highest lunch waste cost: Chantilly, Hayfield, Oakton.")

st.subheader("📌 Top Wasted Menu Items")
st.markdown("- **Insight:** Top 3 breakfast menu items with the highest waste cost: 1% White Milk, Fat Free White Milk, Cinnamon Chex.")
st.markdown("- **Insight:** Top 3 lunch menu items with the highest waste cost: Fat Free Chocolate Milk, 1% White Milk, PB&J Power Pack.")

st.subheader("📌 Cost Deviation by School")
st.markdown("- **Insight:** .")

st.subheader("📌 Popularity vs. Waste by Menu Item")
st.markdown("- **Insight:** Mini Maple Pancakes have high servings but low waste cost.")
st.markdown("- **Insight:** 1% White Milk have high servings and high waste cost.")

st.subheader("📌 Average Food Waste by Day of Week")
st.markdown("- **Insight:** On average, Monday's have the highest breakfast waste cost.")
st.markdown("- **Insight:** On average, Thursday's have the highest lunch waste cost.")

st.subheader("📌 Cost per Student by Region")
st.markdown("- **Insight:** Top 3 Fairfax County regions with the highest cost per student: Region 2, Region 3, Region 6.")

st.subheader("📌 Geographic Distribution of Costs and Waste")
st.markdown("- **Insight:** .")

st.subheader("📌 Interactive School Map with Layers")
st.markdown("- **Insight:** .")

st.subheader("📌 Enhanced School Region Map")
st.markdown("- **Insight:** .")

st.markdown("---")

st.header("📊 Sales EDA Insights")
st.write("Some insights that we have gathered from the Sales dashboard:")

st.subheader("📌 aa")
st.markdown("- **Insight:** Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nibh elementum imperdiet.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Duis sagittis ipsum. Praesent mauris. Fusce nec tellus sed augue semper porta.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Mauris massa. Vestibulum lacinia arcu eget nulla. Class aptent taciti sociosqu ad litora.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Curabitur sodales ligula in libero. Sed dignissim lacinia nunc. Curabitur tortor.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Pellentesque nibh. Aenean quam. In scelerisque sem at dolor. Maecenas mattis.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Sed convallis tristique sem. Proin ut ligula vel nunc egestas porttitor.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Morbi lectus risus, iaculis vel, suscipit quis, luctus non, massa. Fusce ac turpis quis.")

st.subheader("📌 aa")
st.markdown("- **Insight:** Aenean eu leo quam. Pellentesque ornare sem lacinia quam venenatis vestibulum.")
