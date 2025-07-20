import streamlit as st

# Set page config
st.set_page_config(
    page_title="Quick Overview",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Quick Overview")
st.markdown("Quick overview of what we have completed so far in this project:")

st.markdown("Schema of the data we are working with:")

st.image("images/schema.png", caption="Data Schema")

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
st.markdown("- **Insight:** Breakfast tends to be underspent (actual cost is often less than planned), with Falls Church showing the most frequent instances of underspending.")
st.markdown("- **Insight:** Lunch tends to stay closer to budget, with actual costs generally aligning with planned costs.")

st.subheader("📌 Popularity vs. Waste by Menu Item")
st.markdown("- **Insight:** Mini Maple Pancakes have high servings but low waste cost.")
st.markdown("- **Insight:** 1% White Milk have high servings and high waste cost.")

st.subheader("📌 Average Food Waste by Day of Week")
st.markdown("- **Insight:** On average, Monday's have the highest breakfast waste cost.")
st.markdown("- **Insight:** On average, Thursday's have the highest lunch waste cost.")

st.subheader("📌 Cost per Student by Region")
st.markdown("- **Insight:** Top 3 Fairfax County regions with the highest cost per student: Region 2, Region 3, Region 6.")

st.subheader("📌 Geographic Distribution of Costs and Waste")
st.subheader("📌 Interactive School Map with Layers")
st.subheader("📌 Enhanced School Region Map")
st.markdown("- **Insight:** Total waste costs appear to be more heavily concentrated in the Annandale and Falls Church regions.")

st.markdown("---")

st.header("📊 Sales EDA Insights")
st.write("Some insights that we have gathered from the Sales dashboard:")

st.subheader("📌 FRPA Sales Count by Day of Week")
st.markdown("- **Insight:** On average, Free Meals (F) are served more frequently than Full-Priced Meals (P) and Reduced-Priced Meals (R). Adults (A) are served the least.")
st.markdown("- **Insight:** Sales demand generally increases as the week progresses.")
st.markdown("- **Insight:** For breakfast, a similar pattern is observed, though there is a noticeable decline on Fridays.")
st.markdown("- **Insight:** For lunch, the increasing sales trend across the week is also evident.")

st.subheader("📌 FRPA Sales Count by Month")
st.markdown("- **Insight:** May recorded the highest sales demand, followed by March and then April.")
st.markdown("- **Insight:** This pattern is consistent across both breakfast and lunch. Breakfast sees a higher number of Free Meals (F), while Lunch shows more Full-Priced Meals (P).")

st.subheader("📌 Top 5 Schools with Highest Sales Volume")
st.markdown(
    "- **Insight:** The top 5 schools with the highest total meal sales are:\n"
    "    1. Falls Church High – 116,685 meals\n"
    "    2. Annandale High – 114,402 meals\n"
    "    3. Lake Braddock Secondary – 98,055 meals\n"
    "    4. Glasgow Middle – 84,302 meals\n"
    "    5. Holmes Middle – 79,704 meals"
)

st.subheader("📌 Top 5 Schools with Lowest Sales Volume")
st.markdown(
    "- **Insight:** The 5 schools with the lowest total meal sales are:\n"
    "    1. Vienna Elementary – 8,846 meals\n"
    "    2. Franklin Sherman Elementary – 9,268 meals\n"
    "    3. Chesterbrook Elementary – 10,418 meals\n"
    "    4. Cherry Run Elementary – 11,481 meals\n"
    "    5. Armstrong Elementary – 11,504 meals"
)

st.subheader("📌 Top 5 Schools with Highest Sales Variation")
st.markdown(
    "- **Insight:** Schools with the highest variation in daily sales (standard deviation):\n"
    "    1. Falls Church High – 256.5\n"
    "    2. Lynbrook Elementary – 127.2\n"
    "    3. Glasgow Middle – 121.0\n"
    "    4. Annandale High – 120.5\n"
    "    5. Lake Braddock Secondary – 119.4"
)

st.subheader("📌 Top 5 Schools with Lowest Sales Variation")
st.markdown(
    "- **Insight:** Schools with the most consistent daily sales (lowest standard deviation):\n"
    "    1. Armstrong Elementary – 12.0\n"
    "    2. Little Run Elementary – 13.1\n"
    "    3. Franklin Sherman Elementary – 14.0\n"
    "    4. Olde Creek Elementary – 15.1\n"
    "    5. Belle View Elementary – 16.7"
)
