import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="ImageNet Sales Dashboard", layout="wide")
st.title("📊 ImageNet Category Sales Dashboard")

# Load CSS styling
def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ {file_name} not found!")

load_local_css("style.css")
# Load Data
df = pd.read_csv("imagenet_sales.csv", parse_dates=["date"])

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    selected_categories = st.multiselect("Select ImageNet Categories", df["category"].unique(), default=list(df["category"].unique()))
    start_date = st.date_input("Start Date", df["date"].min())
    end_date = st.date_input("End Date", df["date"].max())
    show_charts = st.multiselect("Select Charts to Display", [
        "Line Chart",
        "Bar Chart",
        "Area Chart",
        "Pie Chart",
        "Box Plot",
        "Multi-Line Chart",
        "Heatmap",
    ], default=["Line Chart", "Bar Chart", "Area Chart", "Pie Chart", "Box Plot", "Multi-Line Chart", "Heatmap"])

# Filter data
filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# KPIs
st.metric("📦 Total Sales", int(filtered_df["sales"].sum()))
st.metric("📈 Average Daily Sales", round(filtered_df["sales"].mean(), 2))

# Charts
if "Line Chart" in show_charts:
    st.subheader("📈 Line Chart")
    cols = st.columns(len(selected_categories))

    for i, category in enumerate(selected_categories):
        chart_data = filtered_df[filtered_df["category"] == category]
        fig = px.line(chart_data, x="date", y="sales", title=f"{category}", markers=True)
        cols[i].plotly_chart(fig, use_container_width=True)

if "Bar Chart" in show_charts:
    st.subheader("📊 Bar Chart")
    bar_fig = px.bar(filtered_df, x="category", y="sales", color="category", barmode="group", title="Sales by Category")
    st.plotly_chart(bar_fig, use_container_width=True)

if "Area Chart" in show_charts:
    st.subheader("📊 Area Chart")
    area_fig = px.area(filtered_df, x="date", y="sales", color="category", title="Sales Volume Over Time")
    st.plotly_chart(area_fig, use_container_width=True)

if "Pie Chart" in show_charts:
    st.subheader("📎 Pie Chart")
    pie_data = filtered_df.groupby("category")["sales"].sum().reset_index()
    pie_fig = px.pie(pie_data, names="category", values="sales", hole=0.4, title="Sales Distribution by Category")
    st.plotly_chart(pie_fig, use_container_width=True)

if "Box Plot" in show_charts:
    st.subheader("📦 Box Plot")
    box_fig = px.box(filtered_df, x="category", y="sales", title="Sales Distribution per Category")
    st.plotly_chart(box_fig, use_container_width=True)

if "Multi-Line Chart" in show_charts:
    st.subheader("📈 Multi-Line Chart")
    multi_line_fig = px.line(filtered_df, x="date", y="sales", color="category", title="Category Comparison Over Time")
    st.plotly_chart(multi_line_fig, use_container_width=True)

if "Heatmap" in show_charts:
    st.subheader("📅 Heatmap (Pivot Table)")
    pivot = filtered_df.pivot_table(index="category", columns="date", values="sales", aggfunc="sum", fill_value=0)
    st.dataframe(pivot.style.background_gradient(cmap='Blues'), use_container_width=True)

# Data Table
st.subheader("🔎 Filtered Sales Data")
st.dataframe(filtered_df)
