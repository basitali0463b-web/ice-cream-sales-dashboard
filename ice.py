import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Ice Cream Sales EDA",
    page_icon="🍦",
    layout="wide"
)

st.title("🍦 Ice Cream Sales Dashboard")
st.write("Explore the relationship between temperature and ice cream revenue.")

# Find the CSV next to this Python file
DATA_PATH = Path(__file__).parent / "Ice Cream.csv"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load local CSV, or let the user upload it
if DATA_PATH.exists():
    df = load_data(str(DATA_PATH))
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Ice Cream CSV file", type=["csv"]
    )
    if uploaded_file is None:
        st.info("Place 'Ice Cream.csv' beside ice.py, or upload it using the sidebar.")
        st.stop()
    df = pd.read_csv(uploaded_file)

# Validate expected columns
required_columns = {"Temperature", "Revenue"}
if not required_columns.issubset(df.columns):
    st.error(
        "The CSV must contain these columns: Temperature and Revenue. "
        f"Found: {df.columns.tolist()}"
    )
    st.stop()

# Convert to numeric and remove invalid rows for analysis
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
invalid_rows = int(df[["Temperature", "Revenue"]].isna().any(axis=1).sum())
df = df.dropna(subset=["Temperature", "Revenue"])

if df.empty:
    st.error("No valid numeric rows were found in the CSV.")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")
temp_min = float(df["Temperature"].min())
temp_max = float(df["Temperature"].max())

if temp_min < temp_max:
    selected_range = st.sidebar.slider(
        "Temperature range",
        min_value=temp_min,
        max_value=temp_max,
        value=(temp_min, temp_max)
    )
    filtered_df = df[df["Temperature"].between(*selected_range)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No rows match the selected temperature range.")
    st.stop()

# Summary metrics
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{len(filtered_df):,}")
col2.metric("Average Temperature", f"{filtered_df['Temperature'].mean():.2f}")
col3.metric("Average Revenue", f"{filtered_df['Revenue'].mean():.2f}")
correlation = filtered_df["Temperature"].corr(filtered_df["Revenue"])
col4.metric("Temperature–Revenue Correlation",
            f"{correlation:.3f}" if pd.notna(correlation) else "N/A")

# Data preview
with st.expander("📋 Dataset Preview", expanded=True):
    st.dataframe(filtered_df, use_container_width=True)

# Data quality
st.subheader("🔍 Data Quality")
quality_col1, quality_col2, quality_col3 = st.columns(3)
quality_col1.metric("Missing values in source data", int(df.isna().sum().sum()))
quality_col2.metric("Duplicate rows", int(df.duplicated().sum()))
quality_col3.metric("Rows excluded for invalid numeric data", invalid_rows)

st.subheader("📊 Statistical Summary")
st.dataframe(filtered_df.describe().round(2), use_container_width=True)

# Scatter plot
st.subheader("🌡️ Temperature vs. Revenue")
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=filtered_df, x="Temperature", y="Revenue", ax=ax)
ax.set_title("Temperature vs. Ice Cream Revenue")
ax.set_xlabel("Temperature")
ax.set_ylabel("Revenue")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

# Histograms
st.subheader("📈 Numerical Distributions")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(filtered_df["Temperature"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Temperature Distribution")
sns.histplot(filtered_df["Revenue"], bins=20, kde=True, ax=axes[1])
axes[1].set_title("Revenue Distribution")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    filtered_df[["Temperature", "Revenue"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    ax=ax
)
ax.set_title("Correlation Between Variables")
st.pyplot(fig)
plt.close(fig)

# Revenue pie chart: distribution of records across equal-width revenue bins
st.subheader("🥧 Revenue Category Pie Chart")
revenue_bins = pd.cut(
    filtered_df["Revenue"],
    bins=3,
    labels=["Low Revenue", "Medium Revenue", "High Revenue"],
    duplicates="drop"
)
revenue_counts = revenue_bins.value_counts().sort_index()
revenue_counts = revenue_counts[revenue_counts > 0]

if not revenue_counts.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        revenue_counts.values,
        labels=[str(label) for label in revenue_counts.index],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Share of Records by Revenue Category")
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Not enough revenue variation to create revenue categories.")

# Download filtered data
st.subheader("⬇️ Download")
st.download_button(
    label="Download filtered CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="ice_cream_sales_filtered.csv",
    mime="text/csv"
)

st.caption("EDA dashboard for the Ice Cream Sales dataset.")
