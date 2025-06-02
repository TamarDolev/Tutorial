import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

st.set_page_config(page_title="EV Market Analysis", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/IEA-EV-dataEV-salesHistoricalCars.csv"
    df = pd.read_csv(url)
    return df

# Load and preview data
df = load_data()
st.title("Electric Vehicle Data Analysis")

st.subheader("Raw Data Preview")
st.dataframe(df.head(10))

# Histogram 1: EV Stock Share Distribution (2022)
st.subheader("1. Distribution of EV Stock Share by Country (2022)")
df_2022 = df[(df['parameter'] == 'EV stock share') & (df['year'] == 2022) & (df['unit'] == 'percent')]
fig1, ax1 = plt.subplots()
sns.histplot(df_2022['value'], bins=20, kde=True, ax=ax1)
ax1.set_xlabel("EV Stock Share (%)")
ax1.set_ylabel("Number of Countries")
ax1.set_title("EV Stock Share (2022)")
st.pyplot(fig1)

# Histogram 2: EV Sales by Year (Bar Chart)
st.subheader("2. EV Car Sales by Year")
cars_sales = df[(df['mode'] == 'Cars') & (df['parameter'] == 'EV sales')]
sales_by_year = cars_sales.groupby('year')['value'].sum().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=sales_by_year, x='year', y='value', palette='viridis', ax=ax2)
ax2.set_title("EV Car Sales by Year")
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Cars Sold")
st.pyplot(fig2)

# Histogram 3: Annual EV Stock Increase
st.subheader("3. Annual EV Stock Increase (Cars)")
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_year = ev_stock_cars.groupby('year')['value'].sum().reset_index()
stock_by_year['new_stock_added'] = stock_by_year['value'].diff()
fig3, ax3 = plt.subplots()
sns.histplot(stock_by_year['new_stock_added'].dropna(), bins=15, kde=True, color='teal', ax=ax3)
ax3.set_title("Annual EV Stock Increase")
ax3.set_xlabel("New Cars Added")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

# Histogram 4: EV Sales by Year and Powertrain
st.subheader("4. EV Sales by Powertrain Type")
ev_sales = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')]
ev_sales['value'] = pd.to_numeric(ev_sales['value'], errors='coerce')
ev_sales = ev_sales.dropna(subset=['value'])
sales_by_year_pt = ev_sales.groupby(['year', 'powertrain'])['value'].sum().reset_index()
fig4, ax4 = plt.subplots()
sns.histplot(
    data=sales_by_year_pt,
    x='year',
    weights='value',
    hue='powertrain',
    multiple='stack',
    bins=len(sales_by_year_pt['year'].unique()),
    palette='muted',
    element='bars',
    ax=ax4
)
ax4.set_title("EV Sales by Year and Powertrain")
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Vehicles Sold")
st.pyplot(fig4)

# Scatter Plot: EV Sales vs. EV Share
st.subheader("5. EV Sales vs. EV Sales Share")
ev_sales = df[(df['parameter'] == 'EV sales') & (df['mode'] == 'Cars')]
ev_share = df[(df['parameter'] == 'EV sales share') & (df['mode'] == 'Cars')]
merged = pd.merge(ev_sales[['region', 'year', 'value']],
                  ev_share[['region', 'year', 'value']],
                  on=['region', 'year'],
                  suffixes=('_sales', '_share'))
if merged['value_share'].max() <= 1:
    merged['value_share'] *= 100
fig5, ax5 = plt.subplots()
sns.scatterplot(data=merged, x='value_sales', y='value_share', hue='region', alpha=0.7, ax=ax5)
ax5.set_title("EV Sales vs. Share")
ax5.set_xlabel("EV Sales (Units)")
ax5.set_ylabel("EV Share (%)")
st.pyplot(fig5)

# Line Plot: EV Stock Over Time by Region
st.subheader("6. EV Stock Over Time by Region")
ev_stock = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_region_year = ev_stock.groupby(['region', 'year'])['value'].sum().reset_index()
fig6, ax6 = plt.subplots()
sns.lineplot(data=stock_by_region_year, x='year', y='value', hue='region', marker='o', ax=ax6)
ax6.set_title("EV Stock Trend by Region")
ax6.set_xlabel("Year")
ax6.set_ylabel("Total EV Stock")
st.pyplot(fig6)

st.markdown("---")
st.info("Built with Streamlit · Data from IEA EV dataset · Visualizations powered by Seaborn + Matplotlib")
