import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("electric_vehicle_data.csv")
    return df

df = load_data()

st.title("🔌 Electric Vehicle Data Analysis Dashboard")

# Histogram 1: EV Stock Share Distribution (2022)
st.subheader("1. Distribution of EV Stock Share by Country (2022)")
df_2022_ev_stock_share = df[
    (df['parameter'] == 'EV stock share') &
    (df['year'] == 2022) &
    (df['unit'] == 'percent')
]
fig1, ax1 = plt.subplots()
sns.histplot(df_2022_ev_stock_share['value'], bins=20, kde=True, ax=ax1)
ax1.set_xlabel('EV Stock Share (%)')
ax1.set_ylabel('Number of Countries')
ax1.set_title('Distribution of EV Stock Share by Country (2022)')
st.pyplot(fig1)

# Histogram 2: EV Sales by Year
st.subheader("2. Histogram of EV Car Sales by Year")
cars_sales = df[(df['mode'] == 'Cars') & (df['parameter'] == 'EV sales')]
sales_by_year = cars_sales.groupby('year')['value'].sum().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=sales_by_year, x='year', y='value', palette='viridis', ax=ax2)
ax2.set_title('Histogram of EV Car Sales by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Cars Sold')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Histogram 3: Annual EV Stock Increase
st.subheader("3. Histogram of Annual EV Stock Increase (Cars)")
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_year = ev_stock_cars.groupby('year')['value'].sum().reset_index()
stock_by_year['new_stock_added'] = stock_by_year['value'].diff()
fig3, ax3 = plt.subplots()
sns.histplot(stock_by_year['new_stock_added'].dropna(), bins=15, kde=True, color='teal', ax=ax3)
ax3.set_title('Histogram of Annual EV Stock Increase (Cars)')
ax3.set_xlabel('New EV Cars Added')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# Histogram 4: Sales by Powertrain
st.subheader("4. EV Sales by Powertrain and Year with KDE")
ev_sales = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')]
ev_sales['value'] = pd.to_numeric(ev_sales['value'], errors='coerce')
ev_sales = ev_sales.dropna(subset=['value'])
sales_by_year_powertrain = ev_sales.groupby(['year', 'powertrain'])['value'].sum().reset_index()
fig4, ax4 = plt.subplots()
sns.histplot(
    data=sales_by_year_powertrain,
    x='year',
    weights='value',
    hue='powertrain',
    multiple='stack',
    bins=len(sales_by_year_powertrain['year'].unique()),
    palette='muted',
    element='bars',
    ax=ax4
)
# KDE curves
for power in sales_by_year_powertrain['powertrain'].unique():
    subset = sales_by_year_powertrain[sales_by_year_powertrain['powertrain'] == power]
    kde = gaussian_kde(subset['year'], weights=subset['value'])
    x_range = np.linspace(subset['year'].min(), subset['year'].max(), 200)
    kde_vals = kde(x_range)
    kde_scaled = kde_vals * max(subset['value']) / max(kde_vals)
    ax4.plot(x_range, kde_scaled, label=f'{power} KDE', linewidth=2)
ax4.set_title("EV Sales by Year and Powertrain Type (with KDE)")
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Vehicles Sold")
ax4.legend(title="Powertrain")
st.pyplot(fig4)

# Scatter 1: Sales vs Share
st.subheader("5. EV Sales vs EV Sales Share")
ev_sales_cars = df[(df['parameter'] == 'EV sales') & (df['mode'] == 'Cars')]
ev_share = df[(df['parameter'] == 'EV sales share') & (df['mode'] == 'Cars')]
merged = pd.merge(
    ev_sales_cars[['region', 'year', 'value']],
    ev_share[['region', 'year', 'value']],
    on=['region', 'year'],
    suffixes=('_sales', '_share')
)
if merged['value_share'].max() <= 1:
    merged['value_share'] *= 100
fig5, ax5 = plt.subplots()
sns.scatterplot(data=merged, x='value_sales', y='value_share', hue='region', alpha=0.7, ax=ax5)
ax5.set_title('EV Sales vs EV Sales Share')
ax5.set_xlabel('EV Sales (Units)')
ax5.set_ylabel('EV Sales Share (%)')
ax5.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig5)

# Line Plot: EV Stock Trend by Region
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_region_year = ev_stock_cars.groupby(['region', 'year'])['value'].sum().reset_index()
fig6, ax6 = plt.subplots()
sns.lineplot(data=stock_by_region_year, x='year', y='value', hue='region', marker='o', ax=ax6)
ax6.set_title('Trend of EV Stock Over Time by Region')
ax6.set_xlabel('Year')
ax6.set_ylabel('EV Stock (Total Cars)')
ax6.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig6)

# Scatter 2: Sales vs Stock (log scale)
sales_df = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')][['region', 'year', 'powertrain', 'value']].rename(columns={'value': 'sales'})
stock_df = df[(df['parameter'] == 'EV stock') & (df['unit'] == 'Vehicles')][['region', 'year', 'powertrain', 'value']].rename(columns={'value': 'stock'})
merged_df = pd.merge(sales_df, stock_df, on=['region', 'year', 'powertrain'])
fig7, ax7 = plt.subplots()
sns.scatterplot(data=merged_df, x='sales', y='stock', hue='powertrain', ax=ax7)
ax7.set_title("EV Sales vs Stock by Powertrain")
ax7.set_xlabel("EV Sales")
ax7.set_ylabel("EV Stock")
ax7.set_xscale("log")
ax7.set_yscale("log")
st.pyplot(fig7)
