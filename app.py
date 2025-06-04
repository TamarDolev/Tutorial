
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import gdown

st.set_page_config(layout="wide")
st.title("Electric Vehicle Analysis Dashboard")

@st.cache_data
def load_data():
    file_id = '1cZGBADyRRrARkTxLFvOiQ8NuVGBkD0mp'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'EV_data.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()

# Plot 1: EV stock share distribution
st.subheader("1. Distribution of EV Stock Share by Country (2022)")
df_2022_ev_stock_share = df[(df['parameter'] == 'EV stock share') & (df['year'] == 2022) & (df['unit'] == 'percent')]
fig1, ax1 = plt.subplots()
sns.histplot(df_2022_ev_stock_share['value'], bins=20, kde=True, ax=ax1)
ax1.set_xlabel('EV Stock Share (%)')
ax1.set_ylabel('Number of Countries')
ax1.set_title('Distribution of EV Stock Share by Country (2022)')
st.pyplot(fig1)

# Plot 2: EV Car Sales by Year
st.subheader("2. Histogram of EV Car Sales by Year")
cars_sales = df[(df['mode'] == 'Cars') & (df['parameter'] == 'EV sales')]
sales_by_year = cars_sales.groupby('year')['value'].sum().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=sales_by_year, x='year', y='value', palette='viridis', ax=ax2)
ax2.set_title('EV Car Sales by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Cars Sold')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Plot 3: Annual EV Stock Increase
st.subheader("3. Histogram of Annual EV Stock Increase")
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_year = ev_stock_cars.groupby('year')['value'].sum().reset_index()
stock_by_year['new_stock_added'] = stock_by_year['value'].diff()
fig3, ax3 = plt.subplots()
sns.histplot(stock_by_year['new_stock_added'].dropna(), bins=15, kde=True, color='teal', ax=ax3)
ax3.set_title('Annual EV Stock Increase (Cars)')
ax3.set_xlabel('New EV Cars Added')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# Plot 4: EV Sales by Powertrain
st.subheader("4. Total EV Sales by Year and Powertrain Type")
ev_sales = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')]
ev_sales['value'] = pd.to_numeric(ev_sales['value'], errors='coerce')
ev_sales = ev_sales.dropna(subset=['value'])
sales_by_year_powertrain = ev_sales.groupby(['year', 'powertrain'])['value'].sum().reset_index()
fig4, ax4 = plt.subplots()
sns.histplot(data=sales_by_year_powertrain, x='year', weights='value', hue='powertrain',
             multiple='stack', bins=len(sales_by_year_powertrain['year'].unique()), palette='muted',
             element='bars', ax=ax4)
for power in sales_by_year_powertrain['powertrain'].unique():
    subset = sales_by_year_powertrain[sales_by_year_powertrain['powertrain'] == power]
    kde = gaussian_kde(subset['year'], weights=subset['value'])
    x_range = np.linspace(subset['year'].min(), subset['year'].max(), 200)
    kde_vals = kde(x_range)
    kde_scaled = kde_vals * max(subset['value']) / max(kde_vals)
    ax4.plot(x_range, kde_scaled, label=f'{power} KDE', linewidth=2)
ax4.set_title("EV Sales by Year and Powertrain Type (KDE)")
ax4.set_xlabel("Year")
ax4.set_ylabel("Vehicles Sold")
ax4.legend(title="Powertrain")
st.pyplot(fig4)
