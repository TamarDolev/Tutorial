import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import gdown
from scipy.stats import gaussian_kde

# קונפיגורציה
st.set_page_config(layout="wide")
st.title("🔌 Electric Vehicles Data Analysis Dashboard")

# הורדת הקובץ מגוגל דרייב
@st.cache_data
def load_data():
    file_id = "1cZGBADyRRrARkTxLFvOiQ8NuVGBkD0mp"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "EV_data.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# טען את הנתונים
df = load_data()

# 🧾 הצגת טבלת הנתונים
st.subheader("📊 Raw Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)

# גרף 1: התפלגות EV stock share לשנת 2022
st.subheader("📌 Distribution of EV Stock Share by Country (2022)")
df_2022_ev_stock_share = df[(df['parameter'] == 'EV stock share') & (df['year'] == 2022) & (df['unit'] == 'percent')]
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(df_2022_ev_stock_share['value'], bins=20, kde=True, ax=ax1)
ax1.set_xlabel('EV Stock Share (%)')
ax1.set_ylabel('Number of Countries')
ax1.set_title('Distribution of EV Stock Share by Country (2022)')
ax1.grid(True)
st.pyplot(fig1)

# גרף 2: סך מכירות רכבים פרטיים לפי שנה
st.subheader("📌 Total EV Car Sales by Year")
cars_sales = df[(df['mode'] == 'Cars') & (df['parameter'] == 'EV sales')]
sales_by_year = cars_sales.groupby('year')['value'].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=sales_by_year, x='year', y='value', palette='viridis', ax=ax2)
ax2.set_title('Histogram of EV Car Sales by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Cars Sold')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# גרף 3: שינוי מלאי רכבים חשמליים
st.subheader("📌 Annual EV Stock Increase (Cars)")
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_year = ev_stock_cars.groupby('year')['value'].sum().reset_index()
stock_by_year['new_stock_added'] = stock_by_year['value'].diff()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.histplot(stock_by_year['new_stock_added'].dropna(), bins=15, kde=True, color='teal', ax=ax3)
ax3.set_title('Histogram of Annual EV Stock Increase (Cars)')
ax3.set_xlabel('New EV Cars Added to the Global Fleet')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# גרף 4: סך מכירות לפי powertrain
st.subheader("📌 EV Sales by Year and Powertrain Type with KDE")
ev_sales = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')].copy()
ev_sales['value'] = pd.to_numeric(ev_sales['value'], errors='coerce')
ev_sales = ev_sales.dropna(subset=['value'])
sales_by_year_powertrain = ev_sales.groupby(['year', 'powertrain'])['value'].sum().reset_index()
fig4, ax4 = plt.subplots(figsize=(12, 6))
sns.histplot(data=sales_by_year_powertrain, x='year', weights='value', hue='powertrain', multiple='stack', bins=len(sales_by_year_powertrain['year'].unique()), palette='muted', element='bars', ax=ax4)
for power in sales_by_year_powertrain['powertrain'].unique():
    subset = sales_by_year_powertrain[sales_by_year_powertrain['powertrain'] == power]
    kde = gaussian_kde(subset['year'], weights=subset['value'])
    x_range = np.linspace(subset['year'].min(), subset['year'].max(), 200)
    kde_vals = kde(x_range)
    kde_scaled = kde_vals * max(subset['value']) / max(kde_vals)
    ax4.plot(x_range, kde_scaled, label=f'{power} KDE', linewidth=2)
ax4.set_title("Total EV Sales by Year and Powertrain Type (with KDE)")
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Vehicles Sold")
ax4.legend(title="Powertrain")
st.pyplot(fig4)

# גרף 5: Stackplot לפי סוג הנעה
st.subheader("📌 Total EV Sales by Powertrain Type (World)")
ev_sales_world = df[(df['parameter'] == 'EV sales') & (df['mode'] == 'Cars') & (df['region'] == 'World')]
ev_sales_by_powertrain = ev_sales_world.groupby(['year', 'powertrain'])['value'].sum().reset_index()
pivot_ev_sales = ev_sales_by_powertrain.pivot(index='year', columns='powertrain', values='value').fillna(0).sort_index()
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.stackplot(pivot_ev_sales.index, [pivot_ev_sales['BEV'], pivot_ev_sales['PHEV']], labels=['BEV', 'PHEV'])
ax5.set_title('Total EV Sales by Powertrain Type (World)', fontsize=14)
ax5.set_xlabel('Year')
ax5.set_ylabel('Number of Vehicles Sold')
ax5.legend(loc='upper left', title='Powertrain')
ax5.grid(True)
st.pyplot(fig5)
