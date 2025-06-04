import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import gdown

# הורדת קובץ מהדרייב
@st.cache_data
def load_data():
    file_id = '1cZGBADyRRrARkTxLFvOiQ8NuVGBkD0mp'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'EV_data.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# קריאה והצגת טבלה
df = load_data()
st.title("Electric Vehicle Dataset Overview")
st.dataframe(df.head(50))

# גרף 1: פילוח EV stock share לשנת 2022
st.subheader("Distribution of EV Stock Share by Country (2022)")
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

# גרף 2: מכירות רכבים לפי שנה
st.subheader("Histogram of EV Car Sales by Year")
cars_sales = df[(df['mode'] == 'Cars') & (df['parameter'] == 'EV sales')]
sales_by_year = cars_sales.groupby('year')['value'].sum().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=sales_by_year, x='year', y='value', palette='viridis', ax=ax2)
ax2.set_title('Histogram of EV Car Sales by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Cars Sold')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# גרף 3: תוספת שנתית למלאי EV
st.subheader("Histogram of Annual EV Stock Increase (Cars)")
ev_stock_cars = df[(df['parameter'] == 'EV stock') & (df['mode'] == 'Cars')]
stock_by_year = ev_stock_cars.groupby('year')['value'].sum().reset_index()
stock_by_year['new_stock_added'] = stock_by_year['value'].diff()
fig3, ax3 = plt.subplots()
sns.histplot(stock_by_year['new_stock_added'].dropna(), bins=15, kde=True, color='teal', ax=ax3)
ax3.set_title('Histogram of Annual EV Stock Increase (Cars)')
ax3.set_xlabel('New EV Cars Added to the Global Fleet')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# גרף 4: מכירות לפי סוג הנעה
st.subheader("EV Sales by Year and Powertrain Type")
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
for power in sales_by_year_powertrain['powertrain'].unique():
    subset = sales_by_year_powertrain[sales_by_year_powertrain['powertrain'] == power]
    kde = gaussian_kde(subset['year'], weights=subset['value'])
    x_range = np.linspace(subset['year'].min(), subset['year'].max(), 200)
    kde_vals = kde(x_range)
    kde_scaled = kde_vals * max(subset['value']) / max(kde_vals)
    ax4.plot(x_range, kde_scaled, label=f'{power} KDE', linewidth=2)
ax4.legend(title='Powertrain')
ax4.set_title("Total EV Sales by Year and Powertrain Type (with KDE)")
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Vehicles Sold")
st.pyplot(fig4)
