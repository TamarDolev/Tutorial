import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import gdown

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

# 🧾 הצגת טבלת הנתונים בראש העמוד
st.subheader("📊 Raw Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)

# 🎨 גרף Stackplot – סך המכירות לפי סוג הנעה
st.subheader("📈 EV Sales by Powertrain Type (World)")

# סינון
ev_sales_world = df[
    (df['parameter'] == 'EV sales') &
    (df['mode'] == 'Cars') &
    (df['region'] == 'World')
]

# סכימה
ev_sales_by_powertrain = ev_sales_world.groupby(['year', 'powertrain'])['value'].sum().reset_index()

# מיפוי מחדש
pivot_ev_sales = ev_sales_by_powertrain.pivot(index='year', columns='powertrain', values='value').fillna(0)
pivot_ev_sales = pivot_ev_sales.sort_index()

# גרף
fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(
    pivot_ev_sales.index,
    [pivot_ev_sales['BEV'], pivot_ev_sales['PHEV']],
    labels=['BEV', 'PHEV']
)

ax.set_title('Total EV Sales by Powertrain Type (World)', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Vehicles Sold')
ax.legend(loc='upper left', title='Powertrain')
ax.grid(True)
st.pyplot(fig)
