
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

st.set_page_config(layout="wide")
st.title("🔌 Electric Vehicle Data Dashboard")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/electric_vehicle_data.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Histogram: EV Stock Share by Country (2022)
st.subheader("Histogram: EV Stock Share by Country (2022)")
df_2022 = df[(df['parameter'] == 'EV stock share') & (df['year'] == 2022) & (df['unit'] == 'percent')]
fig1, ax1 = plt.subplots()
sns.histplot(df_2022['value'], bins=20, kde=True, ax=ax1)
ax1.set_xlabel("EV Stock Share (%)")
ax1.set_ylabel("Number of Countries")
st.pyplot(fig1)
