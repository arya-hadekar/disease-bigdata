import streamlit as st
import pandas as pd

st.title("Disease Analytics Dashboard")

df = pd.read_csv("../backend/data/dataset.csv")

st.write("Dataset Preview", df.head())

st.bar_chart(df.iloc[:, :-1].sum())