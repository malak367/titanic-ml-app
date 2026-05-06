import streamlit as st
import pandas as pd

st.title("Titanic ML App")

tab1, tab2, tab3 = st.tabs(["Home", "Insights", "Prediction"])

df = pd.read_csv("train.csv")

with tab1:
    st.write("Welcome to Titanic Project")

with tab2:
    st.write(df.head())

with tab3:
    st.write("Prediction coming soon...")
