import streamlit as st

st.header('California App Predictions')
st.write("""
This app predicts the **California House Price**!
""")
st.write('---')

import pandas as pd

df = pd.read_csv('housing.csv')
df
