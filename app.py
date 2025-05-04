import streamlit as st

st.header('California App Predictions')
st.write("""
This app predicts the **California House Price**!
""")
st.write('---')

import pandas as pd

df = pd.read_csv('housing.csv')
df

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

col_names = ['total_rooms','total_bedrooms','population', 'households']
rooms_id, bedrooms_id, population_id, households_id = [
    housing.columns.get_loc(c) for c in col_names]
print(rooms_id, bedrooms_id, population_id, households_id ) 

