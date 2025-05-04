import streamlit as st
import pandas as pd
import sklearn
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px

st.header('California App Predictions')
st.write("""
This app predicts the **California House Price**!
""")
st.write('---')


housing = pd.read_csv('housing.csv')
housing

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

col_names = ['total_rooms','total_bedrooms','population', 'households']
rooms_id, bedrooms_id, population_id, households_id = [
    housing.columns.get_loc(c) for c in col_names]
# print(rooms_id, bedrooms_id, population_id, households_id ) 




from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self

    def transform(self,X):
        rooms_per_household = X[:, rooms_id] / X[:, households_id]
        population_per_household = X[:, population_id] / X[:, households_id]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_id] / X[:, rooms_id]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

num_attr = X.select_dtypes(include=np.number).columns.to_list()
cat_attr = housing.select_dtypes(exclude=np.number).columns.to_list()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


from sklearn.preprocessing import OneHotEncoder


cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder())
])


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr),
     ("cat", cat_pipeline, cat_attr),
])

full_pipeline.fit(X_train)
print(housing.shape)
model = joblib.load("Decision_tree_regressor.joblib")



fig = px.scatter_map(housing, 
                     lat="latitude", 
                     lon="longitude", 
                     size="population", # Size of markers based on population
                     )
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter_mapbox(housing, lat='latitude',
                        lon='longitude',
                        hover_data = ['housing_median_age', 'total_rooms', 'total_bedrooms',
                                     'population','households','median_income','median_house_value'],
                        color = housing['median_house_value'],
                        color_continous_scale=px.colors.sequential.Agsunset,
                        size=housing["population"]/1e5,
                        zoom=4, height=500)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0, "t":0,"l":0,"b":0})

st.plotly_chart(fig,use_container_width=True)

                        
