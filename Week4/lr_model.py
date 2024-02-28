import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Property_Prices.csv')

dataset.loc[[13, 16], 'Floor Area(ft2)'] = 400
no_missing_area_data = dataset.loc[:, ['Bedrooms', 'Bathrooms', 'Floor Area(ft2)']].dropna(subset=['Floor Area(ft2)'])
missing_area_data = dataset.loc[:, ['Bedrooms', 'Bathrooms', 'Floor Area(ft2)']][dataset['Floor Area(ft2)'].isnull()]
no_missing_area_predictors = no_missing_area_data.drop(columns=['Floor Area(ft2)'])
no_missing_area_targets = no_missing_area_data['Floor Area(ft2)']
missing_area_predictors = missing_area_data.drop(columns=['Floor Area(ft2)'])

mv_model = LinearRegression()
mv_model.fit(no_missing_area_predictors, no_missing_area_targets)
predicted_areas = mv_model.predict(missing_area_predictors)
dataset.loc[dataset['Floor Area(ft2)'].isnull(), ['Floor Area(ft2)']] = predicted_areas

encoding_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
dataset['Council Tax Band'] = dataset['Council Tax Band'].map(encoding_dict)

mode_by_bedrooms = dataset.groupby('Bedrooms')['Council Tax Band'].transform(lambda x: x.mode().iloc[0])
dataset['Council Tax Band'] = dataset['Council Tax Band'].fillna(mode_by_bedrooms)

model_predictors = dataset.iloc[:, :4].values
model_targets = dataset.iloc[:, -1]
regressor = LinearRegression()
regressor.fit(model_predictors, model_targets)
pickle.dump(regressor, open('lr_model.pkl', 'wb'))

mv_model = pickle.load(open('lr_model.pkl', 'rb'))
print(mv_model.predict([[2, 1, 800, 1]]))
