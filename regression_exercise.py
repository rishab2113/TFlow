import pandas as pd
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('cal_housing_clean.csv')

to_normalize = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
       'households', 'medianIncome', 'medianHouseValue']

data[to_normalize] = data[to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

med_age = tf.feature_column.numeric_column('housingMedianAge')
tot_rooms = tf.feature_column.numeric_column('totalRooms')
tot_bedrooms = tf.feature_column.numeric_column('totalBedrooms')
popul = tf.feature_column.numeric_column('population')
household = tf.feature_column.numeric_column('households')
med_income = tf.feature_column.numeric_column('medianIncome')
med_house_value = tf.feature_column.numeric_column('medianHouseValue')

feat_cols = [med_age,tot_rooms,tot_bedrooms,popul,household,med_income]

x_data = data.drop('medianHouseValue', axis=1)

labels = data['medianHouseValue']

X_train, X_test, Y_train, Y_test = train_test_split(x_data,labels,test_size=0.3)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=Y_train, batch_size=10, num_epochs=10, shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)

model.train(input_fn=input_func, steps=1000)

test_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=Y_test, batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(test_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)

final_preds = []

for pred in predictions:
    final_preds.append(pred['predictions'])

print(mean_squared_error(Y_test, final_preds)**0.5)
