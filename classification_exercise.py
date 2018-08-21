import pandas as pd
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

data = pd.read_csv('census_data.csv')

cols_to_norm = ['age', 'education_num',
                'capital_gain', 'capital_loss', 'hours_per_week']

data[cols_to_norm] = data[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))


def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1


data['income_bracket'] = data['income_bracket'].apply(label_fix)

w_class = tf.feature_column.categorical_column_with_hash_bucket(
    key='workclass', hash_bucket_size=1000)
edu = tf.feature_column.categorical_column_with_hash_bucket(
    key='education', hash_bucket_size=1000)
marstat = tf.feature_column.categorical_column_with_hash_bucket(
    key='marital_status', hash_bucket_size=1000)
occ = tf.feature_column.categorical_column_with_hash_bucket(
    key='occupation', hash_bucket_size=1000)
rel = tf.feature_column.categorical_column_with_hash_bucket(
    key='relationship', hash_bucket_size=1000)
per_race = tf.feature_column.categorical_column_with_hash_bucket(
    key='race', hash_bucket_size=1000)
gend = tf.feature_column.categorical_column_with_hash_bucket(
    key='gender', hash_bucket_size=1000)
country = tf.feature_column.categorical_column_with_hash_bucket(
    key='native_country', hash_bucket_size=1000)
per_age = tf.feature_column.numeric_column(key='age')
edu_num = tf.feature_column.numeric_column(key='education_num')
cap_gain = tf.feature_column.numeric_column(key='capital_gain')
cap_loss = tf.feature_column.numeric_column(key='capital_loss')
hpw = tf.feature_column.numeric_column(key='hours_per_week')

feat_cols = [per_age, edu_num, edu, cap_gain, cap_loss, hpw,
             country, gend, w_class, marstat, occ, rel, per_race]

x_data = data.drop('income_bracket', axis=1)

labels = data['income_bracket']

X_train, X_test, Y_train, Y_test = train_test_split(
    x_data, labels, test_size=0.3)

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=Y_train, batch_size=100, num_epochs=None, shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

model.train(input_fn=input_func, steps=10000)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=len(X_test), shuffle=False)

pred_gen = model.predict(pred_input_func)

predictions = list(pred_gen)

final_preds = [pred['class_ids'][0] for pred in predictions]

print(classification_report(Y_test, final_preds))
