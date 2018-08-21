import pandas as pd
import numpy as np
import tensorflow as tf

# Get Training Progress

import logging
logging.getLogger().setLevel(logging.INFO)

from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('pima-indians-diabetes.csv')

# Normalize Data

cols_to_norm = ['Number_pregnant', 'Glucose_concentration',  'Blood_pressure',
                'Triceps', 'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Vocabulary List - Write categories manually

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list(
    'Group', ['A', 'B', 'C', 'D'])

# Hash Bucket - Make categories automatically

assigned_group = tf.feature_column.categorical_column_with_hash_bucket(
    'Group', hash_bucket_size=10)

# Bucket continuous values into categories

age_bucket = tf.feature_column.bucketized_column(
    age, boundaries=[20, 30, 40, 50, 60, 70, 80])

# Combine to form master Feature Columns

feat_cols = [num_preg, plasma_gluc, dias_press, tricep,
             insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]

# Train Test Split

x_data = diabetes.drop('Class', axis=1)

labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(
    x_data, labels, test_size=0.3, random_state=101)

# Creating Input Function

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Defining Model

model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

# Training Model

model.train(input_fn=input_func, steps=1000)

# Testing Model

eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(eval_input_func)

# Testing on New Data

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)

my_pred = list(predictions)

# Convert Feature Columns to DenseColumn for Input to a DNN

embedded_group_col = tf.feature_column.embedding_column(
    assigned_group, dimension=4)

feat_cols = [num_preg, plasma_gluc, dias_press, tricep,
             insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]

# Creating Input Function

input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Defining Model

dnn_model = tf.estimator.DNNClassifier(
    hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)

# Training Model

dnn_model.train(input_fn=input_func, steps=1000)

# Testing Model

eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(eval_input_func)

# Testing on New Data

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)

my_pred = list(predictions)
