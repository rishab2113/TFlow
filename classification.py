import pandas as pd
import numpy as np
import tensorflow as tf

diabetes = pd.read_csv('pima-indians-diabetes.csv')

print(diabetes.head())

# Normalize Data

cols_to_norm = ['Number_pregnant', 'Glucose_concentration',  'Blood_pressure',
       'Triceps', 'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Vocabulary List - Write categories manually

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

# Hash Bucket - Make categories automatically

assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)

# Bucket continuous values into categories

age_bucket = tf.feature_column.bucketized_column(age,boundaries = [20,30,40,50,60,70,80])

# Combine to form master Feature Columns

feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]

# Train Test Split

x_data = diabetes.drop('Class', axis=1)
