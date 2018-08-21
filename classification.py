import pandas as pd
import numpy as np
import tensorflow as tf

diabetes = pd.read_csv('pima-indians-diabetes.csv')

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

# Vocabulary List

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
