import os
import pandas as pd
import sys
from sklearn import preprocessing as preprocessing
import numpy as np
from sklearn.svm import SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyGPs
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

# Read training data
path = "./ponderosa-pine-gaussian-process/"
filename_read = os.path.join(path, "train.csv")
data_train = pd.read_csv(filename_read)

# Preprocessing
data_train.drop('ID', 1, inplace=True)

#soil_type
climatic = (data_train['Soil_Type']/1000).astype(int)
geologic = (((data_train['Soil_Type'] - 1000 * climatic).astype(int)) / 100).astype(int)
mapping = (data_train['Soil_Type'] - 1000 * climatic - 100 * geologic).astype(int)

dummies_climatic = pd.get_dummies(climatic, prefix='climatic')
dummies_geologic = pd.get_dummies(geologic, prefix='geologic')
dummies_mapping = pd.get_dummies(mapping,prefix='mapping')
mapping_type=list(dummies_mapping)

df = (pd.concat([data_train, dummies_climatic, dummies_geologic, dummies_mapping], axis=1)).astype(float)
df.drop(['Soil_Type'], axis=1, inplace=True)

#Aspect
df['Cos_Aspect'] = np.cos(df['Aspect'])
df['Sin_Aspect'] = np.sin(df['Aspect'])
df.drop('Aspect', 1, inplace=True)

#Standardize
scaler = preprocessing.StandardScaler()
scaler_Elevation = scaler.fit(pd.DataFrame(df['Elevation']))
scaler_Slope = scaler.fit(pd.DataFrame(df['Slope']))
scaler_HDTH = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
scaler_VDTH = scaler.fit(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
scaler_HDTR = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
scaler_H9 = scaler.fit(pd.DataFrame(df['Hillshade_9am']))
scaler_HN = scaler.fit(pd.DataFrame(df['Hillshade_Noon']))
scaler_H3 = scaler.fit(pd.DataFrame(df['Hillshade_3pm']))
scaler_Dis_to_fire = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Fire_Points']))
df['Elevation'] = scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Slope'] = scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology'] = scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology'] = scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Horizontal_Distance_To_Roadways'] = scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am'] = scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon'] = scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm'] = scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))
df['Horizontal_Distance_To_Fire_Points'] = scaler_Dis_to_fire.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Fire_Points']))

#generate training data
Ytrain = df['From_Cache_la_Poudre'].values
df.drop('From_Cache_la_Poudre', 1, inplace=True)
Xtrain = df.values
Ytrain =Ytrain * 2 - 1

#train
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(Xtrain, Ytrain)

# Read test data
path = "./ponderosa-pine-gaussian-process/"
filename_read = os.path.join(path, "test.csv")
data_test = pd.read_csv(filename_read)

#Preprocessing
IDs = data_test['ID']
data_test.drop('ID', 1, inplace=True)

#soil_type
climatic = (data_test['Soil_Type']/1000).astype(int)
geologic = (((data_test['Soil_Type'] - 1000 * climatic).astype(int)) / 100).astype(int)
mapping = (data_test['Soil_Type'] - 1000 * climatic - 100 * geologic).astype(int)

dummies_climatic = pd.get_dummies(climatic, prefix='climatic')
dummies_geologic = pd.get_dummies(geologic, prefix='geologic')
dummies_mapping = pd.get_dummies(mapping,prefix='mapping')
for name in dummies_mapping:
	if name not in mapping_type:
		dummies_mapping.drop(name, 1, inplace=True)
df = (pd.concat([data_test, dummies_climatic, dummies_geologic, dummies_mapping], axis=1)).astype(float)
df.drop(['Soil_Type'], axis=1, inplace=True)

#Aspect
df['Cos_Aspect'] = np.cos(df['Aspect'])
df['Sin_Aspect'] = np.sin(df['Aspect'])
df.drop('Aspect', 1, inplace=True)

#Standardize
df['Elevation'] = scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Slope'] = scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology'] = scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology'] = scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Horizontal_Distance_To_Roadways'] = scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am'] = scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon'] = scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm'] = scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))
df['Horizontal_Distance_To_Fire_Points'] = scaler_Dis_to_fire.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Fire_Points']))

#generate test data
Xtest = df.values

#get prediction
pred = gpc.predict_proba(Xtest)
output = pred[:,1]
submit_df = pd.DataFrame()
submit_df['ID'] = IDs
submit_df['From_Cache_la_Poudre'] = output
submit_df.to_csv('GP_prediction.csv', index=False)