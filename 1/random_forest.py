import os
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing as preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


#Read training data
path = "./distance-to-fire-points/"
filename_read = os.path.join(path,"train.csv")
data_train = pd.read_csv(filename_read)


#Preprocessing
data_train.drop('ID',1,inplace=True)
dummies_soilType = pd.get_dummies(data_train['Soil_Type'],prefix='soilType')
soil_type=list(dummies_soilType)
df = (pd.concat([data_train,dummies_soilType],axis=1)).astype(float)
df.drop(['Soil_Type'],axis=1,inplace=True)

scaler = preprocessing.StandardScaler()
scaler_Elevation = scaler.fit(pd.DataFrame(df['Elevation']))
scaler_Aspect = scaler.fit(pd.DataFrame(df['Aspect']))
scaler_Slope = scaler.fit(pd.DataFrame(df['Slope']))
scaler_HDTH = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
scaler_VDTH = scaler.fit(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
scaler_HDTR = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
scaler_H9 = scaler.fit(pd.DataFrame(df['Hillshade_9am']))
scaler_HN = scaler.fit(pd.DataFrame(df['Hillshade_Noon']))
scaler_H3 = scaler.fit(pd.DataFrame(df['Hillshade_3pm']))
df['Elevation']=scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Aspect']=scaler_Aspect.fit_transform(pd.DataFrame(df['Aspect']))
df['Slope']=scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology']=scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology']=scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Horizontal_Distance_To_Roadways']=scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am']=scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon']=scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm']=scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))

yTrain = df['Horizontal_Distance_To_Fire_Points'].values
df.drop('Horizontal_Distance_To_Fire_Points',1,inplace=True)
xTrain = df.values

#Train
reg = RandomForestRegressor(n_estimators=500, max_features=xTrain.shape[1] - 3).fit(xTrain, yTrain)

#Read the test data.
filename_read = os.path.join(path,"test.csv")
data_test = pd.read_csv(filename_read)

#Preprocessing
IDs = data_test['ID']
data_test.drop('ID',1,inplace=True)

dummies_soilType = pd.get_dummies(data_test['Soil_Type'],prefix='soilType')

soil_type_test = list(dummies_soilType)
for name in soil_type_test:
	if name not in soil_type:
		dummies_soilType.drop(name, 1, inplace=True)

df = (pd.concat([data_test,dummies_soilType],axis=1)).astype(float)
df.drop(['Soil_Type'],axis=1,inplace=True)

df['Elevation']=scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Aspect']=scaler_Aspect.fit_transform(pd.DataFrame(df['Aspect']))
df['Slope']=scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology']=scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology']=scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Horizontal_Distance_To_Roadways']=scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am']=scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon']=scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm']=scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))

x_test = df.values

#Predict
result = reg.predict(x_test)

#Create CSV
submit_df=pd.DataFrame()
submit_df['ID']=IDs
submit_df['Horizontal_Distance_To_Fire_Points']=result
submit_df.to_csv('rf_zscore.csv',index=False)
