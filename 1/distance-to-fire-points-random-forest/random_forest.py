import os
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing as preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Read training data
filename_read = os.path.join("./", "train.csv")
data_train = pd.read_csv(filename_read)

#Preprocessing
data_train.drop('ID',1,inplace=True)
data_train['Distance_To_Hydrology1']=np.power(data_train['Horizontal_Distance_To_Hydrology'],2)
data_train['Distance_To_Hydrology2']=np.power(data_train['Vertical_Distance_To_Hydrology'],2)


data_train['climate_zone']=data_train['Soil_Type']//1000
data_train['geological_zone']=(data_train['Soil_Type']-1000*data_train['climate_zone'])//100

dummies_climate=pd.get_dummies(data_train['climate_zone'],prefix='climate')
df=pd.concat([data_train,dummies_climate],axis=1)
dummies_geological=pd.get_dummies(data_train['geological_zone'],prefix='geological')
df=pd.concat([df,dummies_geological],axis=1)
df.drop(['Soil_Type'],axis=1,inplace=True)
df.drop(['climate_zone'],axis=1,inplace=True)
df.drop(['geological_zone'],axis=1,inplace=True)

scaler = preprocessing.StandardScaler()
scaler_Elevation = scaler.fit(pd.DataFrame(df['Elevation']))
scaler_Aspect = scaler.fit(pd.DataFrame(df['Aspect']))
scaler_Slope = scaler.fit(pd.DataFrame(df['Slope']))
scaler_HDTH = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
scaler_VDTH = scaler.fit(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
scaler_DTH1 = scaler.fit(pd.DataFrame(df['Distance_To_Hydrology1']))
scaler_DTH2 = scaler.fit(pd.DataFrame(df['Distance_To_Hydrology2']))
scaler_HDTR = scaler.fit(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
scaler_H9 = scaler.fit(pd.DataFrame(df['Hillshade_9am']))
scaler_HN = scaler.fit(pd.DataFrame(df['Hillshade_Noon']))
scaler_H3 = scaler.fit(pd.DataFrame(df['Hillshade_3pm']))
df['Elevation']=scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Aspect']=scaler_Aspect.fit_transform(pd.DataFrame(df['Aspect']))
df['Slope']=scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology']=scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology']=scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Distance_To_Hydrology1']=scaler_DTH1.fit_transform(pd.DataFrame(df['Distance_To_Hydrology1']))
df['Distance_To_Hydrology2']=scaler_DTH2.fit_transform(pd.DataFrame(df['Distance_To_Hydrology2']))
df['Horizontal_Distance_To_Roadways']=scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am']=scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon']=scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm']=scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))

yTrain = df['Horizontal_Distance_To_Fire_Points'].values
df.drop('Horizontal_Distance_To_Fire_Points',1,inplace=True)
xTrain = df.values

#Train
reg = RandomForestRegressor(n_estimators=500, max_features=xTrain.shape[1] - 3).fit(xTrain, yTrain)

#calculate training error
yTrain_predict = reg.predict(xTrain)
error = np.sqrt(np.sum(np.square(yTrain_predict - yTrain)) / xTrain.shape[0])
print("Training Error:", error)




#Read the test data.
filename_read = os.path.join("./", "test.csv")
data_test = pd.read_csv(filename_read)

#Preprocessing
IDs=data_test['ID']
data_test.drop('ID',1,inplace=True)
data_test['Distance_To_Hydrology1']=np.power(data_test['Horizontal_Distance_To_Hydrology'],2)
data_test['Distance_To_Hydrology2']=np.power(data_test['Vertical_Distance_To_Hydrology'],2)

data_test['climate_zone']=data_test['Soil_Type']//1000
data_test['geological_zone']=(data_test['Soil_Type']-1000*data_test['climate_zone'])//100

dummies_climate=pd.get_dummies(data_test['climate_zone'],prefix='climate')
df=pd.concat([data_test,dummies_climate],axis=1)
dummies_geological=pd.get_dummies(data_test['geological_zone'],prefix='geological')
df=pd.concat([df,dummies_geological],axis=1)
df.drop(['Soil_Type'],axis=1,inplace=True)
df.drop(['climate_zone'],axis=1,inplace=True)
df.drop(['geological_zone'],axis=1,inplace=True)

df['Elevation']=scaler_Elevation.fit_transform(pd.DataFrame(df['Elevation']))
df['Aspect']=scaler_Aspect.fit_transform(pd.DataFrame(df['Aspect']))
df['Slope']=scaler_Slope.fit_transform(pd.DataFrame(df['Slope']))
df['Horizontal_Distance_To_Hydrology']=scaler_HDTH.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Hydrology']))
df['Vertical_Distance_To_Hydrology']=scaler_VDTH.fit_transform(pd.DataFrame(df['Vertical_Distance_To_Hydrology']))
df['Distance_To_Hydrology1']=scaler_DTH1.fit_transform(pd.DataFrame(df['Distance_To_Hydrology1']))
df['Distance_To_Hydrology2']=scaler_DTH2.fit_transform(pd.DataFrame(df['Distance_To_Hydrology2']))
df['Horizontal_Distance_To_Roadways']=scaler_HDTR.fit_transform(pd.DataFrame(df['Horizontal_Distance_To_Roadways']))
df['Hillshade_9am']=scaler_H9.fit_transform(pd.DataFrame(df['Hillshade_9am']))
df['Hillshade_Noon']=scaler_HN.fit_transform(pd.DataFrame(df['Hillshade_Noon']))
df['Hillshade_3pm']=scaler_H3.fit_transform(pd.DataFrame(df['Hillshade_3pm']))

x_test=df.values

#Predict
result=reg.predict(x_test)


#Create CSV
submit_df=pd.DataFrame()
submit_df['ID']=IDs
submit_df['Horizontal_Distance_To_Fire_Points']=result
submit_df.to_csv('rf_zscore.csv',index=False)