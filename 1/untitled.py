import os
import pandas as pd
from sklearn import preprocessing as preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

path = "./distance-to-fire-points/"
filename_read = os.path.join(path,"train.csv")
data_train = pd.read_csv(filename_read)

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

y = df['Horizontal_Distance_To_Fire_Points'].values
df.drop('Horizontal_Distance_To_Fire_Points',1,inplace=True)
x = df.values

accuracy_linear = np.zeros(10)
accuracy_rf = np.zeros(10)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #linear regression
    linear = LinearRegression().fit(X_train, y_train)
    yPr_linear = linear.predict(X_test)
    accuracy_linear[i] = np.sqrt(((y_test-yPr_linear)**2).mean())
    #random forest
    rf = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1] - 3).fit(X_train, y_train)
    yPr_rf = rf.predict(X_test)
    accuracy_rf[i] = np.sqrt(((y_test-yPr_rf)**2).mean())

value, pvalue = ttest_ind(accuracy_linear, accuracy_rf, equal_var=True)
if value >1.812:
    print("At the critical level of 0.05(one-tailed) We can reject the zero hypothesis that both models have same performance and get the conclusion that the random forest performs better than the linear model.")
else:
    print("At the critical level of 0.05(one-tailed) We cannot reject the zero hypothesis that both models have same performance.")

Awin = 0
Bwin = 0
for i in range(10):
    print(accuracy_linear[i], " ", accuracy_rf[i])
    if accuracy_linear[i] < accuracy_rf[i]:
        Awin += 1
    elif accuracy_linear[i] > accuracy_rf[i]:
        Bwin += 1

print("Awin", ": ", Awin)
print("Bwin", ": ", Bwin)

if Bwin >= 8:
    print("At the critical level of 0.05(one-tailed) We can reject the zero hypothesis that P(A_win > B_win) = 0.5 and get the conclusion that the random forest performs better than the linear model.")
else:
    print("At the critical level of 0.05(one-tailed) We cannot reject the zero hypothesis that P(A_win > B_win) = 0.5.")

