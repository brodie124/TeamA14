import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random

df = pd.read_csv("test.csv")
print(df.shape)
print(df.columns)
#prints first 5 results
print(df.head(5))

columnsofinterest = df.columns
for column in columnsofinterest:
  print(column)
print(df.LotArea)
#selecting more than one column
columnsofinterest = ["Id","LotArea"]
datacolumns = df[columnsofinterest]
#This prints the no of entries, the mean, max,minimum upper quartile and lower quartile
print(datacolumns.describe())






#using group by
print(df.groupby("Id")["LotArea"].transform("sum")) #this is used we want to get a new value for each input row.
# finds the mean
print(df.groupby("Id")["LotArea"].mean())
#not 100% sure on what the difference is between .aggregegate and .mean is
print(df.groupby("Id")['LotArea'].aggregate("mean"))
#not sure why this starts on 0 but i think it gave new values of something
print(df.groupby("Id")["LotArea"].transform(lambda x : x.mean()))

#names of the columns that have string values value
cols_to_transform = [ 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig','LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 'Electrical','KitchenQual', 'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence', 'MiscFeature','SaleType','SaleCondition']
#change from string to a number by makeing lots of tempoary columns
df_with_dummies = pd.get_dummies( df, columns = cols_to_transform )
print(df_with_dummies)

#drop na and fill na can be used to fill in missing values
print(df.Alley)
print(df.Alley.fillna(0)) #fills na values with 0
print(df.LotFrontage)
print(df.LotFrontage.fillna(0))
print(df.LotFrontage.dropna())# drops the value all together. I don't think we are going to use this

#corrilation

corr=df.corr()["SalePrice"]
print(corr[np.argsort(corr, axis=0)[::-1]])