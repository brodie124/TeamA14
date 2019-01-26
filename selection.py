import pandas
import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2

attributes = [
    'Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley',
    'LotShape','LandContour','Utilities','LotConfig','LandSlope',
    'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
    'OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle',
    'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea',
    'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
    'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',
    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC',
    'CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF',
    'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
    'BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional',
    'Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish',
    'GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive',
    'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'
    ,'PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold'
    ,'SaleType','SaleCondition'
]

dummy_attributes = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig','LandSlope','Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 'Electrical','KitchenQual',
    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
    'PavedDrive','PoolQC','Fence', 'MiscFeature','SaleType','SaleCondition'
]

ignored_attributes = [
    'Id'
]





num_of_features = 40

print("Training Data...")

data_frame = pandas.read_csv('kaggle_data/test.csv')
data_frame = pandas.get_dummies(data_frame, columns=dummy_attributes)
data_frame = data_frame.dropna()

data_frame.drop(columns=ignored_attributes)

for v in ignored_attributes:
    attributes.remove(v)


# print(data_frame.na)
data_values = data_frame.values

axisX = data_values[:,0:len(attributes)]
axisY = data_values[:,len(attributes)]


model = ExtraTreesClassifier()
model.fit(axisX, axisY)


print("Data Trained")


important_features = model.feature_importances_
features_ranked = []
for i in range(0, len(important_features)):
    features_ranked.append((i, important_features[i]))

features_ranked.sort(key=lambda x: x[1], reverse=True)


print("Top " + str(num_of_features) + " Features: ")
for i in range(0, num_of_features):
    print("(" + str(features_ranked[i][1]) + ") " + attributes[features_ranked[i][0]])
