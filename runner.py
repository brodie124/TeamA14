# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:32:15 2019

@author: bp18125
"""

import FeatureSelector
import NeuralNetwork

        
selector = FeatureSelector('kaggle_data/train.csv')

selector.categorise_attribute([
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
])
    
selector.ignore_attribute([
    'Id', 'PoolQC', 'MiscFeature', 'Fence', 'GarageYrBlt', 'SalePrice'
])


print("Using the following features: ")
features = selector.rank_features(40)
for f in features:
    print(selector.data.columns[f])
    

network = NeuralNetwork(len(features))


for index, row in selector.data.iterrows():
    print(row)
    break