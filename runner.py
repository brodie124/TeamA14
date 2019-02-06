# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:32:15 2019

@author: bp18125
"""

import numpy

from FeatureSelector import FeatureSelector
from NeuralNetwork import NeuralNetwork

        
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
    'Id', 'PoolQC', 'MiscFeature', 'Fence', 'GarageYrBlt'
])


print("Using the following features: ")
features = selector.rank_features(40 + 1)
#for f in features:
 #   print(selector.data.columns[f])
    

network = NeuralNetwork(len(features))

inputs  = []
outputs = []

sale_index = selector.data.columns.get_loc('SalePrice')


for index, row in selector.data.iterrows():

    row_input = []
    
    for f in features:
        print(f)
        row_input.append(row[attr])
        
    inputs.append(row_input)
        
    outputs.append(row['SalePrice'])
    
    
outputs = numpy.array([outputs]).T
inputs = numpy.array(inputs)

network.train(inputs, outputs, 10000)

print(network.weights)
