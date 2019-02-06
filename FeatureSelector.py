"""
Created on Wed Feb  6 13:37:18 2019

@author: bp18125
"""

import pandas
import numpy

from sklearn.ensemble import ExtraTreesClassifier

class FeatureSelector:
    def __init__(self, filename):
        self.ignored_attributes = []
        self.categorised_attributes = []
        self.data = pandas.read_csv(filename)
    
    def get_attributes(self):
        return list(self.data.columns)
    
    
    def ignore_attribute(self, attribute):
        if(type(attribute) is list):
            self.ignored_attributes += attribute
        else:
            self.ignored_attributes.append(attribute)
            
    def categorise_attribute(self, attribute):
        if(type(attribute) is list):
            self.categorised_attributes += attribute
        else:
            self.categorised_attributes.append(attribute)
    
    def rank_features(self, top):  
        for categorised in self.categorised_attributes:
            self.data[categorised] = self.data[categorised].astype('category').cat.codes
        
        self.data = self.data.dropna()
        self.data = self.data.drop(columns=self.ignored_attributes)
    
        data_values = self.data.values
        
        axisX = data_values[:,0:len(self.get_attributes())]
        axisY = data_values[:,len(self.get_attributes())-1]
        
        model = ExtraTreesClassifier()
        model.fit(axisX, axisY)
        
        important_features = model.feature_importances_
        features_ranked = []
        for i in range(0, len(important_features)):
            features_ranked.append((i, important_features[i]))

        features_ranked.sort(key=lambda x: x[1], reverse=True)
        
        return [x[0] for x in features_ranked][0:top]