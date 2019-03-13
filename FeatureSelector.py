"""
Created on Wed Feb  6 13:37:18 2019

@author: bp18125
"""

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class FeatureSelector:

    def __init__(self, data_frame, target):
        self.data = data_frame
        self.target = target
        
    def rank_features(self, top):
        target_index = self.data.columns.get_loc(self.target)

        data_values = self.data.values
        
        axis_x = data_values[:, 0:len(self.data.columns)-1]
        axis_y = data_values[:, target_index]

        lab_enc = LabelEncoder()
        axis_y_encoded = lab_enc.fit_transform(axis_y)

        model = ExtraTreesClassifier()
        model.fit(axis_x, axis_y_encoded)

        important_features = model.feature_importances_
        features_ranked = []
        for i in range(0, len(important_features)):
            features_ranked.append((i, important_features[i]))

        features_ranked.sort(key=lambda x: x[1], reverse=True)
        
        return [x[0] for x in features_ranked][0:top]