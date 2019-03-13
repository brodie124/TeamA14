import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from FeatureSelector import FeatureSelector

categorised_attributes = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

ignored_attributes = [
    'Id', 'PoolQC', 'MiscFeature', 'Fence', 'GarageYrBlt'
]

standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)


def replace_dummies(df):
    """
    Normalises the DataFrame for use
    :param df
    :return data_frame
    """
    for categorised in categorised_attributes:
        if categorised in df:
            df[categorised] = df[categorised].astype('category').cat.codes

    df.fillna(0, inplace=True)
    df = df.drop(columns=[x for x in ignored_attributes if x in df])

    return df




train_data_frame = pandas.read_csv('kaggle_data/train.csv')
train_data_frame = replace_dummies(train_data_frame)

selector = FeatureSelector(train_data_frame, 'SalePrice')
top_features = selector.rank_features(40)

top_named_features = [train_data_frame.columns[x] for x in top_features]
top_named_features.append('SalePrice')

print(top_named_features)

train_data_frame = pandas.read_csv('kaggle_data/train.csv', usecols=top_named_features)
train_data_frame = replace_dummies(train_data_frame)
train_data_frame = train_data_frame.drop(axis=1, columns=[x for x in train_data_frame.columns if x not in top_named_features])


target_index = train_data_frame.columns.get_loc('SalePrice')
feature_count = len(train_data_frame.columns)

print("FEATURE COUNT: ", feature_count)
print("COL COUNT:", len(train_data_frame.columns))



train_x = train_data_frame.values[:, 0:len(train_data_frame.columns) - 1]
train_y = train_data_frame.values[:, target_index]


standard_scaler.fit(train_x)

train_x = standard_scaler.transform(train_x)

mlp = MLPClassifier(hidden_layer_sizes=(feature_count, feature_count, feature_count),
                    learning_rate='constant', max_iter=5000, )
mlp.fit(train_x, train_y)

top_named_features.remove('SalePrice')
top_named_features.append('Id')
test_data_frame = pandas.read_csv('kaggle_data/test.csv', usecols=top_named_features)
top_named_features.remove('Id')

print("Test Rows:", test_data_frame.shape[0])

id_loc = test_data_frame.columns.get_loc('Id')

test_ids = [x.Id for i, x in test_data_frame.iterrows()]


test_data_frame = replace_dummies(test_data_frame)

print("Test Rows:", test_data_frame.shape[0])

print("TEST COL COUNT:", len(test_data_frame.columns))
print(test_data_frame.columns)

#test_data_frame = test_data_frame.drop(axis=1, columns=[x for x in test_data_frame.columns if x not in top_named_features])

#target_index = test_data_frame.columns.get_loc('SalePrice')
#feature_count = len(test_data_frame.columns)

test_x = test_data_frame.values[:, 0:len(test_data_frame.columns)]
#test_y = test_data_frame.values[:, target_index]

test_x = standard_scaler.transform(test_x)

print("Test Rows:", test_data_frame.shape[0])

predictions = mlp.predict(test_x)

print("Test Rows:", test_data_frame.shape[0])

submission = open('submission.csv', 'w')

submission.write("Id,SalePrice\n")

i = 0
for row in train_data_frame.iterrows():
    if i >= len(test_ids):
        print("I (" + str(i) + ") > " + str(len(test_ids)))
        break
    submission.write(str(test_ids[i]) + ",")
    submission.write(str(predictions[i]) + "\n")
    
    
    
    #print("ID: ", row['Id'], " Price:", predictions[i])
    i += 1
    
print(str(i))

submission.close()

