import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
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


print("Loading training data...")
train_data_frame = pandas.read_csv('kaggle_data/train.csv')
train_data_frame = replace_dummies(train_data_frame)


print("Selecting top features for use...")
selector = FeatureSelector(train_data_frame, 'SalePrice')
top_features = selector.rank_features(70)

top_named_features = [train_data_frame.columns[x] for x in top_features]
top_named_features.append('SalePrice')


print("Top features: ")
print(top_named_features)


print("Reloading train data with only top features")
train_data_frame = pandas.read_csv('kaggle_data/train.csv', usecols=top_named_features)
train_data_frame = replace_dummies(train_data_frame)
train_data_frame = train_data_frame.drop(axis=1, columns=[x for x in train_data_frame.columns if x not in top_named_features])


target_index = train_data_frame.columns.get_loc('SalePrice')
feature_count = len(train_data_frame.columns)

print("Splitting train data...")
train_x = train_data_frame.values[:, 0:len(train_data_frame.columns) - 1]
train_y = train_data_frame.values[:, target_index]


print("Training scaler and transforming data...")
standard_scaler.fit(train_x)
train_x = standard_scaler.transform(train_x)


print("Training linear regression model...")
lr = LinearRegression()
lr.fit(train_x, train_y)

top_named_features.remove('SalePrice')
top_named_features.append('Id')
test_data_frame = pandas.read_csv('kaggle_data/test.csv', usecols=top_named_features)
top_named_features.remove('Id')


print("Gathering test data...")
id_loc = test_data_frame.columns.get_loc('Id')
test_ids = [x.Id for i, x in test_data_frame.iterrows()]

test_data_frame = replace_dummies(test_data_frame)

print("Splitting test data and performing predictions...")
test_x = test_data_frame.values[:, 0:len(test_data_frame.columns)]
test_x = standard_scaler.transform(test_x)

train_predictions = lr.predict(train_x)
test_predictions = lr.predict(test_x)

print("Writing predictions to file...")
submission = open('submission.csv', 'w')

submission.write("Id,SalePrice\n")

i = 0
for row in train_data_frame.iterrows():
    if i >= len(test_ids):
        break
    submission.write(str(int(test_ids[i])) + ",")
    submission.write(str(test_predictions[i]) + "\n")

    i += 1

submission.close()

print("Done!")