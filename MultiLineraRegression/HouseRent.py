import pandas as pd
import sklearn
from sklearn import preprocessing, linear_model
import numpy as np

#### LOAD DATA ####
print('-' * 30);print("IMPORTING DATA ");print('-' * 30)
# read in data file
data = pd.read_csv('houses_to_rent.csv', sep = ',')

# define columns to use
data = data[['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance',
              'furniture', 'rent amount']]

print(data.head())

#### PROCESS DATA ####
# R$7,000
# remove first 2 characters from string and remove coma
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',' , '')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',' , '')))
#print(data.head())

# encode data to numerical values
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform(data['furniture'])
# furnished 0
# not furnished 1
print(data.head())


# check for NaNs and missing data
print('-' * 30);print("CHECKING FOR NULL DATA ");print('-' * 30)
print(data.isnull().sum())

# if null values are found there are many ways to deal with this:
    # drop
    # data = data.dropna()
    # this is only vaiable if we have enough data


print('-' * 30);print(" HEAD ");print('-' * 30)
print(data.head())

#### SPLIT DATA ####
print('-' * 30);print(" SPLIT DATA ");print('-' * 30)

# drop the output (dependent) for x
# axis 1: tells the program to drop columns instead of rows
x = np.array(data.drop(['rent amount'], 1))

# output
y = np.array(data['rent amount'])

print('X', x.shape)
print('Y', y.shape)

# split
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print('XTrain', xTrain.shape)
print('XTest', xTest.shape)

#### TRAINING ####
print('-' * 30);print(" TRAINING ");print('-' * 30)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

accuracy = model.score(xTest, yTest)

# if we have multiple linear regression we will have multiple gradients
# which will have multiple coefficients
# c: intercept
# Y = c + m1X1 + m2X2 + m3X3 ...
print('Coefficients: ', model.coef_) # because we had 6 attributes we have 6 gradients
print('Intercept: ', model.intercept_)

# accuracy percentage rounded 3 decimal places
print('Accuracy', round(accuracy * 100, 3))


#### EVALUATION ####
print('-' * 30);print(" MANUAL TESTING  ");print('-' * 30)
testVals = model.predict(xTest)

# check ground truth and predicted values against each other
# to determine by how much the prediction is wrong
print(testVals.shape)

error = []
for i, testVal in enumerate(testVals):
    error.append(yTest[i] - testVal)
    print(f'Actual: {yTest[i]} Predicted: {int(testVal)} Error: {int(error[i])}')
