import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import sklearn


#### LOAD DATA ####
data = pd.read_csv('total_cases.csv', sep= ',')
# add id
data['id'] = data.index
# select columns
data = data[['id', 'World']]

print('-' * 30);print('HEAD');print('-' * 30)
print(data.head())

### PREPARE DATA ####
print('-' * 30);print('PREPARE DATA');print('-' * 30)

# convert to np arrays
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['World']).reshape(-1, 1)
plt.plot(y, '-m')
#plt.show()


polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)
#print(x)


#split
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#### TRAINING DATA ####
print('-' * 30);print('TRAINING ON DATA');print('-' * 30)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

accuracy = model.score(xTest, yTest)
print(f'Accuracy: {round((accuracy * 100), 3)} %')


y0 = model.predict(x)


#### PREDICTION ####
days = 30
print('-' * 30);print('PREDICTION');print('-' * 30)
print(f'Prediction - Cases after {days} days: ', end='')
# prediction in millions for 2 days after tha data ends
# rounded to 2 decimal places
print(round(int(model.predict(polyFeat.fit_transform(([[336 + days]])))) / 1000000, 2), 'Million')

x1 = np.array(list(range(1, 336 + days))).reshape(-1, 1)
y1 = model.predict(polyFeat.fit_transform(x1))

plt.plot(y1, '--r')
plt.plot(y0, '--b')
plt.show()


