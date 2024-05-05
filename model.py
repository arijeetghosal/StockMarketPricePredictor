import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl


# we are using Quandl mainly to find historical data of stocks for example TATAGLOBAL stock

data = quandl.get("NSE/TATAGLOBAL")
print(data.head(10))

plt.figure(figsize=(16,8))


print(plt.plot(data['Close'], label='Closing Price'))

# classification problem: Buy(+1) or sell(-1) the stock

data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna() #for droppping the null values whichever I have 

# Input features to predict whter the customer should buy or sell the stock 

X =data[['Open - Close' , 'High - Low']]
print(X.head())

# Intention is to store +1 for the buy and -1 for the sell signal. The target variable is Y for classification task 

Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)
print(Y)

#Importing sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.25, random_state=44) 

#Implementation of KNN classifer
from sklearn.neighbors import KNeighborsClassifier

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# using gridsearch to find the best parameters

params ={'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

#Fit the model 
model.fit(X_train, y_train)

# Accuracy Score 

accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print('Train_data Accuracy: %.2f' %accuracy_train)
print('Test_data Accuracy: %.2f' %accuracy_test)