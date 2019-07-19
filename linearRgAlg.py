import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
#first read the csv file and load the data into a variable
data = pd.read_csv("student-mat.csv", sep=";")
#only select the features that we need for the prediction
#here we use only the features with regular data sets i.e in numerical format
data = data[['G1','G2','G3','studytime','failures','absences']]
predict = 'G3'
#then we create two numpy array or list of input and output to the model
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
best = 0
for _ in range(20):
    #split the data for training and testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
    #define a model
    linear = linear_model.LinearRegression()
    #train the model
    linear.fit(x_train, y_train)
    #calculate the accuracy of the model and save the best one
    acc = linear.score(x_test,y_test)
    print("Accuracy: ",acc)
    if acc > best:
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

'''for saving and storing the model so that we don't need to train it again and again 
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
#to see the coefficient and intercept of the linear model(y=mx+c)
print('Coefficient:\n',linear.coef_)
print('Intercept:\n', linear.intercept_)
'''

pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


#to view the actual tested data set and the predicted value
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

plot = "G1"
plt.scatter(data[plot],data['G3'])
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
