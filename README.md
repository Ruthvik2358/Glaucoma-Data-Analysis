# Glaucoma-Data-Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

data=pd.read_csv('/content/Glaucoma.zip')

from sklearn.preprocessing import LabelEncoder
target_column='Class'
label_encoder=LabelEncoder()
data[target_column]=label_encoder.fit_transform(data[target_column])

data.shape

data.info()

data.head()

data.tail()

data.describe()

data["ag"].value_counts()

# See the min, max, mean values
print('The highest ag was of:',data['ag'].max())
print('The lowest ag was of:',data['ag'].min())
print('The average ag in the data:',data['ag'].mean())

import matplotlib.pyplot as plt

# Line plot
plt.plot(data['ag'])
plt.xlabel("ag")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Class']==0]['an'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('Having Glaucoma')

data_len=data[data['Class']==1]['an'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having Glaucoma')

fig.suptitle('Glaucoma Levels')
plt.show()

data.duplicated()

newdata=data.drop_duplicates()

data.isnull().sum() #checking for total null values

data[1:5]

from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["ag", "at", "as", "an"])
scaled_df.head()

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Class'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Class']
len(train_X), len(train_Y), len(test_X), len(test_Y)

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)

#Evaluate the model using various metrices
mse = mean_squared_error(test_Y, prediction)
rmse = mean_squared_error(test_Y, prediction, squared=False) #Calculate the square root of MSE
mae = mean_absolute_error(test_Y, prediction)
r_squared=r2_score(test_Y, prediction)

print('Mean squared Error:',mse)
print('Root Mean Squared Error:',rmse)
print('Mean Absolute Error:',mae)
print('R-squared:',r_squared)
