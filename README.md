# Glaucoma Data Analysis

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('/content/Glaucoma.zip')

# Encode the target column
target_column = 'Class'
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Display basic information about the dataset
print('Data Shape:', data.shape)
print('Data Info:')
data.info()
print('Data Head:')
print(data.head())
print('Data Tail:')
print(data.tail())
print('Data Description:')
print(data.describe())

# Display value counts for the "ag" column
print('Value Counts for "ag":')
print(data["ag"].value_counts())

# Display min, max, and mean values for "ag"
print('Min, Max, Mean values for "ag":')
print('The highest ag was of:', data['ag'].max())
print('The lowest ag was of:', data['ag'].min())
print('The average ag in the data:', data['ag'].mean())

# Line plot for "ag"
plt.plot(data['ag'])
plt.xlabel("ag")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

# Histograms for Glaucoma levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

data_len = data[data['Class'] == 0]['an'].value_counts()
ax1.hist(data_len, color='red')
ax1.set_title('Having Glaucoma')

data_len = data[data['Class'] == 1]['an'].value_counts()
ax2.hist(data_len, color='green')
ax2.set_title('NOT Having Glaucoma')

fig.suptitle('Glaucoma Levels')
plt.show()

# Check for duplicates and handle them
data.duplicated()
newdata = data.drop_duplicates()

# Check for null values
print('Total null values:')
print(data.isnull().sum())

# Display a subset of data
print('Subset of Data:')
print(data[1:5])

# Normalize data
from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:, 1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["ag", "at", "as", "an"])
scaled_df.head()

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Class'])
train_X = train[train.columns[:-1]]
train_Y = train[train.columns[-1:]]
test_X = test[test.columns[:-1]]
test_Y = test[test.columns[-1:]]
X = data[data.columns[:-1]]
Y = data['Class']

# Logistic Regression
model = LogisticRegression()
model.fit(train_X, train_Y)
prediction3 = model.predict(test_X)
accuracy_logistic = metrics.accuracy_score(prediction3, test_Y)
report = classification_report(test_Y, prediction3)

print('The accuracy of Logistic Regression is', accuracy_logistic)
print("Classification Report:\n", report)

# Linear Regression
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Evaluate the Linear Regression model
mse = mean_squared_error(test_Y, prediction)
rmse = mean_squared_error(test_Y, prediction, squared=False)
mae = mean_absolute_error(test_Y, prediction)
r_squared = r2_score(test_Y, prediction)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae)
print('R-squared:', r_squared)
