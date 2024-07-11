import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import pickle

# Importing dataset
dataset = pd.read_csv('diabetes.csv')

# Preview data
print(dataset.head())

# Dataset dimensions - (rows, columns)
print(dataset.shape)

# Features data-type
print(dataset.info())

# Statistical summary
print(dataset.describe().T)

# Count of null values
print(dataset.isnull().sum())

# Data visualization
# Outcome countplot
sns.countplot(x='Outcome', data=dataset)
plt.show()

# Histogram of each feature
col = dataset.columns[:8]
plt.subplots(figsize=(20, 15))
length = len(col)
for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length // 2), 3, j + 1)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    dataset[i].hist(bins=20)
    plt.title(i)
plt.show()

# Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset, figsize=(20, 20))
plt.show()

# Pairplot 
sns.pairplot(data=dataset, hue='Outcome')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# Data preprocessing
dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
print(dataset_new.isnull().sum())
dataset_new.fillna({
    "Glucose": dataset_new["Glucose"].mean(),
    "BloodPressure": dataset_new["BloodPressure"].mean(),
    "SkinThickness": dataset_new["SkinThickness"].mean(),
    "Insulin": dataset_new["Insulin"].mean(),
    "BMI": dataset_new["BMI"].mean()
}, inplace=True)
print(dataset_new.describe().T)

# Feature scaling using MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_new)
dataset_scaled = pd.DataFrame(dataset_scaled, columns=dataset_new.columns)

# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled[['Glucose', 'Insulin', 'BMI', 'Age']]
Y = dataset_scaled['Outcome']

# Splitting the dataset X and Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset_new['Outcome'])
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# Data modelling
# Logistic Regression Algorithm
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, Y_train)

# K nearest neighbors Algorithm
X_axis = list(range(1, 31))
acc = []
for i in X_axis:
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, Y_train)
    prediction = knn_model.predict(X_test)
    acc.append(accuracy_score(Y_test, prediction))

plt.plot(X_axis, acc)
plt.xticks(X_axis)
plt.title("Finding best value for n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

print('Highest value:', max(acc))
knn = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2)
knn.fit(X_train, Y_train)

# Support Vector Classifier Algorithm
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, Y_train)

# Naive Bayes Algorithm
nb = GaussianNB()
nb.fit(X_train, Y_train)

# Decision tree Algorithm
dectree = DecisionTreeClassifier(criterion='entropy', random_state=42)
dectree.fit(X_train, Y_train)

# Random forest Algorithm
ranfor = RandomForestClassifier(n_estimators=11, criterion='entropy', random_state=42)
ranfor.fit(X_train, Y_train)

# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)

# Model evaluation
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)
print("Logistic Regression: " + str(accuracy_logreg * 100))
print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred_knn)
sns.heatmap(pd.DataFrame(cm), annot=True)
plt.show()

# Classification report
print(classification_report(Y_test, Y_pred_knn))

# Function to predict new data
def predict(model, data):
    if isinstance(data, pd.DataFrame):
        return model.predict(data)
    else:
        data_df = pd.DataFrame(data, columns=['Glucose', 'Insulin', 'BMI', 'Age'])
        return model.predict(data_df)

# Example new data points
x_new = [148, 0, 33.6, 50]
x_new1 = [80, 0, 30, 10]

# Saving used data
new_data_df = pd.DataFrame([x_new, x_new1], columns=['Glucose', 'Insulin', 'BMI', 'Age'])
with open("used_data.pkl", "wb") as f:
    pickle.dump(new_data_df, f)

# Predict using the best model (KNN model based on your analysis)
for new_data_point in [x_new, x_new1]:
    predictions = predict(knn, [new_data_point])
    print("Predicted target values:", predictions)
