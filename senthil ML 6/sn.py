import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Admission_Predict.csv")

print(data.head())
print(data.info())
print(data.describe())

data = data.drop(["Serial No."], axis=1)

data["Admitted"] = data["Chance of Admit "] >= 0.75
data["Admitted"] = data["Admitted"].astype(int)

data = data.drop("Chance of Admit ", axis=1)

print(data.isnull().sum())

data.hist(figsize=(10,8))
plt.show()

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

sns.boxplot(data=data)
plt.show()

X = data.drop("Admitted", axis=1)
y = data["Admitted"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_student = [[320, 110, 4, 4.5, 4.5, 9.2, 1]]
new_student = scaler.transform(new_student)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Student will get Admission")
else:
    print("Student will NOT get Admission")