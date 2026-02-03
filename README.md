# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: HARI RAMESH
RegisterNumber:  212225100016
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("Placement_Data_Full_Class.csv")

X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]   # Placed / Not Placed

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[67, 91, 58, 0, 1, 1, 1, 0, 0, 88, 1, 67]]
new_student = scaler.transform(new_student)

pred = model.predict(new_student)
print("\nPredicted Status:", pred[0])
*/
```

## Output:
<img width="516" height="556" alt="Screenshot 2026-01-27 091634" src="https://github.com/user-attachments/assets/6230f2ae-7ba3-47c0-93fc-390fe3f051cc" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
