#For 500 data samples
# Random forest using SMOT
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
#Augmentation
# Load your existing dataset
existing_data = pd.read_csv('mlp.csv')

# Exclude the first two columns
existing_data = existing_data.iloc[:, :]

# Repeat the existing dataset to reach the desired size
desired_size = 500
repeat_factor = desired_size // len(existing_data) + 1
augmented_data = pd.concat([existing_data] * repeat_factor, ignore_index=True)

# Trim the augmented dataset to the desired size
augmented_data = augmented_data.head(desired_size)

# Save the augmented dataset
augmented_data.to_csv('mlp500.csv', index=False)

data = pd.read_csv('mlp500.csv')
print(data)
"""


# Importing the dataset
dataset = pd.read_csv('mlp500.csv')
X = dataset.drop(columns=['Amount','sales'])
y = dataset.iloc[:, -1].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, 
strategy='mean')
imputer.fit(X)
X = imputer.transform(X)


# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training thes random forest model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, 
criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#Accuracy:0.7777777777777778


# Predicting new results
#High-0
#Low-1
#Medium-2
A1=classifier.predict(sc.transform([[25, 6, 1000,0, 0,1200,0,0,700,0,0,5,0,0,1050,0,0,3000,0,0,1900,0,0]]))
print("The predicted result for JUNE :" ,A1)
print("")
#high

A2=classifier.predict(sc.transform([[25, 10, 1000,0, 0,120,0,0,70,0,0,5,0,0,250,0,0,300,0,0,1900,0,0]]))
print("The predicted result for OCT :" ,A2)
print("")
#low

A3=classifier.predict(sc.transform([[25, 1, 1000,0, 0,120,0,0,70,0,0,5,0,0,250,0,0,300,0,0,1900,0,0]]))
print("The predicted result for JAN :" ,A3)
print("")
#medium


A4=classifier.predict(sc.transform([[25, 6, 1000,0, 0,120,0,0,70,0,0,5,0,0,250,0,0,300,0,0,1900,0,0]]))
print("The predicted result for JUNE :" ,A4)
print("")
#low


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX :")
print(cm)
print("")
print("ACCURACY SCORE: ",accuracy_score(y_test, y_pred))

#Sensitivity and Specificity
from sklearn.metrics import recall_score,precision_score,mean_squared_error,r2_score
#Calculating Sensitivity
sensitivity=recall_score(y_test, y_pred,average="macro")
#Calculating Specificity
specificity=precision_score(y_test, y_pred,average="macro", zero_division=0)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)

#Classification report
from sklearn.metrics import classification_report
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

#Error!!!
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)
print("")