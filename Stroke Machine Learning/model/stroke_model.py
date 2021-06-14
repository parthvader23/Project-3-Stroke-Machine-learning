import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB



stroke_data = pd.read_csv('stroke_data.csv')
stroke_data.describe()

sns.countplot(stroke_data["age"], label="Count")
plt.show()


scatter_matrix(stroke_data.drop('hypertension', axis=1), figsize=(10,5))
plt.show()

feature_names = ["age", 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
X = stroke_data[feature_names]
y = stroke_data['stroke']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
print('Accuracy of GNB on training', gnb.score(X_train_scaled, y_train))
print('Accuracy of GNB on testing', gnb.score(X_test_scaled, y_test))

pickle.dump(gnb, open('stroke_model.pkl','wb'))


new_data = [[80,1,1,200,45]]
df = pd.DataFrame(new_data,columns= ["age", 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'])
original_df = X.copy()
reference = original_df.append(df).reset_index()

new_data_frame = reference[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(new_data_frame)


model = pickle.load( open('stroke_model.pkl','rb'))
model.predict([list(scaled_df[len(scaled_df)-1])])