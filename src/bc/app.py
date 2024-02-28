import pandas as pd                                  
import matplotlib.pyplot as plt       
import plotly.express as px         
import seaborn as sns      
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import filterwarnings
from sklearn.metrics import classification_report, confusion_matrix
filterwarnings('ignore')

cancer_df = pd.read_csv("datasets/bc/data.csv")
cancer_df = cancer_df.dropna(axis=1)

print("cancer_df : ", type(cancer_df))

print("HEAD\n",cancer_df.head())
print("TAIL\n", cancer_df.tail())

cancer_df.isnull().sum()
cancer_df.info()

cancer_df.drop(columns='id', axis=1, inplace=True)
cancer_df['diagnosis'] = cancer_df['diagnosis'].map({'M':1, 'B': 0})

X = cancer_df.drop(columns='diagnosis', axis=1)
y = cancer_df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print("X_train shape:",X_train.shape) or print('-'*22)
print("X_test shape:",X_test.shape) or print('-'*22)
print("Y_train shape:",y_train.shape) or print('-'*22)
print("Y_test shape:",y_test.shape)

print(type(X_train), X_train)

svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(cm, annot=True)
print(classification_report(y_true=y_test, y_pred=y_pred))

scaler = MinMaxScaler() 
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)
print("y_pred", y_pred)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(cm, annot=True, fmt='d')

print(classification_report(y_true=y_test, y_pred=y_pred))