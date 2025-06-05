from django.db import models


import numpy as np
import pickle



import joblib
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

svm = pickle.load(open(r'C:\Users\dondh\Music\CODING\frontend\svc_park.pkl', 'rb'))
xgb = pickle.load(open(r'C:\Users\dondh\Music\CODING\frontend\xgb_park.pkl', 'rb'))



data = pd.read_csv(r'C:\Users\dondh\Music\CODING\frontend\Parkinsons_test.csv')

# x = data.drop(['Unnamed: 0'],axis=1)

def predict(algo,row):
	#print(x.columns)
	test_data=data.iloc[row].values.reshape(1,-1)
	print(test_data.shape)
	#print(test_data.columns)
	if algo == 'svm':
		y_pred = svm.predict(test_data)
	elif algo == 'xgb':
		y_pred = xgb.predict(test_data)
	return y_pred

	

