#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from joblib import dump, load

raw_file = "raw.h5"
imputed_cleaned_file = "stroke_X_imputed_before_RFE.h5"
RFECV_selector_name = "RFECV_selector_stroke.joblib"
outcome = "CVDSTRK3"    # (Ever told) (you had) a stroke.

random_state = 1


if __name__ ==  "__main__":
	data = pd.read_hdf(raw_file)  # to read cleaned data
	data.shape
	data = data.dropna(subset=[outcome], axis=0)
	data.shape
	data = data[data.DISPCODE != 1200]  # == 1200    final disposition (1100 completed or not 1200)
	data.shape
	
	y = abs(data.CVDSTRK3 - 2)
	X = data.copy()
	X = X.drop([outcome], axis=1)
	
	headers = X.columns.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	empty_train_columns = []
	for col in X_train.columns.values:
		# all the values for this feature are null
		if sum(X_train[col].isnull()) == X_train.shape[0]:
			empty_train_columns.append(col)
	print(empty_train_columns)
	
	X = X.drop(empty_train_columns, axis=1)     # ['TOLDCFS', 'HAVECFS', 'WORKCFS']
	X.shape
	
	imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
	X.isnull().values.any()
	X.to_hdf(imputed_cleaned_file, "X", complevel=2)
	
	selector = RFECV(RandomForestClassifier(), min_features_to_select=30, cv=3, verbose=2, n_jobs=-1)
	selector.fit(X, y)
	
	dump(selector, RFECV_selector_name, compress=3)
	# selector_2 = load(RFECV_selector_name)
	
	features_to_keep = selector.get_feature_names_out(X.columns.values)
	print(features_to_keep)
	
	with open('features.txt', 'w+') as f:
		f.write(str(features_to_keep))
	
	
	pass
