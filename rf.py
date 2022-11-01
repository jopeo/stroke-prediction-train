#!/usr/bin/env python

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss


raw_file = "raw.h5"
cleaned_file = "stroke_cleaned.h5"
model_name = "stroke_model_nm-3.joblib"
outcome = "CVDSTRK3"    # (Ever told) (you had) a stroke.

# todo: check hyperparams:
random_state = 1
n_estimators = [50, 100, 250, 500]    #, 300, 500, 750, 800, 1200]
# criterion = 'gini'
# max_depth = None
# max_features = 'sqrt'
# Bootstrap =
# Min_samples_split =
# Min_sample_leaf =
# Max_leaf_nodes =
# n_jobs =
# max_samples =
# Class_weight =

param_grid = {"bootstrap": [True],
              "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              "max_features": ["auto", "sqrt"],
              "min_samples_leaf": [1, 2, 4],
              "min_samples_split": [2, 5, 10],
              "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
              }

features_cat = ['_STATE',       # geographical state]
                'SEXVAR',       # Sex of Respondent 1 MALE, 2 FEMALE
                '_RFHLTH',      # Health Status  1 Good or Better Health 2 Fair or Poor Health
                                    # 9 Don’t know/ Not Sure Or Refused/ Missing
                '_PHYS14D',     # Healthy Days 1 Zero days when physical health not good
                                    #  2 1-13 days when physical health not good
                                    # 3 14+ days when physical health not good
                                    # 9 Don’t know/ Refused/Missing
                '_MENT14D',     # SAME AS PHYS
                '_HCVU651',     # Health Care Access  1 Have health care coverage 2 Do not have health care coverage 9 Don’t know/ Not Sure, Refused or Missing
                '_TOTINDA',     # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
                '_ASTHMS1',     # asthma? 1 current 2 former 3 never
                '_DRDXAR2',     # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
                '_EXTETH3',     # ever had teeth extracted? 1 no 2 yes 9 dont know
                '_DENVST3',     # dentist in past year? 1 yes 2 no 9 don't know
                '_RACE',        # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispanic Respondents who reported they are of Hispanic origin. ( _HISPANC=1) 9 Don’t know/ Not sure/ Refused
                '_EDUCAG',      # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
                '_INCOMG',      # Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
                '_METSTAT',     # metropolitan status 1 yes, 2 no
                '_URBSTAT',     # urban rural status 1 urban 2 rural
                '_SMOKER3',     # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
                'DRNKANY5',     # had at least one drink of alcohol in the past 30 days
                '_RFBING5',     # binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion 1 no 2 yes
                '_RFDRHV7',     # heavy drinkers 14 drinks per week or less, or Female Respondents who reported having 7 drinks per week or less 1 no 2 yes
                '_PNEUMO3',     # ever had a pneumonia vaccination
                '_RFSEAT3',     # always wear seat belts 1 yes 2 no
                '_DRNKDRV',     # drinking and driving 1 yes 2 no
                '_RFMAM22',     # mammogram in the past two years 1 yes 2 no
                '_FLSHOT7',     # flu shot within the past year 1 yes 2 no
                '_RFPAP35',     # Pap test in the past three years 1 yes 2 no
                '_RFPSA23',     # PSA test in the past 2 years
                '_CRCREC1',     # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
                '_AIDTST4',     # ever been tested for HIV
                'PERSDOC2',     # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
                'CHCSCNCR',     # (Ever told) (you had) skin cancer? 1 yes 2 no
                'CHCOCNCR',     # (Ever told) (you had) any other types of cancer? 1 yes 2 no
                'CHCCOPD2',     #  (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
                'QSTLANG',     # 1 english 2 spanish
                'ADDEPEV3',     # (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)? 1 yes 2 no
                'CHCKDNY2',     # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
                'DIABETE4',     # (Ever told) (you had) diabetes? 1 yes 2 no
                'MARITAL',      #  (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
                '_MICHD'        # ever reported having coronary heart disease (CHD) or myocardial infarction (MI) 1 yes 2 no
                ]

features_num = ['_AGE80',       #  imputed age value collapsed above 80
                'HTM4',  # height in centimeters
                'WTKG3',  # weight in kilograms, implied 2 decimal places
                '_BMI5',  # body mass index
                '_CHLDCNT',  # number of children in household.
                '_DRNKWK1',  # total number of alcoholic beverages consumed per week.
                'SLEPTIM1',  # how many hours of sleep do you get in a 24-hour period?
                ]


def load_data(name):
	data_1 = pd.read_sas('./source/' + name)
	data_2 = data_1.copy()
	return data_1, data_2


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def clean_data(data):
	data = data.dropna(subset=["_MICHD"], axis=0)
	data = data[data.DISPCODE != 1200]  # == 1200    final disposition (1100 completed or not 1200)
	data._RFHLTH = data._RFHLTH.replace(9, int(data._RFHLTH.mode()))
	data._PHYS14D = data._PHYS14D.replace(9, int(data._PHYS14D.mode()))
	data._MENT14D = data._MENT14D.replace(9, int(data._MENT14D.mode()))
	data._HCVU651 = data._HCVU651.replace(9, int(data._HCVU651.mode()))
	data._TOTINDA = data._TOTINDA.replace(9, int(data._TOTINDA.mode()))
	data._ASTHMS1 = data._ASTHMS1.replace(9, int(data._ASTHMS1.mode()))
	data._EXTETH3 = data._EXTETH3.replace(9, int(data._EXTETH3.mode()))
	data._DENVST3 = data._DENVST3.replace(9, int(data._DENVST3.mode()))
	data._RACE = data._RACE.replace(9, int(data._RACE.mode()))
	data._CHLDCNT = data._CHLDCNT.replace(9, int(data._CHLDCNT.mode()))
	data._EDUCAG = data._EDUCAG.replace(9, int(data._EDUCAG.mode()))
	data._INCOMG = data._INCOMG.replace(9, int(data._INCOMG.mode()))
	data._SMOKER3 = data._SMOKER3.replace(9, int(data._SMOKER3.mode()))
	data.DRNKANY5 = data.DRNKANY5.replace(9, int(data.DRNKANY5.mode()))
	data.DRNKANY5 = data.DRNKANY5.replace(7, int(data.DRNKANY5.mode()))
	data._RFBING5 = data._RFBING5.replace(9, int(data._RFBING5.mode()))
	data._DRNKWK1 = data._DRNKWK1.replace(99900, int(data._DRNKWK1.mode()))
	data._RFDRHV7 = data._RFDRHV7.replace(9, int(data._RFDRHV7.mode()))
	data._PNEUMO3 = data._PNEUMO3.replace(9, int(data._PNEUMO3.mode()))
	data._RFSEAT3 = data._RFSEAT3.replace(9, int(data._RFSEAT3.mode()))
	data._DRNKDRV = data._DRNKDRV.replace(9, int(data._DRNKDRV.mode()))
	data._RFMAM22 = data._RFMAM22.replace(9, int(data._RFMAM22.mode()))
	data._FLSHOT7 = data._FLSHOT7.replace(9, int(data._FLSHOT7.mode()))
	data._RFPAP35 = data._RFPAP35.replace(9, int(data._RFPAP35.mode()))
	data._RFPSA23 = data._RFPSA23.replace(9, int(data._RFPSA23.mode()))
	data._AIDTST4 = data._AIDTST4.replace(9, int(data._AIDTST4.mode()))
	data.PERSDOC2 = data.PERSDOC2.replace(9, int(data.PERSDOC2.mode()))
	data.PERSDOC2 = data.PERSDOC2.replace(7, int(data.PERSDOC2.mode()))
	data.SLEPTIM1 = data.SLEPTIM1.replace(77, int(data.SLEPTIM1.mode()))
	data.SLEPTIM1 = data.SLEPTIM1.replace(99, int(data.SLEPTIM1.mode()))
	data.CHCSCNCR = data.CHCSCNCR.replace(7, int(data.CHCSCNCR.mode()))
	data.CHCSCNCR = data.CHCSCNCR.replace(9, int(data.CHCSCNCR.mode()))
	data.CHCOCNCR = data.CHCOCNCR.replace(7, int(data.CHCOCNCR.mode()))
	data.CHCOCNCR = data.CHCOCNCR.replace(9, int(data.CHCOCNCR.mode()))
	data.CHCCOPD2 = data.CHCCOPD2.replace(7, int(data.CHCCOPD2.mode()))
	data.CHCCOPD2 = data.CHCCOPD2.replace(9, int(data.CHCCOPD2.mode()))
	data.ADDEPEV3 = data.ADDEPEV3.replace(7, int(data.ADDEPEV3.mode()))
	data.ADDEPEV3 = data.ADDEPEV3.replace(9, int(data.ADDEPEV3.mode()))
	data.CHCKDNY2 = data.CHCKDNY2.replace(7, int(data.CHCKDNY2.mode()))
	data.CHCKDNY2 = data.CHCKDNY2.replace(9, int(data.CHCKDNY2.mode()))
	data.DIABETE4 = data.DIABETE4.replace(2, 1)
	data.DIABETE4 = data.DIABETE4.replace(4, 3)
	data.DIABETE4 = data.DIABETE4.replace(3, 2)
	data.DIABETE4 = data.DIABETE4.replace(7, int(data.DIABETE4.mode()))
	data.DIABETE4 = data.DIABETE4.replace(9, int(data.DIABETE4.mode()))
	data.MARITAL = data.MARITAL.replace(9, int(data.MARITAL.mode()))
	data.CVDSTRK3 = data.CVDSTRK3.replace(9, int(data.CVDSTRK3.mode()))
	data.CVDSTRK3 = data.CVDSTRK3.replace(7, int(data.CVDSTRK3.mode()))
	data = data[data.QSTLANG < 3]  # responded english or spanish to language (only 1 respondent said other)
	return data


def preprocess(inputs):
	preprocessed = pd.DataFrame()
	
	for cat in features_cat:
		# print(cat)
		one_hots = OneHotEncoder()
		cat_encoded = one_hots.fit_transform(inputs[[cat]])
		cat_encoded_names = one_hots.get_feature_names_out([cat])
		cat_encoded = pd.DataFrame(cat_encoded.todense(), columns=cat_encoded_names)
		# print(cat_encoded_names)
		# print(len(cat_encoded_names))
		preprocessed = pd.concat([preprocessed, cat_encoded], axis=1)
	
	for num in features_num:
		num_scaled = StandardScaler().fit_transform(inputs[[num]])
		num_scaled = pd.DataFrame(num_scaled, columns=[num])
		preprocessed = pd.concat([preprocessed, num_scaled], axis=1)
	
	return preprocessed


def process(prediction_data):
	# rows_to_keep = q.shape[0]
	rows_to_keep = prediction_data.shape[0]
	
	# inputs = pd.concat([X, z])
	inputs = pd.concat([X, prediction_data])
	inputs.shape
	
	# todo: replace NaNs with most frequent (mode) (X_mode)
	
	processed = pd.DataFrame()
	
	for cat in features_cat:
		# print(cat)
		one_hots = OneHotEncoder()
		cat_encoded = one_hots.fit_transform(inputs[[cat]])
		cat_encoded_names = one_hots.get_feature_names_out([cat])
		cat_encoded = pd.DataFrame(cat_encoded.todense(), columns=cat_encoded_names)
		# print(cat_encoded_names)
		# print(len(cat_encoded_names))
		processed = pd.concat([processed, cat_encoded], axis=1)
	
	for num in features_num:
		num_scaled = StandardScaler().fit_transform(inputs[[num]])
		num_scaled = pd.DataFrame(num_scaled, columns=[num])
		processed = pd.concat([processed, num_scaled], axis=1)
	
	to_model = processed.iloc[processed.shape[0] - rows_to_keep:].copy()
	to_model.shape
	
	return to_model


if __name__ ==  "__main__":
	data = pd.read_hdf(raw_file)  # to read cleaned data
	data.shape

	data = clean_data(data)
	data.shape
	
	# data.CVDSTRK3.unique()
	# data._MICHD.describe()
	# cols = data.columns
	# print(cols)
	# data.describe()
	# data.head()
	# data.columns
	
	X = data.drop([i for i in data.columns if i in data.columns and i not in features_cat and i not in features_num and i not in [outcome]], axis=1)
	X.shape
	
	y = abs(data.CVDSTRK3 - 2)
	y.head()
	# y.describe()
	# y.shape
	# len([i for i in y if i == 1])
	# y.value_counts()
	# y.value_counts(1)
	X = X.drop([outcome], axis=1)
	
	
	X.shape
	X.head()
	
	X_mode = X.mode()
	X_mode
	X_mode.shape
	imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
	X.shape
	X.head()
	X.isnull().values.any()
	X
	
	
	# X.to_hdf(cleaned_file, "X", complevel=2)  # to save cleaned data
	# X = pd.read_hdf(cleaned_file)  # to read cleaned data
	X.shape
	
	# rus = RandomUnderSampler(random_state=random_state)
	# X_rus, y_rus = rus.fit_resample(X, y)
	# y_rus.value_counts()
	# X_rus.shape
	# y.value_counts()
	
	# cc = ClusterCentroids(random_state=random_state)
	# X_cc, y_cc = cc.fit_resample(X, y)
	# y_cc.value_counts()
	# X_cc.shape
	
	nm = NearMiss(version=3)
	X_nm, y_nm = nm.fit_resample(X, y)
	y_nm.value_counts()
	X_nm.shape
	
	
	train_X, val_X, train_y, val_y = train_test_split(X_nm, y_nm,  # X_rus, y_rus,  # X, y, # X_cc, y_cc,  #
	                                                  random_state=random_state)  # ,, stratify=y)  #
	train_X.shape
	train_y.describe()
	train_y.value_counts()
	val_y.value_counts()
	
	# X_cats = train_X.drop([i for i in X.columns if i in X.columns and i not in features_cat], axis=1)
	# X_nums = train_X.drop([i for i in X.columns if i in X.columns and i not in features_num], axis=1)
	# X_cats.shape
	# X_nums.shape
	# X_cats.head()
	
	rf = RandomForestClassifier(random_state=random_state)
	rf.fit(X_nm, y_nm)  # train_X, train_y)    # X, y)  # X_rus, y_rus) #
	
	dump(rf, model_name, compress=3)

	
	y_predictions = rf.predict(val_X)
	
	accuracy_score(val_y, y_predictions)    # 0.9605428560967573
	
	matrix = confusion_matrix(val_y, y_predictions)
	matrix
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	val_X.value_counts()
	val_y.value_counts()

	
	plt.figure(figsize=(8, 7))
	sns.set(font_scale=1.4)
	sns.heatmap(matrix, annot=True, annot_kws={
			'size': 10},
	            cmap=plt.cm.Greens, linewidths=0.2)
	class_names = ["No stroke", "Stroke"]
	tick_marks = np.arange(len(class_names))
	tick_marks2 = tick_marks + 0.5
	plt.xticks(tick_marks, class_names, rotation=25)
	plt.yticks(tick_marks2, class_names, rotation=0)
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title('Confusion Matrix for Random Forest Model - NearMiss-3')
	plt.show()
	
	print(classification_report(val_y, y_predictions))
	
	train_scoreNum, test_scoreNum = validation_curve(
			RandomForestClassifier(),
			X=train_X, y=train_y,
			param_name='n_estimators',
			param_range=n_estimators,
			cv=3,
			verbose=2)
	
	train_mean = np.mean(train_scoreNum, axis=1)
	test_mean = np.mean(test_scoreNum, axis=1)
	
	plt.plot(n_estimators, train_mean,
	         marker='o', markersize=5,
	         color='blue', label='Training Accuracy')
	plt.plot(n_estimators, test_mean,
	         marker='o', markersize=5,
	         color='green', label='Validation Accuracy')
	
	
	
	
	rf_random = RandomizedSearchCV(estimator=rf,
	                               param_distributions=param_grid,
	                               n_iter=100,
	                               cv=5,
	                               verbose=2,
	                               random_state=42,
	                               n_jobs=-1)
	rf_random.fit(train_X, train_y)
	
	grid_search = GridSearchCV(estimator=rf,
	                           param_grid=param_grid,
	                           cv=3,
	                           n_jobs=-1,
	                           verbose=2)
	grid_search.fit(train_X, train_y)
	grid_search.best_params_
	best_grid = grid_search.best_estimator_
	
	# now, a random forest model
	forest_model = RandomForestClassifier(random_state=random_state)
	                                      # n_estimators=n_estimators,
	                                      # criterion=criterion,
	                                      # max_depth=max_depth,
	                                      # max_features=max_features)
	forest_model.fit(X, y)
	
	dump(forest_model, model_name, compress=3)
	# loaded_model = load(model_name)
	
	predictions = forest_model.predict_proba(val_X_preprocessed)
	predictions
	probabilities = [i[1] for i in predictions if i is not None]
	max(probabilities)
	preds = forest_model.predict(val_X_preprocessed)
	preds
	accuracy_score(val_y, preds)
	
	print(mean_absolute_error(val_y, predictions))
	
	
	
	
	# categoricals = data.select_dtypes(include=[np.object])
	# categoricals.columns
	# numericals = data.select_dtypes(include=[np.number])
	# numericals.columns
	#
	# sns.pairplot(data, hue="HeartDisease")
	# sns.countplot(x="HeartDisease", data=data)
	#
	# data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 0 if x == 'No' else 1)
	#
	# plt.figure(figsize=(15, 8))
	# ax = sns.kdeplot(data["BMI"][data.HeartDisease == 1], shade=True)  # color="darkturquoise",
	# sns.kdeplot(data["BMI"][data.HeartDisease == 0], shade=True)  # color="lightcoral",
	# plt.legend(['HeartDisease', 'non-HeartDisease'])
	# plt.title('Density Plot of HeartDisease for BMI')
	# ax.set(xlabel='BMI')
	# plt.xlim(10, 50)
	# plt.show()
	#
	# plt.figure(figsize=(15, 8))
	# ax = sns.kdeplot(data["SleepTime"][data.HeartDisease == 1], shade=True)
	# sns.kdeplot(data["SleepTime"][data.HeartDisease == 0], shade=True)
	# plt.legend(['HeartDisease', 'non-HeartDisease'])
	# plt.title('Density Plot of HeartDisease for SleepTime')
	# ax.set(xlabel='SleepTime')
	# plt.xlim(2, 15)
	# plt.show()
	#
	# plt.figure(figsize=(5, 3))
	# sns.barplot('AgeCategory', 'HeartDisease', data=data, )
	# plt.xticks(fontsize=12, rotation=90)
	# plt.yticks(fontsize=12)
	# plt.title('Density Plot of HeartDisease for Age')
	# plt.xlabel('AgeCategory', fontsize=11)
	# plt.ylabel('HeartDisease', fontsize=11)
	# plt.show()
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals.columns:
	# 	plt.subplot(6, 3, n)
	# 	sns.displot(data[feature], kde=True)
	# 	plt.xlabel(feature)
	# 	plt.ylabel("Count")
	# 	n += 1
	#
	# for column in data.columns:
	# 	if data[column].dtypes == "object":
	# 		data[column] = data[column].fillna(data[column].mode().iloc[0])
	# 		uniques = len(data[column].unique())
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in categoricals.columns:
	# 	plt.subplot(6, 3, n)
	# 	sns.countplot(x=feature, hue="HeartDisease", data=data)
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for col in categoricals:
	# 	plt.subplot(6, 3, n)
	# 	sns.countplot(x='Sex', hue=categoricals[col], data=data)
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['AlcoholDrinking'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['Diabetic'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['PhysicalActivity'])
	# 	n += 1
	#
	# n = 1
	# plt.figure(figsize=(15, 25))
	# for feature in numericals:
	# 	plt.subplot(6, 3, n)
	# 	sns.boxplot(y=data[feature], x=data['Race'])
	# 	n += 1
	
	
	
	
	pass