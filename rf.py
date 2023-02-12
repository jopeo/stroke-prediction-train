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
import seaborn as sns
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from bioinfokit.visuz import stat
import warnings
warnings.filterwarnings("ignore")


raw_file = "raw.h5"
cleaned_file = "stroke_cleaned_2.h5"
model_name = "stroke_model_RUS_2.joblib"
outcome = "CVDSTRK3"    # (Ever told) (you had) a stroke.

# todo: check hyperparams:
random_state = 1
# n_estimators = [50, 100, 250, 500]    #, 300, 500, 750, 800, 1200]
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

# param_grid = {"bootstrap": [True],
#               "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#               "max_features": ["auto", "sqrt"],
#               "min_samples_leaf": [1, 2, 4],
#               "min_samples_split": [2, 5, 10],
#               "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
#               }

# features_cat = ['_STATE',       # geographical state]
#                 'SEXVAR',       # Sex of Respondent 1 MALE, 2 FEMALE
#                 '_RFHLTH',      # Health Status  1 Good or Better Health 2 Fair or Poor Health
#                                     # 9 Don’t know/ Not Sure Or Refused/ Missing
#                 '_PHYS14D',     # Healthy Days 1 Zero days when physical health not good
#                                     #  2 1-13 days when physical health not good
#                                     # 3 14+ days when physical health not good
#                                     # 9 Don’t know/ Refused/Missing
#                 '_MENT14D',     # SAME AS PHYS
#                 '_HCVU651',     # Health Care Access  1 Have health care coverage 2 Do not have health care coverage 9 Don’t know/ Not Sure, Refused or Missing
#                 '_TOTINDA',     # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
#                 '_ASTHMS1',     # asthma? 1 current 2 former 3 never
#                 '_DRDXAR2',     # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
#                 '_EXTETH3',     # ever had teeth extracted? 1 no 2 yes 9 dont know
#                 '_DENVST3',     # dentist in past year? 1 yes 2 no 9 don't know
#                 '_RACE',        # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispanic Respondents who reported they are of Hispanic origin. ( _HISPANC=1) 9 Don’t know/ Not sure/ Refused
#                 '_EDUCAG',      # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
#                 '_INCOMG',      # Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
#                 '_METSTAT',     # metropolitan status 1 yes, 2 no
#                 '_URBSTAT',     # urban rural status 1 urban 2 rural
#                 '_SMOKER3',     # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
#                 'DRNKANY5',     # had at least one drink of alcohol in the past 30 days
#                 '_RFBING5',     # binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion 1 no 2 yes
#                 '_RFDRHV7',     # heavy drinkers 14 drinks per week or less, or Female Respondents who reported having 7 drinks per week or less 1 no 2 yes
#                 '_PNEUMO3',     # ever had a pneumonia vaccination
#                 '_RFSEAT3',     # always wear seat belts 1 yes 2 no
#                 '_DRNKDRV',     # drinking and driving 1 yes 2 no
#                 '_RFMAM22',     # mammogram in the past two years 1 yes 2 no
#                 '_FLSHOT7',     # flu shot within the past year 1 yes 2 no
#                 '_RFPAP35',     # Pap test in the past three years 1 yes 2 no
#                 '_RFPSA23',     # PSA test in the past 2 years
#                 '_CRCREC1',     # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
#                 '_AIDTST4',     # ever been tested for HIV
#                 'PERSDOC2',     # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
#                 'CHCSCNCR',     # (Ever told) (you had) skin cancer? 1 yes 2 no
#                 'CHCOCNCR',     # (Ever told) (you had) any other types of cancer? 1 yes 2 no
#                 'CHCCOPD2',     #  (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
#                 'QSTLANG',     # 1 english 2 spanish
#                 'ADDEPEV3',     # (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)? 1 yes 2 no
#                 'CHCKDNY2',     # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
#                 'DIABETE4',     # (Ever told) (you had) diabetes? 1 yes 2 no
#                 'MARITAL',      #  (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
#                 '_MICHD'        # ever reported having coronary heart disease (CHD) or myocardial infarction (MI) 1 yes 2 no
#                 ]
#
# features_num = ['_AGE80',       #  imputed age value collapsed above 80
#                 'HTM4',  # height in centimeters
#                 'WTKG3',  # weight in kilograms, implied 2 decimal places
#                 '_BMI5',  # body mass index
#                 '_CHLDCNT',  # number of children in household.
#                 '_DRNKWK1',  # total number of alcoholic beverages consumed per week.
#                 'SLEPTIM1',  # how many hours of sleep do you get in a 24-hour period?
#                 ]

features = [
		'_STATE',       # State FIPS Code
		'_AGE80',  # Imputed Age value collapsed above 80
		'SEXVAR',  # Sex of Respondent 1 MALE, 2 FEMALE
		'_RACE',  # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispanic Respondents who reported they are of Hispanic origin. ( _HISPANC=1) 9 Don’t know/ Not sure/ Refused
		# 'HTIN4',  # Reported height in inches
		# 'WTKG3',  # Reported weight in kilograms
		'_BMI5',  # Body Mass Index (BMI)
		'_TOTINDA',  # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
		'_EDUCAG',  # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
		'GENHLTH',      # Would you say that in general your health is:
		'PHYSHLTH',     # Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?
		'MENTHLTH',     # Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?
		'POORHLTH',     # During the past 30 days, for about how many days did poor physical or mental health keep you from doing your usual activities, such as self-care, work, or recreation?
		'SLEPTIM1',     # On average, how many hours of sleep do you get in a 24-hour period?
		'MARITAL',  # Are you: (marital status)
		'EMPLOY1',  # Are you currently…?
		'_INCOMG',  # Income categories
		'_DRNKWK1',  # Calculated total number of alcoholic beverages consumed per week
		'_SMOKER3',  # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
		'PERSDOC2',  # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
		'CVDINFR4',  # (Ever told) you had a heart attack, also called a myocardial infarction?
		'CVDCRHD4',  # (Ever told) (you had) angina or coronary heart disease?
		'RMVTETH4',  # Not including teeth lost for injury or orthodontics, how many of your permanent teeth have been removed because of tooth decay or gum disease?
		'_DENVST3',  # dentist in past year? 1 yes 2 no 9 don't know
		'FALL12MN',     # In the past 12 months, how many times have you fallen?
		'DIABETE4',     # (Ever told) (you had) diabetes?
		'CHCKDNY2',     # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
		'FLUSHOT7',     # During the past 12 months, have you had either flu vaccine that was sprayed in your nose or flu shot injected into your arm?
		'_CRCREC1',     # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
		'_AIDTST4',     # ever been tested for HIV
		'CHCSCNCR',     # (Ever told) (you had) skin cancer? 1 yes 2 no
		'CHCOCNCR',     # (Ever told) (you had) any other types of cancer? 1 yes 2 no
		'CHCCOPD2',     # (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
		'_ASTHMS1',     # asthma? 1 current 2 former 3 never
		'_DRDXAR2',     # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
]
len(features)

RFE_features = ['_STATE',       # State FIPS Code
                # 'FMONTH',       # File Month
                # 'IDATE',        #  Interview Date
                # 'IMONTH',       #
                # 'IDAY',         #
                # 'SEQNO',        # Annual Sequence Number
                # '_PSU',         # Primary Sampling Unit (Equal to Annual Sequence Number)
                'GENHLTH',      # Would you say that in general your health is:
                'PHYSHLTH',     # Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?
                'MENTHLTH',     # Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?
                'POORHLTH',     # During the past 30 days, for about how many days did poor physical or mental health keep you from doing your usual activities, such as self-care, work, or recreation?
                'SLEPTIM1',     # On average, how many hours of sleep do you get in a 24-hour period?
                'CVDINFR4',     # (Ever told) you had a heart attack, also called a myocardial infarction?
                'CVDCRHD4',     # (Ever told) (you had) angina or coronary heart disease?
                # 'DIABAGE3',     # How old were you when you were told you had diabetes?
                'RMVTETH4',     # Not including teeth lost for injury or orthodontics, how many of your permanent teeth have been removed because of tooth decay or gum disease?
                'MARITAL',      # Are you: (marital status)
                'EMPLOY1',      # Are you currently…?
                # 'INCOME2',      # Is your annual household income from all sources: (If respondent refuses at any income level, code ´Refused.´)
                # 'WEIGHT2',      # About how much do you weigh without shoes? (If respondent answers in metrics, put a 9 in the first column)[Round fractions up.]
                # 'HEIGHT3',      # About how tall are you without shoes? (If respondent answers in metrics, put a 9 in the first column)[Round fractions down.]
                # 'FLSHTMY3',     # During what month and year did you receive your most recent flu vaccine that was sprayed in your nose or flu shot injected into your arm?
                'FALL12MN',     # In the past 12 months, how many times have you fallen?
                # 'COLNTEST',     # How long has it been since you had this test?
                # 'HIVTSTD3',     # Not including blood donations, in what month and year was your last H.I.V. test? (If response is before January 1985, code ´777777´.)
                # 'CNCRAGE',      # At what age were you told that you had cancer? (If Response = 2 (Two) or 3 (Three or more), ask: “At what age was your first diagnosis of cancer?”)
                # '_STSTR',       # Sample Design Stratification Variable (Prior to 2011: _STSTR is a five digit number that combines the values for _STATE (first two characters), _GEOSTR (third and fourth character), and _DENSTR2 (final character).)
                # '_STRWT',       # Stratum weight
                # '_WT2RAKE',     # Design weight used in raking (Stratum weight (_STRWT) multiplied by the raw weighting factor (_RAWRAKE).)
                # '_DUALCOR',     # Dual phone use correction factor
                # '_LLCPWT2',     # Truncated design weight used in adult combined land line and cell phone raking
                # '_LLCPWT',      # Truncated design weight used in adult combined land line and cell phone raking
                # '_MICHD',       # Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
                # '_AGEG5YR',     # Fourteen-level age category
                '_AGE80',       # Imputed Age value collapsed above 80
                'HTIN4',        # Reported height in inches
                # 'HTM4',         # Reported height in meters
                'WTKG3',        # Reported weight in kilograms
                '_BMI5'         # Body Mass Index (BMI)
                '_INCOMG'   ,   # Income categories
                '_DRNKWK1'  ]   # Calculated total number of alcoholic beverages consumed per week

add_features = [
		'DIABETE4',         # (Ever told) (you had) diabetes?
		'CHCKDNY2',  # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
		'FLUSHOT7',         # During the past 12 months, have you had either flu vaccine that was sprayed in your nose or flu shot injected into your arm?
		'_CRCREC1',     # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
		'_AIDTST4',  # ever been tested for HIV
		'CHCSCNCR',  # (Ever told) (you had) skin cancer? 1 yes 2 no
		'CHCOCNCR',  # (Ever told) (you had) any other types of cancer? 1 yes 2 no
		'CHCCOPD2',  # (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
		'_RACE',  # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispanic Respondents who reported they are of Hispanic origin. ( _HISPANC=1) 9 Don’t know/ Not sure/ Refused
		'_EDUCAG',  # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
		'SEXVAR',  # Sex of Respondent 1 MALE, 2 FEMALE
		'_TOTINDA',  # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
		'_ASTHMS1',  # asthma? 1 current 2 former 3 never
		'_DRDXAR2',  # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
		'_DENVST3',  # dentist in past year? 1 yes 2 no 9 don't know
		'_SMOKER3',  # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
		'PERSDOC2',  # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
]


def load_data(name):
	# data_1 = pd.read_sas('./source/' + name)
	data_1 = pd.read_hdf(raw_file)  # to read cleaned data
	data_2 = data_1.copy()
	return data_1, data_2


def clean_data(data):
	data = data.dropna(subset=[outcome], axis=0)
	data = data[data.DISPCODE != 1200]  # == 1200    final disposition (1100 completed or not 1200)
	
	data = data.drop([i for i in data.columns if i in data.columns and i not in features and i not in outcome], axis=1)
	
	data.DIABETE4 = data.DIABETE4.replace(2, 1)
	data.DIABETE4 = data.DIABETE4.replace(4, 3)
	data.DIABETE4 = data.DIABETE4.replace(3, 2)

	for feat in [i for i in features + [outcome] if i in features + [outcome] and i not in ['_INCOMG', '_RACE']]:
		to_replace = [7, 9, 77, 99, 99900]
		for num in to_replace:
			if num in data[feat].values:
				data[feat] = data[feat].replace(num, int(data[feat].mode()))

		to_zero = [8, 88]
		for z in to_zero:
			if z in data[feat].values:
				data[feat] = data[feat].replace(z, 0)
				
		print(data[feat].value_counts())
	
	X = data.drop([outcome], axis=1)
	
	y = abs(data[outcome] - 2)
	y.value_counts()
	

	return X, y


def imp_mode(data):
	imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	X = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
	return X


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
	data_o, data = load_data(raw_file)
	data_o, data = load_data(cleaned_file)
	data.shape

	X, y = clean_data(data)
	X.shape
	y.value_counts()
	
	X = imp_mode(X)
	X.isnull().values.any()
	X.shape

	# X.to_hdf(cleaned_file, "X", complevel=2)  # to save cleaned data
	# X = pd.read_hdf(cleaned_file)  # to read cleaned data
	
	rus = RandomUnderSampler(random_state=random_state)
	X_rus, y_rus = rus.fit_resample(X, y)
	y_rus.value_counts()
	X_rus.shape
	y.value_counts()
	
	# cc = ClusterCentroids(random_state=random_state)
	# X_cc, y_cc = cc.fit_resample(X, y)
	# y_cc.value_counts()
	# X_cc.shape
	
	nm = NearMiss(version=1)
	X_nm, y_nm = nm.fit_resample(X, y)
	y_nm.value_counts()
	X_nm.shape
	
	
	train_X, val_X, train_y, val_y = train_test_split(X_nm, y_nm,  # X_rus, y_rus,  # X, y, # X_cc, y_cc,  #
	                                                  random_state=random_state)	# , # , stratify=y)  # )
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
	rf.fit(train_X, train_y)    # X, y)  # X_rus, y_rus)  # X_nm, y_nm)  #
	
	dump(rf, model_name, compress=3)

	
	y_predictions = rf.predict(val_X)
	
	accuracy_score(val_y, y_predictions)    # 0.9605428560967573
	
	matrix = confusion_matrix(val_y, y_predictions)
	matrix
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	val_X.value_counts()
	val_y.value_counts()
	pd.DataFrame(y_predictions).value_counts()

	
	plt.figure(figsize=(12,9))
	sns.set(font_scale=1.4)
	sns.heatmap(matrix, annot=True, annot_kws={
			'size': 10},
	            cmap=plt.cm.Blues, linewidths=0.2)     # plt.cm.Greens plt.cm.Blues  BuPu
	class_names = ["No stroke", "Stroke"]
	tick_marks = np.arange(len(class_names))
	tick_marks2 = tick_marks + 0.5
	plt.xticks(tick_marks, class_names, rotation=25)
	plt.yticks(tick_marks2, class_names, rotation=0)
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title('Confusion Matrix for Random Forest Model - RUS_2')
	plt.show()
	
	print(classification_report(val_y, y_predictions))

	fpr, tpr, thresholds = roc_curve(y_true=list(val_y), y_score=list(y_predictions))
	auc = roc_auc_score(y_true=list(val_y), y_score=list(y_predictions))
	# plot ROC
	stat.roc(fpr=fpr, tpr=tpr, auc=auc, shade_auc=True, per_class=True, legendpos='upper center',
			 legendanchor=(0.5, 1.08), legendcols=3)

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
	
	pass