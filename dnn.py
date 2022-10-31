#!/usr/bin/env python

# conda activate tf

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.impute import SimpleImputer
print(tf.config.list_physical_devices('GPU'))
print(tf.reduce_sum(tf.random.normal([1000, 1000])))


#  todo: visit https://www.cdc.gov/brfss/annual_data/annual_2020.html
#   to download the data.
#   the SAS Transport Format is used here:

full_file = "LLCP2020.XPT"
df_name = "./source/" + "df.h5"
model_name = "model8.h5"
fig_name = model_name.split('.')[0] + '_plots'


def load_data(name):
	data_1 = pd.read_sas('./source/' + name)
	data_2 = data_1.copy()
	return data_1, data_2


# todo: check state entries compared with model numbers
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
	
	to_model = processed.iloc[processed.shape[0]-rows_to_keep:].copy()
	to_model.shape
	
	return to_model


# data.head()
#
# len(data[data._PHYS14D == 9])
#
# data.DIABETE4.unique()
# data.DIABETE4.describe()
# data.HTIN4.mode()


#   todo: visit https://www.cdc.gov/brfss/annual_data/2020/pdf/codebook20_llcp-v2-508.pdf
#    for catalog of features


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
                'MARITAL'      #  (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
                ]

features_num = ['_AGE80',       #  imputed age value collapsed above 80
                'HTM4',  # height in centimeters
                'WTKG3',  # weight in kilograms, implied 2 decimal places
                '_BMI5',  # body mass index
                '_CHLDCNT',  # number of children in household.
                '_DRNKWK1',  # total number of alcoholic beverages consumed per week.
                'SLEPTIM1',  # how many hours of sleep do you get in a 24-hour period?
                ]

# train_X = train_X.toarray()
# val_X = val_X.toarray()


# def clean_data(data):
# 	# data = data.dropna(axis=1)
# 	#
# 	# data = data.drop([
# 	# 		'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE',
# 	# 		'SEQNO', '_PSU', 'QSTVER', '_STSTR', '_STRWT','_RAWRAKE',
# 	# 		'_WT2RAKE', '_DUALUSE', '_LLCPWT2', '_LLCPWT', '_AGEG5YR',
# 	# 		'_AGE65YR', '_AGE_G'
# 	# 		], axis=1)
# 	# X = data.drop(['_MICHD', 'CVDCRHD4', 'CVDSTRK3',
# 	#                ], axis=1)
# 	# y = abs(data._MICHD - 2)
#
# 	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# 	return data, y, X, train_X, val_X, train_y, val_y


if __name__ ==  "__main__":
	data_o, data = load_data(full_file)
	
	data = clean_data(data)
	
	data_o.shape
	data.shape
	data.head()
	

	X = data.drop([i for i in data.columns if i in data.columns and i not in features_cat and i not in features_num and i not in ['_MICHD']], axis=1)
	X.shape
	
	y = abs(data._MICHD - 2)
	y.head()
	y.describe()
	X = X.drop(['_MICHD'], axis=1)
	
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
	
	# os.getcwd()
	X.to_hdf(df_name, "X")
	# X2 = pd.read_hdf(df_name)
	# X2.shape
	# X2.head()


	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
	# input_shape = [train_X.shape[1]]
	# input_shape  # = 45
	
	X_cats = train_X.drop([i for i in X.columns if i in X.columns and i not in features_cat], axis=1)
	X_nums = train_X.drop([i for i in X.columns if i in X.columns and i not in features_num], axis=1)
	X_cats.shape
	X_nums.shape
	X_cats.head()
	
	# train_X_preprocessed = preprocess(train_X)
	# val_X_preprocessed = preprocess(val_X)
	# input_shape = [train_X_preprocessed.shape[1]]
	
	X_preprocessed = preprocess(X)
	X_preprocessed.shape
	input_shape = [X_preprocessed.shape[1]]
	
	input_shape

	# encoded = pd.concat([nums_scaled, cats_encoded], axis=1)
	# encoded

	
	# my_layers = []
	# layer = layers.IntegerLookup(output_mode='one_hot')
	# layer.adapt(X[features_cat[0]])
	# layer([6])
	# layer.get_vocabulary()
	# my_layers.append(layer)
	
# 	my_layers = []
# 	for feat in features_cat:
# 		layer = layers.CategoryEncoding(
# 				num_tokens=len(train_X[feat].unique()),
# 				output_mode='one_hot'
# 		)
# 		my_layers.append(layer)
# 		print(f'Appended {feat} layer {layer}')
#
# 	for feat in features_num:
# 		layer = layers.Normalization()
# 		layer.adapt(train_X[feat])
# 		my_layers.append(layer)
# 		print(f'Appended {feat} layer {layer}')
#
# 	def preprocessing_layer(inputs):
# 		preprocessed = pd.DataFrame()
# 		for cat in features_cat:
# 			pre_data = my_layers[features_cat.index(cat)](inputs[cat])
# 			pre_data = pd.DataFrame(pre_data)
# 			preprocessed = pd.concat([preprocessed, pre_data], axis=1)
# 			print(pre_data)
# 		for num in features_num:
# 			# print(features_num.index(feat)+38)
# 			# print(features_num.index(feat))
# 			pre_data = my_layers[features_num.index(num)+38](inputs[num])
# 			pre_data = pd.DataFrame(pre_data)
# 			preprocessed = pd.concat([preprocessed, pre_data], axis=1)
# 			# print(pre_data)
# 		return preprocessed
#
# 	q = preprocessing_layer(z.iloc[0])
# 	q
# 	q.shape
# 	z.iloc[0]
# 	z
# 	w = pd.concat(q)
# 	w.concat(q)
# 	type(q)
#
# z.shape


	
	# layer = layers.CategoryEncoding(
		# 		num_tokens=len(X[features_cat[0]].unique()),
		# 		output_mode='one_hot'
		# )
		# layer(X[features_cat[0]])
		
	# layer.get_vocabulary()
	# layer([1])
	# len(X[features_cat[0]].unique())
	# X[features_cat[0]].unique()
	
	# preprocessor = make_column_transformer(
	# 		(StandardScaler(), features_num),
	# 		(encoder, features_cat),
	# )
	# X = pd.DataFrame(preprocessor.fit_transform(X).toarray())
	
	# X.isnull().values.any()
	# X.shape
	
	# lookup = layers.IntegerLookup(output_mode='int')
	# lookup.adapt(train_X)
	# lookup(val_X)
	
	early_stopping = EarlyStopping(
			min_delta=0.001,  # minimium amount of change to count as an improvement
			patience=5,  # how many epochs to wait before stopping
			restore_best_weights=True,
	)
	
	m = 2
	
	#  'relu' activation -- 'elu', 'selu', and 'swish'
	model = keras.Sequential([
			layers.BatchNormalization(input_shape=input_shape),
			# the hidden ReLU layers
			layers.Dense(units=64 * m, activation='relu'),  # , input_shape=input_shape),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			layers.Dense(units=64 * m, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			layers.Dense(units=64 * m, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
			# the linear output layer
			# layers.Dense(units=1),
			layers.Dense(1, activation='sigmoid')
	])
	
	model.compile(
			optimizer="adam",
			# loss="mae",
			loss='binary_crossentropy',
			metrics=['binary_accuracy'],
	)
	
	# outs = []
	# for array in train_X, train_y, val_X, val_y:
	# 	array = np.asarray(array).astype('float32')
	# 	outs.append(array)
	
	history = model.fit(
			X_preprocessed, y,   #train_X_preprocessed, train_y,  # X, y,    # train_X, train_y,  #
			# validation_data=(val_X_preprocessed, val_y),
			# validation_data=(val_X, val_y),
			batch_size=256*2*m,
			epochs=8,
			callbacks=[early_stopping],  # put your callbacks in a list
			# verbose=0,  # turn off training log
	)
	
	# convert the training history to a dataframe
	# history_df = pd.DataFrame(history.history)
	# history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
	# history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
	#
	# df1 = pd.DataFrame(history_df.loc[:, ['loss', 'val_loss']])
	# df2 = pd.DataFrame(history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']])
	# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 16))
	# axe = axes.ravel()
	# df1.plot(ax=axe[0], title="Cross-entropy")
	# df2.plot(ax=axe[1], title="Accuracy")
	# plt.tight_layout()
	# fig.savefig('./source/' + fig_name)
	# plt.show()
	
	# model.save('./source/' + model_name)

	# model = load_model('./source/' + model_name)
	
	# z = pd.DataFrame(0, index=range(3), columns=X.columns)
	# z.shape
	# z.iloc[0] = X.mean().astype(int).transpose()
	# z.iloc[1] = pd.DataFrame(0, index=range(1), columns=X.columns)
	# z = z.astype(float)
	# z.shape
	# z
	# z.columns
	#
	# n = 1
	# z.iloc[n] = pd.DataFrame(0, index=range(1), columns=X.columns)
	# z.iloc[n]._STATE        = 6         # geographical state]
	# z.iloc[n].SEXVAR        = 1          # Sex of Respondent 1 MALE, 2 FEMALE
	# z.iloc[n]._RFHLTH       = 1        # Health Status  1 Good or Better Health 2 Fair or Poor Health	# 9 Don’t know/ Not Sure Or Refused/ Missing
	# z.iloc[n]._PHYS14D      = 1        # Healthy Days 1 Zero days when physical health not good 	#  2 1-13 days when physical health not good # 3 14+ days when physical health not good # 9 Don’t know/ Refused/Missing
	# z.iloc[n]._MENT14D      = 2        # SAME AS PHYS
	# z.iloc[n]._HCVU651      =  1       # Health Care Access  1 Have health care coverage 2 Do not have health care coverage 9 Don’t know/ Not Sure, Refused or Missing
	# z.iloc[n]._TOTINDA      =  1       # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
	# z.iloc[n]._ASTHMS1      =  3      # asthma? 1 current 2 former 3 never
	# z.iloc[n]._DRDXAR2      =  2       # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
	# z.iloc[n]._EXTETH3      =  1       # ever had teeth extracted? 1 no 2 yes 9 dont know
	# z.iloc[n]._DENVST3      =  2       # dentist in past year? 1 yes 2 no 9 don't know
	# z.iloc[n]._RACE     =     1          # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispan
	# z.iloc[n]._EDUCAG       =  3       # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
	# z.iloc[n]._INCOMG       =   5      # Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
	# z.iloc[n]._METSTAT      =   2      # metropolitan status 1 yes, 2 no
	# z.iloc[n]._URBSTAT      =   1      # urban rural status 1 urban 2 rural
	# z.iloc[n]._SMOKER3      =   4      # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
	# z.iloc[n].DRNKANY5      =   1      # had at least one drink of alcohol in the past 30 days
	# z.iloc[n]._RFBING5      =   1      # binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion 1 no 2 yes
	# z.iloc[n]._RFDRHV7      =   1      # heavy drinkers 14 drinks per week or less, or Female Respondents who reported having 7 drinks per week or less 1 no 2 yes
	# z.iloc[n]._PNEUMO3      =   2      # ever had a pneumonia vaccination
	# z.iloc[n]._RFSEAT3      =   1      # always wear seat belts 1 yes 2 no
	# z.iloc[n]._DRNKDRV      =   2      # drinking and driving 1 yes 2 no
	# z.iloc[n]._RFMAM22      =   2      # mammogram in the past two years 1 yes 2 no
	# z.iloc[n]._FLSHOT7      =   1      # flu shot within the past year 1 yes 2 no
	# z.iloc[n]._RFPAP35      =   2      # Pap test in the past three years 1 yes 2 no
	# z.iloc[n]._RFPSA23      =   2      # PSA test in the past 2 years
	# z.iloc[n]._CRCREC1      =   3      # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
	# z.iloc[n]._AIDTST4      =   2      # ever been tested for HIV
	# z.iloc[n].PERSDOC2      =   3      # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
	# z.iloc[n].CHCSCNCR      =   2      # (Ever told) (you had) skin cancer? 1 yes 2 no
	# z.iloc[n].CHCOCNCR      =   2      # (Ever told) (you had) any other types of cancer? 1 yes 2 no
	# z.iloc[n].CHCCOPD2      =   2      # (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
	# z.iloc[n].QSTLANG       =   1      # 1 english 2 spanish
	# z.iloc[n].ADDEPEV3      =   2      # (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)? 1 yes 2 no
	# z.iloc[n].CHCKDNY2      =   2      # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
	# z.iloc[n].DIABETE4      =   2      # (Ever told) (you had) diabetes? 1 yes 2 no
	# z.iloc[n].MARITAL       =   1      # (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
	# z.iloc[n]._AGE80        =   32      # imputed age value collapsed above 80
	# z.iloc[n].HTM4      =       177        # height in centimeters
	# z.iloc[n].WTKG3     =       73        # weight in kilograms, implied 2 decimal places
	# z.iloc[n]._BMI5     =       (177/(73^2))        # body mass index
	# z.iloc[n]._CHLDCNT      = 1        # number of children in household.
	# z.iloc[n]._DRNKWK1      = 0        # total number of alcoholic beverages consumed per week.
	# z.iloc[n].SLEPTIM1      = 6        # how many hours of sleep do you get in a 24-hour period?
	#
	# n = 2
	# z.iloc[n] = pd.DataFrame(0, index=range(1), columns=X.columns)
	# z.iloc[n]._STATE        = 1         # geographical state]
	# z.iloc[n].SEXVAR        = 1          # Sex of Respondent 1 MALE, 2 FEMALE
	# z.iloc[n]._RFHLTH       = 2        # Health Status  1 Good or Better Health 2 Fair or Poor Health	# 9 Don’t know/ Not Sure Or Refused/ Missing
	# z.iloc[n]._PHYS14D      = 3        # Healthy Days 1 Zero days when physical health not good 	#  2 1-13 days when physical health not good # 3 14+ days when physical health not good # 9 Don’t know/ Refused/Missing
	# z.iloc[n]._MENT14D      = 2        # SAME AS PHYS
	# z.iloc[n]._HCVU651      =  2       # Health Care Access  1 Have health care coverage 2 Do not have health care coverage 9 Don’t know/ Not Sure, Refused or Missing
	# z.iloc[n]._TOTINDA      =  2       # Exercise 1 Had physical activity or exercise 2 No physical activity or exercise in last 30 days 9 Don’t know/ Refused/ Missing
	# z.iloc[n]._ASTHMS1      =  2      # asthma? 1 current 2 former 3 never
	# z.iloc[n]._DRDXAR2      =  1       # ever arthritis? 1 Diagnosed with arthritis 2 Not diagnosed with arthritis
	# z.iloc[n]._EXTETH3      =  2       # ever had teeth extracted? 1 no 2 yes 9 dont know
	# z.iloc[n]._DENVST3      =  2       # dentist in past year? 1 yes 2 no 9 don't know
	# z.iloc[n]._RACE     =     2          # 1 White only, nonHispanic, 2 Black only, nonHispanic, 3 American Indian or Alaskan Native only,Non-Hispanic 4 Asian only, nonHispanic  5 Native Hawaiian or other Pacific Islander only, Non-Hispanic 6 Other race only, nonHispanic 7 Multiracial, nonHispanic 8 Hispan
	# z.iloc[n]._EDUCAG       =  2       # level of education completed 1 no grad high school, 2 high school, 3 some college, 4 graduated college, 9 don't know
	# z.iloc[n]._INCOMG       =   3      # Income categories (1 Less than $15,000, 2 $15,000 to less than $25,000, 3 $25,000 to less than $35,000, 4 $35,000 to less than $50,000, 5 $50,000 or more, 9 dont know
	# z.iloc[n]._METSTAT      =   2      # metropolitan status 1 yes, 2 no
	# z.iloc[n]._URBSTAT      =   2      # urban rural status 1 urban 2 rural
	# z.iloc[n]._SMOKER3      =   1      # four-level smoker status: everyday smoker, someday smoker, former smoker, non-smoker
	# z.iloc[n].DRNKANY5      =   2      # had at least one drink of alcohol in the past 30 days
	# z.iloc[n]._RFBING5      =   2      # binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion 1 no 2 yes
	# z.iloc[n]._RFDRHV7      =   2      # heavy drinkers 14 drinks per week or less, or Female Respondents who reported having 7 drinks per week or less 1 no 2 yes
	# z.iloc[n]._PNEUMO3      =   1      # ever had a pneumonia vaccination
	# z.iloc[n]._RFSEAT3      =   2      # always wear seat belts 1 yes 2 no
	# z.iloc[n]._DRNKDRV      =   1      # drinking and driving 1 yes 2 no
	# z.iloc[n]._RFMAM22      =   2      # mammogram in the past two years 1 yes 2 no
	# z.iloc[n]._FLSHOT7      =   1      # flu shot within the past year 1 yes 2 no
	# z.iloc[n]._RFPAP35      =   2      # Pap test in the past three years 1 yes 2 no
	# z.iloc[n]._RFPSA23      =   2      # PSA test in the past 2 years
	# z.iloc[n]._CRCREC1      =   3      # fully met the USPSTF recommendations for rectal cancer screening 1 yes, 2 yes but not within time, 3 never
	# z.iloc[n]._AIDTST4      =   2      # ever been tested for HIV
	# z.iloc[n].PERSDOC2      =   3      # personal doctor yes = 1, more = 2, no = 3 Do you have one person you think of as your personal doctor or health care provider? (If ´No´ ask ´Is there more than one or is there no person who you think of as your personal doctor or health care provider?´.)
	# z.iloc[n].CHCSCNCR      =   1      # (Ever told) (you had) skin cancer? 1 yes 2 no
	# z.iloc[n].CHCOCNCR      =   1      # (Ever told) (you had) any other types of cancer? 1 yes 2 no
	# z.iloc[n].CHCCOPD2      =   1      # (Ever told) (you had) chronic obstructive pulmonary disease, C.O.P.D., emphysema or chronic bronchitis? 1 yes 2 no
	# z.iloc[n].QSTLANG       =   2      # 1 english 2 spanish
	# z.iloc[n].ADDEPEV3      =   1      # (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)? 1 yes 2 no
	# z.iloc[n].CHCKDNY2      =   1      # Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?  1 yes 2 no
	# z.iloc[n].DIABETE4      =   1      # (Ever told) (you had) diabetes? 1 yes 2 no
	# z.iloc[n].MARITAL       =   2      # (marital status) 1 married 2 divorced 3 widowed 4 separated 5 never married 6 member of unmarried couple
	# z.iloc[n]._AGE80        =   70      # imputed age value collapsed above 80
	# z.iloc[n].HTM4      =       177        # height in centimeters
	# z.iloc[n].WTKG3     =       93        # weight in kilograms, implied 2 decimal places
	# z.iloc[n]._BMI5     =       int(z.iloc[n].WTKG3) / (int(z.iloc[n].HTM4) ^ 2)        # body mass index
	# z.iloc[n]._CHLDCNT      = 5        # number of children in household.
	# z.iloc[n]._DRNKWK1      = 12        # total number of alcoholic beverages consumed per week.
	# z.iloc[n].SLEPTIM1      = 4        # how many hours of sleep do you get in a 24-hour period?
	#
	#
	# z.isnull().values.any()
	# z.shape
	#
	# q = process(z)
	# q.shape
	#
	# # X_new = [[...], [...]]
	# y_new = model.predict(q)
	# print(y_new)

	pass







