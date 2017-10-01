# feature_handler
#
#	standardizes an input pandas dataframe by:
	 - normalizing numerical features
	 - one-hot encoding categorical features and 
	 	dropping one column to avoid collinearity
	 - converting ranked string features to integer values
	 	based on an input dictionary

# use:

	from FeatureHandler import FeatureHandler
	import pandas as pd

	train_frame = pd.DataFrame(...)
	test_frame  = pd.DataFrame(...)

	class_dict = {'feature1':'numerical, 'feature2':'categorical', 'feature3':'rank', ...}
	rank_dict  = {'feature3':{'f31':0, 'f32':1, 'f33':2, ...}

	fh 	    = FeatureHandler(train_frame, class_dict, rank_dict = rank_dict)
	train_frame = fh.fit_transform()	# fit encodings to training data
	test_frame  = fh.transform(test_frame)	# transform testing data

