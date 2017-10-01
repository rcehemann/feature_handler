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
	fh 	    = FeatureHandler(train_frame)
	train_frame = fh.fit_transform()	# fit encodings to training data
	test_frame  = fh.transform(test_frame)	# transform testing data

