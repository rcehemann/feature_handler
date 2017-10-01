# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:10:52 2017

FEATURE HANDLER class for standardizing pandas dataframes

@author: blenderherad


FeatureHandler class for converting a pandas DataFrame systematically based on 
the class of feature. accepts numerical, categorical and rank class types.

Takes as input: pandas dataframe, dictionary containing a class type for each
feature, and keword arguments. 

kwargs:
    'scaling_method' : if 'robust', uses sklearn RobustScaler for normalizing 
                        numerical data. otherwise StandardScaler is used.
                        
    'rank_dict'      : REQUIRED if any feature is rank class. must be a nested dictionary
                        containing a dict for each ranking.
                        
                        e.g.: rank_dict = {'Badness' :{'bad': 0, 'worse': 1, 'worst': 2},
                                           'Goodness':{'good':0, 'great': 1, 'best':2}}
                        
                        where 'Badness' and 'Goodness' are exact feature names
                        and their dict keys are exact entries. will throw error if dict is incomplete

must be fit to training data by calling fit_transform()
can then be used to transform testing data by calling transform(test_frame)    
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from numpy import reshape, delete

class FeatureHandler:
    
    #%%
    # initialize to empty DataFrame and dict if called with no arguments
    # otherwise initialize to inputs
    def __init__(self, dataFrame = None, class_dict = None, **kwargs):

        self.kwargs = kwargs
        self.dataFrame  = pd.DataFrame() if dataFrame is None else dataFrame       
        assert isinstance(dataFrame, pd.DataFrame), "First argument should be pandas DataFrame"
        
        self.class_dict = {} if class_dict is None else class_dict
        assert isinstance(class_dict, dict), "Second argument should be dictionary of class types"
        
        self.initial_features = self.dataFrame.columns.values
        assert len(self.initial_features) == len(self.class_dict), "DataFrame and dictionary should be same length"
        
        self.is_fit = False
        
        return None
    
    #%%
    # public method to fit and transform dataFrame.
    def fit_transform(self):

        assert not self.is_fit, "feature handler has already been fit"
        
        # first split by class:
        self.__split_by_class()
        
        if len(self.numerical_columns) > 0:
            # fit and transform numerical columns
            self.__fit_transform_numerical()
            
        if len(self.categorical_columns) > 0:
            # fit and transform categorical columns
            self.__fit_transform_categorical()
            
        if len(self.rank_columns) > 0:        
            # transform rank columns
            self.__transform_rank()
        
        # recombine separated frames
        del self.dataFrame
        self.dataFrame = self.__recombine_frames()
        
        # update is_fit flag
        self.is_fit = True
                
        # return the transformed dataFrame
        return self.dataFrame
    

    #%%
    # public method for transforming a data frame input_frame
    def transform(self, input_frame):
        
        assert isinstance(input_frame, pd.DataFrame),              "Requires pandas dataframe with same features as initial frame"
        assert len(self.initial_features) == len(input_frame.columns), "Requires pandas dataframe with same features as initial frame"
        assert all(self.initial_features == input_frame.columns),  "Requires pandas dataframe with same features as initial frame"
        
        # first split by class:
        self.__split_on_input(input_frame)

        if len(self.numerical_columns) > 0:        
            # transform numerical columns
            self.__transform_numerical()

        if len(self.categorical_columns) > 0:        
            # transform categorical columns
            self.__transform_categorical()

        if len(self.rank_columns) > 0:        
            # transform rank columns
            self.__transform_rank()

        # recombine
        del input_frame
        input_frame = self.__recombine_frames()        

        return input_frame
        
    
    #%%
    # private method to split dataframe by feature class:
    # numerical (N), categorical (C), or rank (R). sets lists of features by class
    def __split_by_class(self):
        
        self.numerical_columns   = []
        self.categorical_columns = []
        self.rank_columns        = []
        for column in self.dataFrame.columns.values:
            c = self.class_dict[column][0].upper()
            if   c == 'N':
                self.numerical_columns.append(column)
            elif c == 'C':
                self.categorical_columns.append(column)
            elif c == 'R':
                self.rank_columns.append(column)
            else:
                raise Exception('Class ' + c + ' is invalid')
        
        # produce temporary dataframe for each class
        self.numerical_frame    = self.dataFrame.loc[:, self.numerical_columns]
        self.categorical_frame  = self.dataFrame.loc[:, self.categorical_columns]
        self.rank_frame         = self.dataFrame.loc[:, self.rank_columns]
        
        
        
        return self
    
        #%%
    # private method to split dataframe by feature class:
    # numerical (N), categorical (C), or rank (R). sets lists of features by class
    def __split_on_input(self, input_frame):
        
        # produce temporary dataframe for each class
        self.numerical_frame    = input_frame.loc[:, self.numerical_columns]
        self.categorical_frame  = input_frame.loc[:, self.categorical_columns]
        self.rank_frame         = input_frame.loc[:, self.rank_columns]
        
        return self
    
    #%%
    # private method for recombining frames. deletes temporary class frames
    def __recombine_frames(self):
        
        # concatenate frames, overwriting self.dataFrame
        frame = pd.concat([self.numerical_frame, self.categorical_frame, self.rank_frame], axis=1)
        
        # clean up temporary frames
        del self.numerical_frame
        del self.categorical_frame
        del self.rank_frame
        
        return frame   
        
    #%%
    # private method to normalize numerical columns
    def __fit_transform_numerical(self):
        
        # check if robust scaling
        if 'scaling_method' in self.kwargs and self.kwargs['scaling_method'].lower() == 'robust':
            self.numerical_scaler = RobustScaler()
        else:
            self.numerical_scaler = StandardScaler()
            
        self.numerical_frame = pd.DataFrame(self.numerical_scaler.fit_transform(self.numerical_frame), columns=self.numerical_frame.columns)
        
        return self
    
    #%%
    # private method for processing numerical without re-fitting 
    def __transform_numerical(self):

        self.numerical_frame = pd.DataFrame(self.numerical_scaler.transform(self.numerical_frame), columns=self.numerical_frame.columns)
    
        return self
    
    #%%
    # private method to process categorical data
    # one-hot encode each categorical feature, then drop one of the produced
    # columns to avoid collinearity of the new features. 
    def __fit_transform_categorical(self):
   
        # store encodings for later use on test set
        self.LEs = {}; self.OHEs = {}; self.cat_labels = {}
        for column in self.categorical_columns:
            
            # fit and store label encoder, then one-hot encoder.
            self.LEs[column]  = LabelEncoder()
            labeled_col = self.LEs[column].fit_transform(self.categorical_frame[column])
            self.cat_labels[column] = 1+max(labeled_col)

            self.OHEs[column] = OneHotEncoder(sparse=False, n_values=self.cat_labels[column])
            ohe_col     = self.OHEs[column].fit_transform([[l] for l in labeled_col])
            
            # reshape and remove last column
            ohe_col     = delete(reshape(ohe_col, (len(labeled_col), self.cat_labels[column])), -1, 1)

            # produce new column names
            newnames    = [column + str(c) for c in range(self.cat_labels[column]-1)]
            
            # drop original feature and merge one-hot features into categorical frame
            self.categorical_frame = pd.concat([self.categorical_frame, pd.DataFrame(ohe_col, columns=newnames)], axis=1)
            self.categorical_frame.drop(column, axis=1, inplace=True)

            
        return self
    
    #%%
    # private method for transforming categorical data without re-fitting encoders
    def __transform_categorical(self):
        
        for column in self.categorical_columns:

            # generate temp frame which clips any new classes to the LAST class, such that the column
            # is dropped upon conversion to one-hot encoding and we don't mis-label anything.
            temp_col    = self.categorical_frame[column].apply(lambda s:
                          self.LEs[column].classes_[-1] if s not in self.LEs[column].classes_ else s)
            labeled_col = self.LEs[column].transform(temp_col)
            ohe_col     = self.OHEs[column].transform([[l] for l in labeled_col])
            
            # reshape and remove last column
            ohe_col     = delete(reshape(ohe_col, (len(labeled_col), self.cat_labels[column])), -1, 1)
            
            # produce new column names
            newnames    = [column + str(c) for c in range(self.cat_labels[column]-1)]
            
            # drop original feature and merge one-hot features into categorical frame
            self.categorical_frame.drop(column, axis=1, inplace=True)
            self.categorical_frame = pd.concat([self.categorical_frame, pd.DataFrame(ohe_col, columns=newnames)], axis=1)
            
        return self
    
    #%%
    # private method to process rank data. requires NESTED DICTIONARY containing rankings for each ranked feature
    def __transform_rank(self):
        
        if len(self.rank_columns) > 0:
            assert 'rank_dict' in self.kwargs, "Use of ranked features requires a nested dictionary containing rankings for each"
                
        for column in self.rank_columns:

            self.rank_frame.loc[:,column] = self.rank_frame[column].apply(lambda x: float(self.kwargs['rank_dict'][column][x]))
            
        return self
    