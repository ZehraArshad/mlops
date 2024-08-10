import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformations_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            num_features =["reading_score","writing_score"]
            cat_features =["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))])
            logging.info('Numerical columns standard scaling has been completed')
            logging.info('Categorical columns encoding has been completed')

            preprocessor = ColumnTransformer([
                ('numerical_pipelines', num_pipeline, num_features),
                ('categorical_pipeline', cat_pipeline, cat_features) 
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("'Reading test and training data has been completed")
            processor_obj = self.get_data_transformer_object()
            target_column_name ="math_score"
            num_features =["reading_score","writing_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            input_feature_train_arr=processor_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=processor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)
                              ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'saved preprocessing obj')
            save_obj(
                file_path = self.data_transformations_config.preprocessor_obj_file_path,
                obj=processor_obj
            )
            return(
                train_arr, test_arr, self.data_transformations_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
