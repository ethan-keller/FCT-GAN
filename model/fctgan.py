"""
Generative model training algorithm based on the FCTGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.fctgan_synthesizer import FCTGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class FCTGAN():
    def __init__(self,
                 dataset = "",
                 raw_csv_path = "",
                 test_ratio = 0.20,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {},
                 integer_columns = [],
                 problem_type= {None: None},
                 epochs=300):

        self.__name__ = 'FCTGAN'
        self.synthesizer = FCTGANSynthesizer(epochs=epochs, dataset=dataset)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        
    def fit(self):
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
        mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.raw_df)) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
