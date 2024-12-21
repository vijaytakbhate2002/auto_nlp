import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ..nlp_config import nlp_config
import joblib
import os
import sys

class Vectorizer: 
    """ 
        Vectorizatoin process of dataframe with avoid numercal column function 
        MIN_DF: float (tuning parameter of vectorizers to avoid repetitive words from document
                suppose MIN_DF = 0.01 then words with 1% occurance in document will be avoided)

        """
    
    def __init__(self, MIN_DF:float=0.001) -> None:
        self.MIN_DF = MIN_DF

    def tf_idfVectorizer(self, X:pd.Series) -> pd.DataFrame:
        """ 
            Apply TF-IDF Vectorizer on given column of dataframe, 
            then call avoidNumerical (Avoids all numerical strings from specified dataframe column)
            
            Args: df (pandas dataframe)
            Return: pd.DataFrame
            """
        tf = TfidfVectorizer(min_df=self.MIN_DF)
        tf_df = tf.fit_transform(X)
        arr = tf_df.toarray()
        df = pd.DataFrame(arr)

        if os.path.exists(nlp_config.VECTORIZER_FOLDER) != True:
            os.makedirs(nlp_config.VECTORIZER_FOLDER)
        joblib.dump(tf, filename=nlp_config.VECTORIZER_FILE)

        return df
    
    def countVectorizer(self, X:pd.Series) -> pd.DataFrame:
        """ 
            Apply TF-IDF Vectorizer on given column of dataframe, 
                df: pd.DataFrame (dataframe to apply vectorization)
                colunm_name: str (column on which vectorization should apply)
            Return: pd.DataFrame is True, 
            then call avoidNumerical (Avoids all numerical strings from specified dataframe column)            
        """

        tf = CountVectorizer(min_df=self.MIN_DF)
        tf_df = tf.fit_transform(X)
        arr = tf_df.toarray()
        df = pd.DataFrame(arr)

        folder_path = os.path.join('\\'.join(__file__.split('\\')[:-2]), nlp_config.VECTORIZER_FOLDER)

        if os.path.exists(folder_path) != True:
            os.makedirs(folder_path)

        joblib.dump(tf, filename=os.path.join(folder_path, 'vectorizer.pkl'))
        
        return df
    
    def vectorize(self, X:pd.Series, vectorizer_abbrivation:str) -> pd.DataFrame:
        """ apply vectorization on given df and return
            Args:
                vectorizer_abbrivation: str (choose from ('tf-idf', 'count'))
                df: pd.DataFrame (dataframe to apply vectorization)
                colunm_name: str (column on which vectorization should apply)
            Return: pd.DataFrame
            """
        
        if vectorizer_abbrivation == 'tf-idf':
            return self.tf_idfVectorizer(X=X)
        elif vectorizer_abbrivation == 'count':
            return self.countVectorizer(X=X)
        
        raise ValueError("wrong vectorizer_abbrivation choose from ('tf-idf', 'count')")