from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import ClassifierMixin
import numpy as np
import pandas as pd
from typing import Union

class Trainer:
    """ fit data with fit method and train model as per need
    """

    def fit(self, X_train:Union[np.ndarray, pd.DataFrame], 
                 y_train:Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """ fit data into class and return None"""
        
        self.X_train = X_train
        self.y_train = y_train

    def logisticRegression(self, **params) -> ClassifierMixin:
        """ Train LogisticRegression model with given params
            
            Args: **params (model parameters need to pass while training model)
            Return: ClassifierMixin (A trained model)
            """
        model = LogisticRegression(**params)
        model.fit(self.X_train, self.y_train)
        return model
    
    def randomForestClassifer(self, **params) -> ClassifierMixin:
        """ Train RandomForestClassifier model with given params
            
            Args: **params (model parameters need to pass while training model)
            Return: ClassifierMixin (A trained model)
            """
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        return model
    
    def decisionTreeClassifier(self, **params) -> ClassifierMixin:
        """ Train DecisionTreeClassifier model with given params
            
            Args: **params (model parameters need to pass while training model)
            Return: ClassifierMixin (A trained model)
            """
        model = DecisionTreeClassifier(**params)
        model.fit(self.X_train, self.y_train)
        return model
    
    def multinomialNB(self, **params) -> ClassifierMixin:
        """ Train MultinomialNB model with given params
            
            Args: **params (model parameters need to pass while training model)
            Return: ClassifierMixin (A trained model)
            """
        model = MultinomialNB(**params)
        model.fit(self.X_train, self.y_train)
        return model
    
    def trainAmodel(self, model_abbrivation:str, **params) -> ClassifierMixin:
        """ train a specified model with given parameters 
            Args:
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf'))

            Return:
                ClassifierMixin
                """
        if model_abbrivation == 'lr':
            model = self.logisticRegression(**params)
            return model
        
        if model_abbrivation == 'dt':
            model = self.decisionTreeClassifier(**params)
            return model
        
        if model_abbrivation == 'nb':
            model = self.multinomialNB(**params)
            return model
        
        if model_abbrivation == 'rf':
            model = self.randomForestClassifer(**params)
            return model
        raise ValueError("model_abbrivation is wrong choose from ('lr', 'dt', 'nb', 'rf')")

if __name__ == "__main__":
    print(dir)