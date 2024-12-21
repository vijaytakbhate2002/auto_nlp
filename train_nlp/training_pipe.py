import os
import joblib
from .processes.text_processor import textProcess
from .processes.vectorizer import Vectorizer
from .processes.model_tester import Tester
from .processes.trainer import Trainer
import pandas as pd
from sklearn.base import ClassifierMixin
from .analysis.analysis import compare, ModelPerformanceAnalyzer
from sklearn.model_selection import GridSearchCV, train_test_split
from .processes.hyperparameters import models_with_params
from sklearn.preprocessing import LabelEncoder
from typing import Union
from .nlp_config import nlp_config

class BaseOperations():
    def show(self):
        return self.df

class PrepareData(BaseOperations):
    def __init__(self, file_path:str, file_type:str, target_col_name:str, input_col_name:str) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.target_col_name = target_col_name
        self.input_col_name = input_col_name

    def readData(self) -> None:
        """ read csv file and store it in class objects df variable
            Return: None"""
        if self.file_type == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_type == '.xlsx':
            self.df = pd.read_excel(self.file_path)

    def splitData(self) -> tuple:
        """ Splits data into X and y
            Return: tuple"""
        self.X = self.df[self.input_col_name]
        self.y = self.df[self.target_col_name]
        return self.X, self.y


class ProcessData:

    def __init__(self, vectorizer_abbrivation:str, MIN_DF:float=0.01) -> None:
        
        self.vectorizer_abbrivation=vectorizer_abbrivation
        self.MIN_DF = MIN_DF

    def fit(self, X:pd.Series, y:pd.Series) -> None:
        self.X = X
        self.y = y

    def dropNull(self) -> tuple[pd.Series]:
        """ drops null values from X and y pandas series 
            Args:
                X: pd.Series 
                y: pd.Series
                
            Return:
                tuple[pd.Series]
        """
        self.X.name = 'input'
        self.y.name = 'output'
        df = pd.concat([self.X, self.y], axis='columns')
        df.dropna(inplace=True)
        self.X = df['input']
        self.y = df['output']
    
    def labelEncoder(self) -> Union[pd.Series, pd.DataFrame]:
        """ encode target column with LabelEncoder, replace old labels with new encodings
            
            Args: None
            Return pd.DataFrame
            """
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(y=self.y)

        folder_path = os.path.join('\\'.join(__file__.split('\\')[:-1]), nlp_config.ENCODER_FOLDER)
        if os.path.exists(folder_path) != True:
            os.makedirs(folder_path)
        joblib.dump(encoder, filename=os.path.join(folder_path, 'encoder.pkl'))
        return self.y
    
    def processData(self) -> tuple:
        """ Drops null values, apply text processing (text filteration), vectorization and Label Encoding
            
            Args:
                vectorizer_abbrivation: str (choose from ('count', 'tf-idf'))
                MIN_DF: float (words to be neglected from document 
                        eg. 0.01 means word appeard 1% in document will be neglected)
                
            Return:
                tuple (X, y)
        """
        self.dropNull()
        self.X = self.X.apply(lambda x: textProcess(x))
        self.dropNull()
        vectorizer_obj = Vectorizer(MIN_DF=self.MIN_DF)
        self.X = vectorizer_obj.vectorize(X=self.X, vectorizer_abbrivation=self.vectorizer_abbrivation)
        self.labelEncoder()
        print(f"After vectorization: X = {self.X.shape}, y = {self.y.shape}")
        return self.X, self.y
    

class Analyze:

    def __init__(self, shuffle:bool=True, test_size:float=0.33):
        self.shuffle = shuffle
        self.test_size = test_size
        self.tester = Tester()

    def fit(self, X:pd.DataFrame, y:pd.Series) -> None:
        """ create class X and y class variable to access it inot other functions 
            Return: None"""
        self.X = X
        self.y =y 

    def confusionMatrix(self, y_true:pd.Series, y_pred:pd.Series) -> None:
        """ Plots confusion matrix """
        analyze = ModelPerformanceAnalyzer(results=None)
        print("y_true ", y_true)
        print("y_pred ", y_pred)
        analyze.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=nlp_config.LABELS)

    def analyzeModel(self, y_true:pd.Series, y_pred:pd.Series, model_abbrivation:str) -> dict[dict]:
        """
            Analyze a specified model which is mentioned in model_bbrivation
            Args:
                y_true: pd.Series (Correct lables)
                y_pred: pd.Series (Predicted lables)
            Return:
                dict[dict] (model metrics)
        """
        tester = Tester()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        result = tester.classificationMatrices(model_abbrivation, X_train, X_test, y_train, y_test)
        result = dict({model_abbrivation:result})
        compare(results=result, categories=nlp_config.LABELS)
        return result 

    def trainAndAnalyze(self, all:bool=True, model_abbrivation:str=None, **params) -> dict[dict]:
        """ 
            process data and analyze it with graphs
            Args:
                all: bool (if True, analysis applies on every algorithm from Tester class)
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf') in case single model analysis)
            Return: 
                dict[dict] (all model metrics)
            """
        
        tester = Tester(stratify=self.y, 
                        shuffle=self.shuffle, 
                        test_size=self.test_size)
        if all:
            print(self.X, self.y)
            results = tester.testAllModels(X=self.X, y=self.y)
        else:
            results = tester.testAModel(X=self.X, y=self.y, 
                                        model_abbrivation=model_abbrivation,
                                        **params)
        compare(results=results, categories=nlp_config.LABELS)
        return results


class HyperParameterTuner:

    def __init__(self, model:str, scoring:str='f1_weighted') -> None:
        """ Args:
                model: str (choose from ('lr', 'dt', 'rf', 'nb'))
                scoring: str (by default f1_weighted)
        """
        self.model = model
        self.scoring = scoring
        
    def bestEstimator(self, X:pd.DataFrame, y:pd.Series, model:str) -> dict:
        """ Try all possible combinations of parameters and find best version of calssifier
        
            Args:
                X: pd.DataFrame (input data)
                y: pd.Series (target data) 
                model: str (choose from ('lr', 'dt', 'rf', 'nb'))
                
            Return:
                dict
        """

        grid_search = GridSearchCV(estimator=models_with_params[model][0], param_grid=models_with_params[model][1], 
                                   scoring=self.scoring, cv=3)
        print("searching best parameters...")
        grid_search.fit(X, y)
        print("grid search done")
        best_estimator = grid_search.best_estimator_

        folder_path = os.path.join('\\'.join(__file__.split('\\')[:-1]), nlp_config.CLASSIFIER_FOLDER)
        if os.path.exists(folder_path) != True:
            os.makedirs(folder_path)
        joblib.dump(best_estimator, filename=os.path.join(folder_path, model + ".pkl"))
        result = {"best_estimator":grid_search.best_estimator_, "best_score":grid_search.best_score_,}
        return result
    
    def bestEstimatorSelector(self, X:pd.DataFrame, y:pd.Series) -> dict[dict]:
        """ Try all possible combinations of parameters on every algorithm
        
            Args:
                X: pd.DataFrame (input data)
                y: pd.Series (target data)
                
            Return:
                dict
            """
        result = {}
        best_score = 0
        best_estimator = None
        best_model_name = None

        for key, val in models_with_params.items():
            print(f"Searching for model {val[2]} ...")
            grid_search = GridSearchCV(estimator=val[0], param_grid=models_with_params[key][1], 
                                    scoring='f1_weighted', cv=3)
            grid_search.fit(X, y)
            result[val[2]] = {
                              'best_estimator':grid_search.best_estimator_, 
                              'best_score':grid_search.best_score_
                              }
            if grid_search.best_score_ > best_score:
                print("best score = ", best_score, "grid searched score = ", grid_search.best_score_)
                best_score = grid_search.best_score_
                best_estimator = grid_search.best_estimator_
                best_model_name = val[2]

        if os.path.exists(nlp_config.CLASSIFIER_FOLDER) == False:
                os.mkdir(nlp_config.CLASSIFIER_FOLDER)    
        joblib.dump(best_estimator, nlp_config.CLASSIFIER_FOLDER + "\\" + best_model_name + ".pkl")
        return result

class TraineProcess:

    def trainAmodel(self, X:pd.DataFrame, y:pd.Series, model_abbrivation:str, **params) -> ClassifierMixin:
        
        """ Process data and train a model
            Args:   
                X: pd.DataFrame
                y: pd.Series
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf'))
            Return: ClassifierMixin"""

        trainer = Trainer()
        trainer.fit(X_train=X, y_train=y)
        model = trainer.trainAmodel(model_abbrivation=model_abbrivation, **params)
        return model



if __name__ == "__main__":

    prepare = PrepareData(file_path=nlp_config.DATA_READ_PATH, 
                          file_type=nlp_config.DATA_FILE_TYPE, 
                          target_col_name=nlp_config.OUTPUT_COLUMN_NAME, 
                          input_col_name=nlp_config.INPUT_COLUMN_NAME)
    prepare.readData()
    X, y = prepare.splitData()
    processor = ProcessData(vectorizer_abbrivation='count', 
                            MIN_DF=0.01)
    processor.fit(X=X, y=y)
    X, y = processor.processData()

    analyzer = Analyze()
    trainer = TraineProcess()   
    analyzer.fit(X=X, y=y)
    res = analyzer.trainAndAnalyze()
    print(res)

    model_abbrivation = input("Enter model abbrivation to train model..\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                        shuffle=True, random_state=42)
    model = trainer.trainAmodel(X=X_train, y=y_train, model_abbrivation=model_abbrivation)
    y_pred = model.predict(X_test)
    analyzer.confusionMatrix(y_true=y_test, y_pred=y_pred)
    result = analyzer.analyze(y_true=y_test, y_pred=y_pred, 
                              model_abbrivation=model_abbrivation)
    print(result)

    grid_searcher = HyperParameterTuner(model=model_abbrivation, scoring='accuracy')
    result = grid_searcher.bestEstimator(X=X, y=y, model=model_abbrivation)
    print(result)

    command = input("Do you wnat to apply grid search on all available models, (it will take some time)?, if you want press Y otherwise n")
    if command.lower() == 'y':
        result = grid_searcher.bestEstimatorSelector(X=X, y=y)
