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
    def __init__(self, target_col_name:str, input_col_name:str) -> None:
        self.input_col_name = input_col_name
        self.target_col_name = target_col_name

    def show(self):
        return self.df

    def fit(self, df:pd.DataFrame):
        self.df = df

    def fit_col_names(self, target_col_name:str, input_col_name:str):
        self.target_col_name = target_col_name
        self.input_col_name = input_col_name

class PrepareData(BaseOperations):
    def __init__(self, file_path:str, file_type:str, target_col_name:str, input_col_name:str) -> None:
        super().__init__(target_col_name=target_col_name, input_col_name=input_col_name)
        self.file_path = file_path
        self.file_type = file_type

    def readData(self) -> None:
        """ read csv file and store it in class objects df variable
            Return: None"""
        if self.file_type == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_type == '.xlsx':
            self.df = pd.read_excel(self.file_path)

class ProcessData(BaseOperations):
    LABELS = None
    def __init__(self, target_col_name:str, input_col_name:str) -> None:
        super().__init__(target_col_name=target_col_name, input_col_name=input_col_name)

    def dropNull(self) -> tuple[pd.Series]:
        """ drops null values from X and y pandas series 
            Args:
                df:pd.DataFrame
                
            Return:
                pd.DataFrame
        """
        self.df = self.df.dropna()
    
    def labelEncoder(self) -> Union[pd.Series, pd.DataFrame]:
        """ encode target column with LabelEncoder, replace old labels with new encodings
            
            Args: None
            Return pd.DataFrame
            """
        
        encoder = LabelEncoder()
        self.df[self.target_col_name] = encoder.fit_transform(y=self.df[self.target_col_name])
        self.LABELS = encoder.classes_
        folder_path = os.path.join('\\'.join(__file__.split('\\')[:-1]), nlp_config.ENCODER_FOLDER)
        if os.path.exists(folder_path) != True:
            os.makedirs(folder_path)
        joblib.dump(encoder, filename=os.path.join(folder_path, 'encoder.pkl'))
    
    def textProcessor(self) -> None:
        """ Filters the text, apply stemming and lemmatization """
        self.df[self.input_col_name] = self.df[self.input_col_name].apply(textProcess)

    def vectorizer(self, vectorizer_abbrivation:str, MIN_DF:float):
        """ uses vectorization techniques for converting text into numbers, there are two options
            count (count vectorizer) or tf-idf """
        vectorizer_obj = Vectorizer(MIN_DF=MIN_DF)
        self.X = vectorizer_obj.vectorize(X=self.df[self.input_col_name], 
                                          vectorizer_abbrivation=vectorizer_abbrivation)
        self.df = pd.concat([self.X, self.df[self.target_col_name]], axis='columns')

    def processData(self, vectorizer_abbrivation:str, MIN_DF:float) -> tuple:
        """ Drops null values, apply text processing (text filteration), vectorization and Label Encoding
            
            Args:
                vectorizer_abbrivation: str (choose from ('count', 'tf-idf'))
                MIN_DF: float (words to be neglected from document 
                        eg. 0.01 means word appeard 1% in document will be neglected)
                
            Return:
                tuple (X, y)
        """
        self.df.dropna()
        self.textProcessor()
        self.df.dropna()
        self.vectorizer(vectorizer_abbrivation, MIN_DF)
        self.labelEncoder()

class Analyze(BaseOperations):
    def __init__(self, target_col_name:str, labels:str, shuffle:bool=True, test_size:float=0.33):
        self.target_col_name = target_col_name
        self.labels = labels
        self.shuffle = shuffle
        self.test_size = test_size
        self.tester = Tester()

    def splitter(self):
        """ Splits data into X_train, X_test, y_train, y_test
            these variables are saved into object variables,
            to access X_train use self.X_train """
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([self.target_col_name], axis='columns'), 
                                                            self.df[self.target_col_name], shuffle=self.shuffle,
                                                            test_size=self.test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def confusionMatrix(self, y_true:pd.Series, y_pred:pd.Series) -> None:
        """ Plots confusion matrix """
        analyze = ModelPerformanceAnalyzer(results=None)
        print("y_true ", y_true)
        print("y_pred ", y_pred)
        analyze.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.labels)

    def analyzeModel(self, model_abbrivation:str) -> dict[dict]:
        """
            Analyze a specified model which is mentioned in model_bbrivation
            Args:
                y_true: pd.Series (Correct lables)
                y_pred: pd.Series (Predicted lables)
            Return:
                dict[dict] (model metrics)
        """
        tester = Tester()
        self.splitter()

        # result = tester.classificationMatrices(model_abbrivation, self.X_train, self.X_test, 
        #                                        self.y_train, self.y_test)

        result = tester.classificationMatrices(model_abbrivation=model_abbrivation, 
                                        X_train=self.X_train, X_test=self.X_test,
                                        y_train=self.y_train, y_test=self.y_test)
        compare(results=result, categories=self.labels)
        return result 

    def trainAndAnalyze(self, apply_all:bool=True, model_abbrivation:str=None) -> dict[dict]:
        """ 
            process data and analyze it with graphs
            Args:
                all: bool (if True, analysis applies on every algorithm from Tester class)
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf') in case single model analysis)
            Return: 
                dict[dict] (all model metrics)
            """

        tester = Tester(stratify=self.df[self.target_col_name], 
                        shuffle=self.shuffle, 
                        test_size=self.test_size)
        if apply_all:
            results = tester.testAllModels(model_abbrivation=model_abbrivation, 
                                           X=self.df.drop([self.target_col_name], axis='columns'), 
                                           y=self.df[self.target_col_name])
        else:
            # results = tester.testAModel(X=self.df.drop([self.target_col_name], axis='columns'), 
            #                             y=self.df[self.target_col_name],
            #                             model_abbrivation=model_abbrivation,
            #                             **params)
            self.splitter()
            results = tester.classificationMatrices(model_abbrivation=model_abbrivation, 
                                                    X_train=self.X_train, X_test=self.X_test,
                                                    y_train=self.y_train, y_test=self.y_test)
            results = {model_abbrivation:results}
        compare(results=results, categories=self.labels)
        return results

class HyperParameterTuner(BaseOperations):

    def __init__(self, scoring:str='f1_weighted') -> None:
        """ Args:
                model: str (choose from ('lr', 'dt', 'rf', 'nb'))
                scoring: str (by default f1_weighted)
        """
        self.scoring = scoring
        
    def fit(self, X:pd.DataFrame, y:pd.Series):
        self.X = X
        self.y = y

    def bestEstimator(self, model:str) -> dict:
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
        grid_search.fit(self.X, self.y)
        print("grid search done")
        best_estimator = grid_search.best_estimator_

        folder_path = os.path.join('\\'.join(__file__.split('\\')[:-1]), nlp_config.CLASSIFIER_FOLDER)
        if os.path.exists(folder_path) != True:
            os.makedirs(folder_path)
        joblib.dump(best_estimator, filename=os.path.join(folder_path, model + ".pkl"))
        result = {"best_estimator":grid_search.best_estimator_, "best_score":grid_search.best_score_,}
        return result
    
    def bestEstimatorSelector(self) -> dict[dict]:
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
            grid_search.fit(self.X, self.y)
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
