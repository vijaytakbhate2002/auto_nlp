import os
LABELS = sorted(['Experience', 'Personal', 'Project'])

CLASSIFIER_FOLDER = "model_metadata\\trained_model"
VECTORIZER_FOLDER = "model_metadata\\vectorizer"
VECTORIZER_FILE = os.path.join(VECTORIZER_FOLDER, "vectorizer.pkl")
ENCODER_FOLDER = "model_metadata\\encoder"
ENCODER_FILE = os.path.join(ENCODER_FOLDER, "encoder.pkl")

DATA_READ_PATH = "train_nlp\\data\\question_category.csv"
DATA_FILE_TYPE = '.csv'
INPUT_COLUMN_NAME = "Question"
OUTPUT_COLUMN_NAME = "Category"