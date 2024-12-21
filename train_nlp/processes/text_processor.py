import re
from nltk import PorterStemmer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
import logging

warnings.filterwarnings('ignore')
nltk.download('wordnet', download_dir="E:\Projects\Profile\portfolio\processes\\nltk_data")
nltk.download('punkt', download_dir="E:\Projects\Profile\portfolio\processes\\nltk_data")
nltk.download('stopwords', download_dir="E:\Projects\Profile\portfolio\processes\\nltk_data")
nltk.download('omw-1.4', download_dir="E:\Projects\Profile\portfolio\processes\\nltk_data")  

lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

ps = PorterStemmer()

def stemmerAndLemmitization(text:str):
    tokens = list(text.split(' '))
    tokens = [ps.stem(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def textProcess(text):
    """ Filters the text with some regix expressions and apply stemming and lemmatization """
    try:
        logging.info(f"Enterd into textProcess with text = {text}")
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        tokens = stemmerAndLemmitization(text)
        processed_text = ' '.join(tokens)
        if len(processed_text) == 0:
            processed_text = None
        return processed_text

    except Exception as e:
        logging.warning(f"could not perform text processing for {text}")
        return None



# import re
# import spacy
# from string import punctuation

# # Load SpaCy's small English model
# nlp = spacy.load("en_core_web_sm")

# # Create a custom list of stop words
# stop_words = nlp.Defaults.stop_words

# def stemmer_and_lemmatization(text: str):
#     """Performs tokenization, removes stop words, and applies lemmatization."""
#     doc = nlp(text)
#     tokens = [
#         token.lemma_ for token in doc 
#         if token.text not in stop_words and token.text not in punctuation
#     ]
#     return tokens

# def textProcess(text: str):
#     """Filters the text using regex, and applies stemming and lemmatization."""
#     try:
#         text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#         text = re.sub(r'@\w+|#\w+', '', text)
#         text = re.sub(r'<.*?>', '', text)
#         text = re.sub(r'[^A-Za-z\s]', '', text)
#         text = text.lower()  

#         tokens = stemmer_and_lemmatization(text)
#         processed_text = ' '.join(tokens)
#         if len(processed_text) == 0:
#             processed_text = None
#         return processed_text

#     except Exception as e:
#         print(f"Could not process text: {text}. Error: {e}")
#         return None
