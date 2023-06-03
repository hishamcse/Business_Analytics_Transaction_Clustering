import numpy as np
import pandas as pd
import nltk
import re
import sys

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('words')

from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")

words = set(nltk.corpus.words.words())
from nltk.corpus import stopwords
st = set(stopwords.words("english"))

lemmatizer=WordNetLemmatizer()

from gensim.models import Word2Vec

import pickle

ignored_words = ['eftn', 'ft', 'bkash', 'nogod', 'rtgs', 'pos','cib', 'paywell', 'challan', 'npsb', 'dps', 'atm', 'trf', 'sonod']

# instead of recalculating NER_lists, we can just import it from the notebook
NER_lists = {'para', 'weekly', 'ba', 'saddik', 'oct', 'eighteen', 'ka', 'c', 'dor', 'na', 'gate', 'point', 'dal', 'feb', 'german', 'december', 'id', 'begum', 'ink', 'zero', 'mim', 'bas', 'tapu', 'orient', 'mo', 'abu', 'brother', 'tala', 'daud', 'new', 'type', 'title', 'outlet', 'jan', 'name', 'october', 'th', 'kokan', 'aug', 'currier', 'doll', 'u', 'august', 'service', 'tara', 'nov', 'tony', 'mar', 'bin', 'february', 'k', 'china', 'martin', 'st', 'jowel', 'x', 'sha', 'dada', 'today', 'noon', 'ad', 'yesterday', 'ae', 'amir', 'sweety', 'mother', 'mu', 'hanif', 'mullah', 'july', 'first', 'nova', 'japan', 'ge', 'rocky', 'rana', 'pu', 'annual', 'second', 'omer', 'bibi', 'fakir', 'southeast', 'da', 'cotton', 'apr', 'coxs', 'al', 'jun', 'sima', 'e', 'bakula', 'dola', 'pur', 'quarterly', 'amenia', 'shanghai', 'shahin', 'babu', 'ar', 'bu', 'tania', 'p', 'm', 'june', 'patwari', 'barman', 'dey', 'sir', 'daily', 'i', 'khan', 'raj', 'rani', 'week', 'boro', 'momo', 'sep', 'b', 'pally', 'sultana', 'fourteen', 'link', 'palli', 'ghat', 'chad', 'l', 'das', 'dec', 'mir', 'march', 'hour', 'sri', 'kaka', 'september', 'r', 'auto', 'nandi', 'month', 'amt', 'kazi', 'year', 'puja', 'hasan', 'november', 'amin', 'may', 'date', 'monthly', 'razor', 'sheik', 'road', 'gore', 'january', 'bari', 'nid', 'say', 'april', 'total', 'twelve', 'shah', 'sec', 'fifteen', 'doc', 'son', 'maria', 'jul', 'two'}


def preprocess(s):
  s = str(s)
  s = s.lower()
  s = re.sub('[^A-Za-z ]+', '', s)
  s = ' '.join(word.lower() for word in s.split() if word not in st)
  s = nltk.word_tokenize(s)
  s = [lemmatizer.lemmatize(word, 'v') for word in s]

  t = []
  for x in s:
    if 'withdraw' in x:
      x = 'withdraw'
    if 'deposit' in x:
      x = 'deposit'
    if (x in words or not x.isalpha() or x in ignored_words):
      if x not in NER_lists:
         t.append(x)

  return t

def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def preprocessVectorizeData(df_test, fieldName):
    test_narrations = df_test[fieldName]

    preprocessed = []
    for te in test_narrations:
      preprocessed.append(preprocess(te))

    gensim_model = pickle.load(open('models/gensim_model.sav', 'rb'))

    vectorized_test = vectorize(preprocessed, model=gensim_model)

    print(len(vectorized_test))
    print(len(vectorized_test[0]))

    return preprocessed, vectorized_test


def predict(df_test, fieldName, modelName, preprocessed, vectorized_test):
    if modelName == 'kmeans':
        loaded_model = pickle.load(open('models/kmeans_model.sav', 'rb'))
        most_freq = pickle.load(open('models/most_freq_kmeans.txt', 'rb'))
        cluster_str = 'kmeans_cluster'
        most_freq_str = 'kmeans_most_freq_keywords'
        result_csv = 'test_kmeans_result.csv'
    elif modelName == 'minibatch':
        loaded_model = pickle.load(open('models/minibatch_model.sav', 'rb'))
        most_freq = pickle.load(open('models/most_freq_minibatch.txt', 'rb'))
        cluster_str = 'minibatch_cluster'
        most_freq_str = 'minibatch_most_freq_keywords'
        result_csv = 'test_minibatch_result.csv'
    elif modelName == 'bisect':
        loaded_model = pickle.load(open('models/bisect_model.sav', 'rb'))
        most_freq = pickle.load(open('models/most_freq_bisect.txt', 'rb'))
        cluster_str = 'bisect_cluster'
        most_freq_str = 'bisect_most_freq_keywords'
        result_csv = 'test_bisect_result.csv'
    else:
        print('Invalid Model Name')
        return

    test_labels = loaded_model.predict(vectorized_test)
    print(test_labels)

    df_clusters_test = pd.DataFrame({
    "narrations": df_test[fieldName],
    "tokens": preprocessed,
     cluster_str: test_labels
    })

    df_clusters_test[most_freq_str] =  df_clusters_test.apply(lambda x: most_freq[x[cluster_str]], axis=1)

    df_clusters_test.to_csv(result_csv)

    print("please see complete result at ", result_csv, " file")


def processAndResultData(filePath, fieldName, modelName):
    df_test = pd.read_csv(filePath)
    df_test.head()

    preprocessed, vectorized_test = preprocessVectorizeData(df_test, fieldName)
    predict(df_test, fieldName, modelName, preprocessed, vectorized_test)


if __name__ == "__main__":
    # input file path, field name, model name from arguments
    filePath = sys.argv[1]
    fieldName = sys.argv[2]
    modelName = sys.argv[3]

    processAndResultData(filePath, fieldName, modelName)