import re
import ssl

import numpy as np
import pandas as pd
from nltk import PorterStemmer, WordNetLemmatizer, SnowballStemmer, tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import emoji
import nltk
import contractions
from gensim.models import Word2Vec
from textblob import TextBlob

#  Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Download the stopwords from NLTK
print("Downloading NLTK data...")
nltk.download('punkt', force=True)
nltk.download('stopwords' , force=True)
nltk.download('wordnet' , force=True)
nltk.download('averaged_perceptron_tagger')  # POS tagger

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Load DataSet
DataSet = pd.read_csv('Twitter_Data.csv', index_col=False)
Datas = pd.DataFrame(DataSet)
# Taking .5 % datas from the set
Data = Datas.sample(frac=0.005, random_state=42)
# Let's check missing value, it gives you the sum of missing values
missing_values = Data.isna().sum()
print("Missing Values:", missing_values)

# Let's drop if we have any missing values, dropna automatically removes all the missing values data
Data = Data.dropna()

# Let's lower case every letter
Data['clean_text'] = Data['clean_text'].str.casefold()
# Helper function to convert POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Text Processing
# Let's make a function that handles unwanted words and all
def cleanText(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove punctuations except for hashtags
    text = re.sub(r'[^\w\s#]', '', text)
    # Remove extra spaces
    text = text.strip()
    # Now lets use stop words to remove
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    # Define a regex pattern to match web links (http, https, www)
    url_pattern = r'http[s]?://\S+|www\.\S+'
    # Define a regex pattern to match HTML links
    html_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1'
    # Remove web links
    text = re.sub(url_pattern, '', text)
    # Remove HTML links
    text = re.sub(html_pattern, '', text)
    # Let' remove hashtags but keep the words
    text = re.sub(r'#(\w+)',r'\1', text)
    # Remove dashes
    text = re.sub(r'-{1,}', '', text)
    # Let's use contractions
    text = contractions.fix(text)
    # Let's correct the spelling using Textblob
    text = str(TextBlob(text).correct())
    # Let's Tokenize
    tokenize = word_tokenize(text)
    # Let' perform stemming now, the stemmed tokens are joined back into a single string.
    stemmed = [stemmer.stem(word) for word in tokenize]
    # Let's perform pos tagging now
    pos_tags =pos_tag(stemmed)
    # Let's perform Lemmatizations now
    lemma = [lemmatizer.lemmatize(words, get_wordnet_pos(tag)) for words, tag in pos_tags]
    return ' '. join(lemma)

# Apply the cleanText function
Data['freshData'] = Data['clean_text'].apply(cleanText)
print("Fresh Data:\n", Data['freshData'].head(10))

# Let's apply WordVec Algorithms now
# Let's Train the WordVec model
word2vec_model = Word2Vec(sentences= Data['freshData'], vector_size=100,window=5, min_count= 2, workers=4)
# Lets save the model , so we can train using Deep Learning Models
model_path = "word2vec_model"
word2vec_model.save(model_path)
import ssl

import numpy as np
import pandas as pd
from nltk import PorterStemmer, WordNetLemmatizer, SnowballStemmer, tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import emoji
import nltk
import contractions
from gensim.models import Word2Vec
from textblob import TextBlob

#  Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Download the stopwords from NLTK
print("Downloading NLTK data...")
nltk.download('punkt', force=True)
nltk.download('stopwords' , force=True)
nltk.download('wordnet' , force=True)
nltk.download('averaged_perceptron_tagger')  # POS tagger

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Load DataSet
DataSet = pd.read_csv('Twitter_Data.csv', index_col=False)
Datas = pd.DataFrame(DataSet)
# Taking .5 % datas from the set
Data = Datas.sample(frac=0.005, random_state=42)
# Let's check missing value, it gives you the sum of missing values
missing_values = Data.isna().sum()
print("Missing Values:", missing_values)

# Let's drop if we have any missing values, dropna automatically removes all the missing values data
Data = Data.dropna()

# Let's lower case every letter
Data['clean_text'] = Data['clean_text'].str.casefold()
# Helper function to convert POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Text Processing
# Let's make a function that handles unwanted words and all
def cleanText(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove punctuations except for hashtags
    text = re.sub(r'[^\w\s#]', '', text)
    # Remove extra spaces
    text = text.strip()
    # Now lets use stop words to remove
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    # Define a regex pattern to match web links (http, https, www)
    url_pattern = r'http[s]?://\S+|www\.\S+'
    # Define a regex pattern to match HTML links
    html_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1'
    # Remove web links
    text = re.sub(url_pattern, '', text)
    # Remove HTML links
    text = re.sub(html_pattern, '', text)
    # Let' remove hashtags but keep the words
    text = re.sub(r'#(\w+)',r'\1', text)
    # Remove dashes
    text = re.sub(r'-{1,}', '', text)
    # Let's use contractions
    text = contractions.fix(text)
    # Let's correct the spelling using Textblob
    text = str(TextBlob(text).correct())
    # Let's Tokenize
    tokenize = word_tokenize(text)
    # Let' perform stemming now, the stemmed tokens are joined back into a single string.
    stemmed = [stemmer.stem(word) for word in tokenize]
    # Let's perform pos tagging now
    pos_tags =pos_tag(stemmed)
    # Let's perform Lemmatizations now
    lemma = [lemmatizer.lemmatize(words, get_wordnet_pos(tag)) for words, tag in pos_tags]
    return ' '. join(lemma)

# Apply the cleanText function
Data['freshData'] = Data['clean_text'].apply(cleanText)
print("Fresh Data:\n", Data['freshData'].head(10))

# Let's apply WordVec Algorithms now
# Let's Train the WordVec model
word2vec_model = Word2Vec(sentences= Data['freshData'], vector_size=100,window=5, min_count= 2, workers=4)
# Lets save the model , so we can train using Deep Learning Models
model_path = "word2vec_model"
word2vec_model.save(model_path)
# Let's load the model now
loadModel = Word2Vec.load(model_path)
# Let's get the vector of a word now
def getVectorization(cleanData, model):
    wordVectors = [model.wv[word] for word in cleanData if word in model.wv]
    if len (wordVectors) > 0:
        return np.mean(wordVectors, axis = 0) # when axis is 0 it computes whole column, if 1 then whole row
    else:
        return  np.zeros(model.vector_size)
# Let's apply the vectorization function to all sentences
Data['sentences_vectorize'] = Data['freshData'].apply(lambda x: getVectorization(x,loadModel))
print("sentences vectors:\n", Data['sentences_vectorize'].head())
# Let's load the model now
loadModel = Word2Vec.load(model_path)
# Let's get the vector of a word now
def getVectorization(cleanData, model):
    wordVectors = [model.wv[word] for word in cleanData if word in model.wv]
    if len (wordVectors) > 0:
        return np.mean(wordVectors, axis = 0) # when axis is 0 it computes whole column, if 1 then whole row
    else:
        return  np.zeros(model.vector_size)
# Let's apply the vectorization function to all sentences
Data['sentences_vectorize'] = Data['freshData'].apply(lambda x: getVectorization(x,loadModel))
print("sentences vectors:\n", Data['sentences_vectorize'].head())