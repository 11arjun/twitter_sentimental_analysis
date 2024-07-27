import re
import pandas as pd
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import emoji
import nltk
# Download the stopwords from NLTK
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # POS tagger

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load DataSet
DataSet = pd.read_csv('Twitter_Data.csv', index_col=False)
Data = pd.DataFrame(DataSet)

# Let's check missing value, it gives you the sum of missing values
missing_values = Data.isna().sum()
print("Missing Values:", missing_values)

# Let's drop if we have any missing values, dropna automatically removes all the missing values data
Data = Data.dropna()

# Let's lower case every letter
Data['clean_text'] = Data['clean_text'].str.casefold()

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
    return text
# Apply the cleanText function
Data['freshData'] = Data['clean_text'].apply(cleanText)
print("Fresh Data:\n", Data['freshData'].head())
# Now lets Apply tokenizing,stemming,lemmatizing and part of speech at once
def preProcess(data):
    # Tokenizing
    # data = word_tokenize(data)
    # Stemming
    data = [stemmer.stem(word) for word in data]
    # Lemmatization
    data = [lemmatizer.lemmatize(word) for word in data]
    # POS Tagging
    data = pos_tag(data)
    return data
Data['preProcess'] = Data['freshData'].apply(preProcess)
print(" New Process Data", Data['preProcess'].head())

