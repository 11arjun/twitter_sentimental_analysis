import numpy as np
import ssl
import pandas as pd
from nltk.corpus import stopwords
import emoji
import nltk
import contractions
import re
from textblob import TextBlob
from transformers import RobertaTokenizer
#  Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Loading  DataSet
Datas = pd.read_csv('Twitter_Data.csv', index_col=False)
# Taking .1 % datas from the set
data = Datas.sample(frac=0.001, random_state=42)
pd.set_option('display.max_colwidth', None) # Shows full column content
# Now lets drop missing value for further efficient process.
data.dropna(inplace=True)

# Let's create a  function to clean the data
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
    return text

# Applying cleaning Function
data['clean_data'] = data['clean_text'].apply(cleanText)

# Loading the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Let's create a function to tokenize a data
def tokenize_text(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=512)

# Let's apply the tokenization function
data['tokenized'] = data['clean_data'].apply( lambda  x: tokenize_text(x))
# Let's extract it to view the Id's and attention mask
def Extratcions(tokenized):
    return tokenized['input_ids']

data['input_ids'] = data['tokenized'] .apply(lambda x: x['input_ids'])
data['attention_mask'] = data['tokenized'].apply(lambda x: x['attention_mask'])

# Print Tokenization
# print(" Tokenize text\n ", data['clean_data'].head() ,'\n', data['tokenized'].head())
print("Extractions Id's" , data[['input_ids', 'attention_mask']].head())
