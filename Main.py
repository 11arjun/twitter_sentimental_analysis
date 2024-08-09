import numpy as np
import ssl
import pandas as pd
from nltk.corpus import stopwords
import emoji
import nltk
import contractions
import re
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from transformers import RobertaTokenizer
from transformers import TFRobertaModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
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
def get_Sentiments(text):
    analysis = TextBlob(text)
    # Let's Determine the  sentiment Polarity
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'
# Apply Sentiments Labeling
data['sentiment'] = data['clean_data'].apply(get_Sentiments)
print(data[['clean_data','sentiment']].head())

# Let's Encode the sentiments.
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['sentiment'])
# Now, Let's split the data set , Training and Testing dataSet
train_data , test_data = train_test_split(data,test_size=0.2, random_state=42)
# Let's Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Let's create a function to tokenize a data
def tokenize_text(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=512)
# Let's apply the tokenization function
train_data['tokenized'] = train_data['clean_data'.apply(lambda x: tokenize_text(x))]
test_data['tokenized'] = test_data['clean_data'].apply(lambda x : tokenize_text(x))
# Let's Extract input ids and Attention mask from the Tokenized Data
train_data['input_ids'] = train_data['tokenized'].apply(lambda x: x['input_ids'])
train_data['attention_mask'] = train_data['tokenized'].apply(lambda x: x ['attention_mask'])

test_data['input_ids'] = test_data['tokenized'].apply(lambda x: x['input_ids'])
test_data['attention_mask'] = test_data['tokenized'].apply(lambda x: x ['attention_mask'])
# Loading the pre-train model  Roberta Model , using to understand the meaning of the sentences(Vectorization)
roberta_model = TFRobertaModel.from_pretrained('roberta-base')
# Defining The input Layers
input_ids_layer =  Input(shape=(512), dtype= 'int32', name = 'input_ids')
attention_mask_layer = Input(shape=(512), dtype = 'int32', name = 'attention_mask')
# Let's Get the Roberta Embeddings now
roberta_embeddings = roberta_model(input_ids_layer,attention_mask_layer)[0]  
# Lets Add  LSTM Layer for further Processing now