import re
import pandas as pd
from nltk.corpus import stopwords
import emoji
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import nltk

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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
    # Extract and retain hashtags as separate tokens
    hashtags = re.findall(r'#\w+', text)
    # Remove hashtags from the main text
    text = re.sub(r'#\w+', '', text)
    # Remove dashes
    text = re.sub(r'-{1,}', '', text)
    # Join the text with hashtags
    if hashtags:
        text = ' '.join([text] + hashtags)
    return text

# Apply the cleanText function
Data['freshData'] = Data['clean_text'].apply(cleanText)
print("Fresh Data:\n", Data['freshData'].head())
