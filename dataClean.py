import re
import pandas as pd
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import nltk
import ssl

# Configure SSL context to bypass SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Load DataSet
DataSet = pd.read_csv('Twitter_Data.csv', index_col= False)
Data = pd.DataFrame(DataSet)
# Checking data if it empties or null
missing_values = (Data.isna().sum())
print(missing_values)
#  Identify rows with missing values ,
rows_with_missing = Data[Data.isna().any(axis=1)]
# Print rows with missing values
print(" \nRows with missing values: \n" , rows_with_missing)
# Now lets drop missing value for further efficient process.
Data.dropna(inplace=True)
print("Missing value", (Data.isna().sum()))
# A Function is build to remove the numbers , punctuation and unwanted spaces and slashes,stop words
def clean_Data(clean):
    # remove numbers
    clean = re.sub(r'\d+', '', clean)
    # remove punctuation
    clean = re.sub(r'[^\w\s]','', clean)
    # remove extra spaces
    clean = clean.strip()
    return clean
# Applying the cleaning function to each list of tokens
Data['cleanData'] = Data['clean_text'].apply(clean_Data)
print("\nCleaned Text:\n", Data['cleanData'].head())
# Let's Tokenize Datas now
def tokenize_text(text):
    return text.split()
# Now,Applying Tokenizing to  the Cleaned Data
Data['tokens'] = Data['cleanData'].apply(tokenize_text)
# Function to remove stop words
def removeStopWords(tokenText):
    return[token for  token in tokenText  if token.lower() not in stop_words]
# Applying the stop words now
Data['freshData'] = Data['tokens'].apply(removeStopWords)
# flattening the tokens back to a single string ,join method joins each token
Data['cleaned_text_final'] = Data['freshData'].apply(lambda tokens: ' '.join(tokens))
# Printing the final Freshed Cleaned Data now
print("\nFinal cleaned text:\n", Data[['cleaned_text_final']].head())
Data['cleaned_text_final'].to_csv('twitter_fresh_data.csv', index= False)
print("\n  New Fresh Datset is ready \n")


