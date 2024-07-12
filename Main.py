import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from transformers import RobertaTokenizer

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Loading  DataSet
data = pd.read_csv('Twitter_Data.csv', index_col=False)
pd.set_option('display.max_colwidth', None) # Shows full column content
# Now lets drop missing value for further efficient process.
data.dropna(inplace=True)


# Let's create a  function to clean the data
def cleanData(clean):
    # remove numbers
    clean = re.sub(r'\d+', '', clean)
    # remove punctuation
    clean = re.sub(r'[^\w\s]', '', clean)
    # remove extra spaces
    clean = clean.strip()
    # Remove stop words
    clean = ' ' .join([word for word in clean.split() if word.lower() not in stop_words])
    return clean

# Applying cleaning Function
data['clean_data'] = data['clean_text'].apply(cleanData)

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
print(" Tokenize text\n ", data['clean_data'].head() ,'\n', data['tokenized'].head())
print("Extractions Id's" , data[['input_ids', 'attention_mask']].head())