import pandas as pd
from nltk.corpus import stopwords
DataSet = pd.read_csv('Twitter_Data.csv', index_col= False)
Data = pd.DataFrame(DataSet)
# Checking data if it empties or null
missing_values = (Data.isna().sum())
print(missing_values)
#  Identify rows with missing values ,
rows_with_missing = Data[Data.isna().any(axis=1)]
# Print rows with missing values
print(" \nRows with missing values: \n" , rows_with_missing)
# Now lets drop missing value  for further efficient process.
Data.dropna(inplace=True)
print("Missing value", (Data.isna().sum()))
# Let's Tokenize the Text first
def tokenize_text(text):
    return text.split()
# Now,Applying Tokenizing to  the text
Data['tokens'] = Data['clean_text'].apply(tokenize_text)
print("Split text" , Data['tokens'])
tokenizedData = pd.DataFrame(Data['tokens'])
# Lets clean the data removing unwanted, first lets count the unwanted commas and words
# Function to count unwanted commas
def unwanted_commas(dataframe):
    unwanted_commas_count = dataframe['tokens'].apply(lambda x: x.count(',')).sum()
    return unwanted_commas_count
# Count unwanted commas
un = unwanted_commas(tokenizedData)
print("Sum of commas:", un)