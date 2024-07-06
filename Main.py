import pandas as pd

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