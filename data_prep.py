import numpy as np
import nltk
import string
import random
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer      


################################################
# Preprocessing of dataset
################################################

# Read dataset file
df = pd.read_csv("C:\\Users\\ulian\\OneDrive\\project\\BooksDataset.csv")

# Choose needed columns
df = df[["Title", "Authors", "Description","Category"]]

# Lower all letters in needed columns
df[["Title", "Description", "Category"]] = df[["Title", "Description", "Category"]].applymap(lambda x: x.lower() if isinstance(x, str) else x)


nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')

# Make tokens out of sentences and phrases, make new columns with tokenized content
names_of_col = ["Description","Title", "Category"]
for column in names_of_col:
    df[f"{column}_Tokonized"] = df[column].apply(lambda x: word_tokenize(str(x)) if isinstance(x, (str, float)) else [])

# Bring words to their core basic form
    
lemmatizer = WordNetLemmatizer()
df["Description_Tokonized"] = df["Description_Tokonized"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
df["Category_Tokonized"] = df["Category_Tokonized"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# Create a column with all tokens for one row

df["Combined_Tokens"] = df.apply(lambda row: set(
    token for col in ["Description_Tokonized", "Category_Tokonized"]
    for token in row[col]
), axis=1)

# Save file
df.to_csv(r'C:\Users\ulian\FINAL PROJECT\myproject\final_dataset.csv', index=False)