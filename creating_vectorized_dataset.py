import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import spacy
import numpy as np






# create set of stopwords
stop_words = set(stopwords.words('english'))

# initialise stemmer
stemmer = PorterStemmer()

# load the spacy word embedding model
nlp = spacy.load('en_core_web_sm')





# function used to turn airline sentiment from pos, neut, neg to 2, 1, 0
def airline_sentiment_numberifier(sentiment):

    if sentiment == "positive":
        return 2
    elif sentiment == "neutral":
        return 1
    elif sentiment == "negative":
        return 0
    else:
        raise ValueError("SENTIMENT IN DATASET WAS NEITHER POSTIIVE, NEUTRAL OR NEGATIVE")

# function to preprocess each airline review
def text_preprocessor(review):

    # make lower case
    review = review.lower()

    # remove punctuation
    review = re.sub(r'[^\w\s]', '', review)

    # tokenize and remove stopwords
    review = word_tokenize(review)
    review = [word for word in review if word not in stop_words]

    # remove first word in "text" - this is always the name of the airline
    review = review[1:]

    # remove any urls - remove any words that contain "http"
    review = [word for word in review if word[:4] != "http"]

    # apply stemming
    review = [stemmer.stem(token) for token in review]

    return review
    
# # function to create matrix of word embeddings per review - each word is across a row
def create_word_embeddings(review):

    list_of_word_embeddings = []

    for word in review:
        list_of_word_embeddings.append(nlp(word).vector)
    
    matrix_review_embedding = np.array(list_of_word_embeddings)

    return matrix_review_embedding






### 0. Load Data
dataset_file_path = r"C:\Users\kailf\OneDrive\Documents\2023_Summer\NLP_project\airline_reviews_dataset.xlsx"
dataset_df = pd.read_excel(dataset_file_path)

# # take first 100 rows
# dataset_df = dataset_df.iloc[:100]


# only keep useful columns
dataset_df = dataset_df[["airline_sentiment", "airline_sentiment_confidence", "retweet_count", "text"]]


### 1. Data Preprocessing

# exclude weird characters
special_chars_pattern = r'[^\x00-\x7F]+'
mask = dataset_df['text'].str.contains(special_chars_pattern)
dataset_df = dataset_df[~mask]


# duplicate based on retweet_count
df_of_nonzero_retweets_only = dataset_df[dataset_df['retweet_count'] > 0]
rows_to_repeat = [row for i, row in df_of_nonzero_retweets_only.iterrows() for i in range(row['retweet_count'])]
df_of_repeats = pd.DataFrame(rows_to_repeat, columns=df_of_nonzero_retweets_only.columns)
dataset_df = pd.concat([dataset_df, df_of_repeats])
dataset_df = dataset_df[["airline_sentiment", "airline_sentiment_confidence", "text"]]


# eliminate rows with confidence less than 0.6
mask = (dataset_df['airline_sentiment_confidence'] >= 0.6)
dataset_df = dataset_df[mask]
dataset_df.reset_index(drop=True, inplace=True)


# turn positive, neutral and negative into 2, 1, 0
dataset_df["airline_sentiment"] = dataset_df["airline_sentiment"].apply(airline_sentiment_numberifier)

# make lower case, remove punctuation, tokenize, remove stopwords, remove airline name, remove any urls, apply stemming
# remove any words not in the word embedding model
dataset_df["text"] = dataset_df["text"].apply(text_preprocessor)


# remove any row with reviews that are empty or longer than 10 words
dataset_df["text length"] = dataset_df["text"].apply(len)
dataset_df = dataset_df[dataset_df["text length"] > 0]
dataset_df = dataset_df[dataset_df["text length"] <= 10]
dataset_df.reset_index(drop=True, inplace=True)

# create word embeddings matrices
dataset_df["word embeddings"] = dataset_df["text"].apply(create_word_embeddings)



# write to excel file
dataset_df.to_excel(r"C:\Users\kailf\OneDrive\Documents\2023_Summer\NLP_project\NLPed_dataset2.xlsx", index=False, engine='openpyxl')
