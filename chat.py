import ast
import numpy as np
import nltk
import string
import random
import pandas as pd


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer      

# Read dataset file
df = pd.read_csv(r'C:\Users\ulian\FINAL PROJECT\myproject\final_dataset_2.csv')
df['Combined_Tokens'] = df['Combined_Tokens'].apply(ast.literal_eval)

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')


#################################################
# Prepering user input
#################################################


custom_stop_words = [
    'something', 'maybe', 'book', 'sth', 'want', 'read', 'reading', 'novel', 'story', 'author', 'plot',
    'recommend', 'recommendation', 'looking', 'looking for', 'searching', 'searching for', 'find', 'finding',
    'finding for', 'good', 'interesting', 'enjoyable', 'great', 'like', 'love', 'prefer', 'preferable', 'best',
    'favorite', 'top', 'new', 'old', 'must', 'must-read', 'page-turner',
    'adult', 'hardcover', 'paperback', 'ebook', 'audiobook', 'kindle', 'library', 'shelf', 'cover', 'page',
    'chapter', 'verse', 'paragraph', 'sentence', 'word', 'line', 'ending', 'beginning', 'middle', 'climax',
    'resolution', 'twist', 'turn', 'predictable', 'surprising', 'unique', 'different', "me", "anything", "everything", "all", "better", 'only', "yet", "genre", "wanna", "proper"
]


lemmer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

# Lemmatizing user's tokens and removing stop words
def LemTokens(tokens):
   
    lem_tokens = [lemmer.lemmatize(token) for token in tokens]
   
    all_stop_words = set(stopwords.words('english') + custom_stop_words)
    filtered_tokens = [word for word in lem_tokens if word.lower() not in all_stop_words]
    
    return filtered_tokens

# Remove punctuation
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

# Tokenize input, bring text to lower
# Send to remove_punct_dict function and to LemTokens function, return the result from it with ready tokens

def LemNormalize(text):
    return LemTokens(tokenizer.tokenize(text.lower().translate(remove_punc_dict)))

#####################################################
#Greeting User
#####################################################


greet_inputs = [
    "hi",
    "hello",
    "hey",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you?",
    "what's up?",
    "howdy",
    "hi there",
    "yo",
    "what's going on?",
    "how do you do?",
    "what's new?",
    "wassup?",
]


greet_responses = [
    "Bot: Hey there! What's up? How can I help?",
    "Bot: Hi! Ready for a chat? Let me know what you need!",
    "Bot: Hey There! How can I assist you today?",
    "Bot: Hey! Great to see you! What's on your mind?",
    "Bot: Hi How can I make your day better?",
    "Bot: Hello! What's cooking? How can I be of service?"
]

       
# Assigning score to the matches of input and books info in dataset, return best 3 matches
def jaccard_similarity(user_input_fin):
   
    df["Jaccard_Similarity"] = df.apply(lambda row: len(set(user_input_fin).intersection(set(row["Combined_Tokens"]))) / len(set(user_input_fin).union(set(row["Combined_Tokens"]))), axis=1)
    df_with_best = df.nlargest(3, "Jaccard_Similarity")
   
    titles_list = df_with_best["Title"].tolist()
    author_list = df_with_best["Authors"].tolist()

    return (
    "Bot: I recommend:\n\n"
    "~ <b>{}</b> {}\n\n".format(titles_list[0].title(), author_list[0]) +
    "~ <b>{}</b> {}\n\n".format(titles_list[1].title(), author_list[1]) +
    "~ <b>{}</b> {}\n\n".format(titles_list[2].title(), author_list[2]) + 
    "<b>Bot:</b> Say 'more' or 'other' to receive more recommendations or type a new prompt."
)


#######################################################
    # If asked for more recomendations
#######################################################
   
def more_rec(user_input_fin, num_of_more):

    df["Jaccard_Similarity"] = df.apply(lambda row: len(set(user_input_fin).intersection(set(row["Combined_Tokens"]))) / len(set(user_input_fin).union(set(row["Combined_Tokens"]))), axis=1)
   
    number = 3 * num_of_more
    df_with_best = df.nlargest(number, "Jaccard_Similarity").iloc[(num_of_more - 1) * 3:number]
   
    if not df_with_best.empty:
        titles_list = df_with_best["Title"].tolist()
        author_list = df_with_best["Authors"].tolist()
        return (

        "Bot: I recommend:\n\n"
        "~ <b>{}</b> {}\n\n".format(titles_list[0].title(), author_list[0]) +
        "~ <b>{}</b> {}\n\n".format(titles_list[1].title(), author_list[1]) +
        "~ <b>{}</b> {}\n\n".format(titles_list[2].title(), author_list[2]) +
        "<b>Bot:</b> Say 'more' or 'other' to receive more recommendations or type a new prompt."
    )
        
    else:
       return (

        "I don't have enough recommendations."
    )


###############################################
        # Starting a bot
###############################################
    
previous_response= 0
num_of_more = 1

def main(user_response):
    global previous_response
    global num_of_more

    bye_words = [
        'bye',
        'goodbye',
        'farewell',
        'see you later',
        'see ya',
        'later',
        "i'm leaving",
        'i have to go',
        'time to go',
        'signing off',
        'take care',
        'catch you later',
        'until next time',
        'adios',
        'toodles',
    ]
   
   # Process user input
    user_response = user_response.lower()

    # Check for special cases
    if any(keyword in user_response for keyword in bye_words):
        return "Bot: Goodbye! Have a great day!"
    elif user_response.lower() in greet_inputs:
        return random.choice(greet_responses)
    
    elif user_response in ["thank you", "thanks"]:
        return "Bot: You Are Welcome."
    
    elif user_response in ["more", "other", "other books"]:
        num_of_more += 1

        return more_rec(previous_response, num_of_more) # Call more_rec and return the result
        
    else:
        # Process user input and return recommendations
        final_user_text = LemNormalize(user_response)
        previous_response = final_user_text
        if not final_user_text:
            return "Bot: I do not quite understand. Please, provide your book describtion again."
        else:
            recs = jaccard_similarity(final_user_text)
            return recs