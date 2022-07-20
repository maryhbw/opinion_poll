#import packages
import re
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
#LDA Analysis
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from twitter_import import save_tweets_csv

def load_saved_data(keyword):
    path = f'/Users/maryward/code/maryhbw/opinion_poll/opinion_poll/data/'
    path = f'{path}{keyword}.csv'
    count = 0
    for file in os.listdir(path):
        print(file)
        # open the file
        with open(os.path.join(path, file), 'rb') as f:
            print(f)
            df = pd.read_csv(f)
            df['date'] = file[:10]
            if count ==0:
                df2 = df
                count +=1
            else:
                print(count)
                df2 = pd.concat([df,df2], axis = 0)
    return df2


def clean_text_data(df):
    '''removes urls, mentions, and hashtags and lowercases the text
    returns DataFrame with clean_text, mentions, and hashtags as new columns'''
    df2 = df.copy()
    df2['clean_text'] = df2['text'].apply(lambda x: re.sub(r"https?://\S+", '', str(x), flags=re.MULTILINE))
    #store mentions
    mentions =df2['text'].apply(lambda x : re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
    df2['mentions'] = mentions
    #store hashtags
    hashtags = df2['text'].apply(lambda x : re.findall("#([a-zA-Z0-9_]{1,50})", str(x)))
    df2['hashtags'] = hashtags
    #remove mentions, remove hashtags
    df2['clean_text'] = df2['clean_text'].apply(lambda x: re.sub("@([a-zA-Z0-9_]{1,50})", '', str(x)))
    df2['clean_text'] = df2['clean_text'].apply(lambda x: re.sub("#([a-zA-Z0-9_]{1,50})", '', str(x)))
    #lowercase
    df2['clean_text'] = df2['clean_text'].str.lower().str.strip()
    return df2

def label_retweets(df2):
    '''assigns 1 if "retweet"  and 0 if source tweet
    returns df2 with 'retweet' column'''
    retweet = []
    for  x in df2['clean_text']:
        if 'rt :' in x:
            retweet.append(1)
        else:
            retweet.append(0)
    #assign boolean to 'retweet'
    df2['retweet'] = retweet
    return df2

def polarity_sentiment(df2):
    ''' use TextBlob to label 'clean_text' for polarity
    (Subjective/Objective) 0.5 cut off and for sentiment
    (Positive = 1, Negative = -1, Neutral = 0)'''
    #change 'clean_text' column to a list of texts
    lst = df2['clean_text'].to_list()
    polarity_list_blob = []
    subjectivity_list_blob = []
    for sentence in lst:
        sentence = sentence.replace('"','').replace("'", '')
        polarity = TextBlob(sentence).sentiment[0]
        subjectivity = TextBlob(sentence).sentiment[1]

        if len(TextBlob(sentence)):

            if polarity > 0.2:
                polarity = 1 #"Positive"
                polarity_list_blob.append(polarity)
            elif polarity < -0.2:
                polarity = -1 #"Negative"
                polarity_list_blob.append(polarity)
            else:
                polarity = 0 #"Neutral"
                polarity_list_blob.append(polarity)
            if  subjectivity > 0.5:
                subjectivity = "Subjective"
                subjectivity_list_blob.append(subjectivity)
            else:
                subjectivity = "Objective"
                subjectivity_list_blob.append(subjectivity)

        else:
            polarity_list_blob.append(np.NaN)
            subjectivity_list_blob.append(np.NaN)
    #add scores to the DataFrame
    df2['blob_polarity'] = polarity_list_blob
    df2['blob_subjectivity'] = subjectivity_list_blob

    return df2

def clean_lemmatize(text):
    """cleans and lemmatizes the words with WordNetLemmatizer()"""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized

def apply_clean_lemmatize(df2):
    df2['clean_text_lemma'] = df2['clean_text'].apply(clean_lemmatize)
    df2['clean_text_lemma'] = df2['clean_text_lemma'].astype('str')
    return df2

def LDA_trainer(df2):
    '''Train a LDA model on text data, source, to extract potential topics
    currently 3 topic node used in the model'''
    source_tweet = df2[df2['retweet']==0]
    #initialize vectorizer
    vectorizer = CountVectorizer()
    data_vectorized = vectorizer.fit_transform(source_tweet['clean_text_lemma'])
    #initialize LDA model
    lda_model = LatentDirichletAllocation(n_components=3)
    lda_vectors = lda_model.fit_transform(data_vectorized)
    #get topics and put into a dict
    topic_dict = {}
    for idx, topic in enumerate(lda_model.components_):
        topic_dict["Topic %d:" % (idx)] = [(vectorizer.get_feature_names_out()[i], topic[i])for i in topic.argsort()[:-10 - 1:-1]]
    return topic_dict, vectorizer, lda_model

def get_topic(df2, vectorizer, lda_model):
    """takes a clean_text tweet and determines the topic it
    most likely belongs to. Good for tracking re-tweets and predicting trends
    based on word"""
    df = df2.copy()
    topic_LDA = []
    for i in range(len(df)):
        example = [df['clean_text'].iloc[i]]
        #Vectorize the example
        example_vectorized = vectorizer.transform(example)
        #use trained LDA model to predict topic
        lda_vectors = lda_model.transform(example_vectorized)
        ldas = {lda_vectors[0][0]:1,lda_vectors[0][1]:2, lda_vectors[0][2]:3}
        top = ldas[np.max([lda_vectors[0][0],lda_vectors[0][1],lda_vectors[0][2]])]
        topic_LDA.append(top)
    df['topic_LDA'] = topic_LDA
    return df



if __name__ in "__main__":
    df2 = load_saved_data(keyword = 'abortion')
    df2 = clean_text_data(df2)
    df2 = filter_retweets(df2)
    df2 = polarity_sentiment(df2)
    df2 = apply_clean_lemmatize(df2)
    topic_dict, vectorizer, lda_model = LDA_trainer(df2)
    df2 = get_topic(df2, vectorizer, lda_model)
    save_tweets_csv(df2,keyword = 'abortion')
