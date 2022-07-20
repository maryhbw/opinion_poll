import pandas as pd
from clean_label_tweets import load_saved_data
import matplotlib.pyplot as plt
import re


df2 = load_saved_data(keyword = 'abortion')
path = '/Users/maryward/code/maryhbw/opinion_poll/opinion_poll/'
#distribution of tweets by time
def time_of_tweet(df2,keyword):
    '''returns a histogram of tweets by time posted
    proxy for geographical location and nice for tageting ads by
    taget active time'''
    df2['created_at'] = pd.to_datetime(df2['created_at'])
    df2['time'] = [str(x)[11:13] for x in df2['created_at']]
    df2['time'].value_counts().sort_index().plot(kind='bar')
    plt.title(f'Time of tweet by users')
    plt.xlabel('Hour of day UTC')
    plt.ylabel('Tweet volume')
    plt.close()
    plt.savefig(f'{path}data/{keyword}.time_tweet.png')
    plt.show()

#distribution of tweets by author_id

def tweets_by_user(df2, keyword):
    '''returns a histogram of tweets by author_id
    to find key opinion leaders and basic twitter use by account'''
    df2["author_id"].value_counts().plot(kind = 'hist') #.sort_values(ascending = False)
    plt.title(f"Twitter Use per day by User, keyword: '{keyword}'")
    plt.xlabel('Number of tweets per day')
    plt.ylabel('Number of users')
    plt.savefig(f'{path}data/{keyword}.user_tweet_freq_all.png')
    plt.show()
    #histogram of accounts that tweet more than 2x per day
    users_df = df2.groupby("author_id").agg("count")
    high_users = users_df[users_df['created_at']>2] #109/1399 users tweet more than 1 time per week
    high_users['created_at'].plot(kind = 'hist')
    plt.title(f"Twitter Use per day by User, keyword: '{keyword}'")
    plt.xlabel('Number of tweets per day')
    plt.ylabel('Number of users')
    plt.savefig(f'{path}data/{keyword}.user_tweet_freq_high.png')
    plt.show()

#distribution of tweets by sentiment
def tweets_by_sentiment(df2,keyword):
    df2['blob_polarity'].value_counts().plot(kind='bar')
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticklabels(['Neutral','Positive','Negative'], minor = False, rotation= 45)
    plt.title(f"Tweet sentiment, keyword: '{keyword}'")
    plt.xlabel('Sentiment')
    plt.ylabel('Number of tweets')
    plt.savefig(f'{path}data/{keyword}.sentiment.png')
    plt.show()

#distribution of tweets by polarity
def tweets_by_polarity(df2, keyword):
    df2['blob_subjectivity'].value_counts().plot(kind='bar')
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticklabels(['Objective','Subjective'], minor = False, rotation= 45)
    plt.title(f"Tweet Subjectivity, keyword: '{keyword}'")
    plt.xlabel('Subjectivity')
    plt.ylabel('Number of tweets')
    plt.savefig(f'data/{keyword}.sentiment.png')
    plt.show()

#distribution of tweets by topic
def tweets_by_topic(df2, keyword):
    df2['topic_LDA'].value_counts().plot(kind='bar')
    plt.title(f"Tweet Topics, keyword: '{keyword}'")
    plt.xlabel('Topic')
    plt.ylabel('Number of tweets')
    plt.savefig(f'{path}data/{keyword}.topic.png')
    plt.show()
#distibution of tweet by source
def tweets_by_souce(df2, keyword):
    '''distribution of tweets be hardware source,
    ie iphone, android, etc'''
    df2['source'].value_counts().plot(kind='bar')
    plt.title(f"Tweet Source, keyword: '{keyword}'")
    plt.xlabel('Source')
    plt.ylabel('Number of tweets')
    plt.savefig(f'{path}data/{keyword}.source.png')
    plt.show()
#retweet by topic
def retweet_by_topic(df2,keyword):
    '''distibution of retweet by topic'''
    df_retweet = df2[df2['retweet']==1]
    df_retweet['topic_LDA'].value_counts().sort_index().plot(kind = 'bar')
    plt.title(f"Distribution of Retweets by Topic, keyword: '{keyword}'")
    plt.xlabel('Topic')
    plt.ylabel('Number of tweets')
    plt.savefig(f'{path}data/{keyword}.topic_retweet.png')
    plt.show()
#retweet by author_id
#depending on the call, popular tweets' author may not be captured
def top_retweets(df2, keyword):
    '''groups retweets by tweet and returns the content of the
    top 5 tweets and saves the filtered DataFrame as a CSV'''
    df2['clean_text'] = df2['clean_text'].apply(lambda x: re.sub('rt : ','', str(x)))
    retweet = df2[df2['retweet']==1]
    retweet_freq = retweet.groupby('clean_text').agg('count').reset_index().sort_values('created_at', ascending = False).iloc[:5]
    count = 0
    lst = retweet_freq['clean_text'].to_list()
    for i, row in enumerate(retweet):
        if retweet['clean_text'].iloc[i] in lst:
            if count == 0:
                s1 = retweet.iloc[i]
                df = pd.DataFrame(s1)
                count +=1
                print(i)
            else:
                s1 = retweet.iloc[i]
                df = df.join(s1)
                print(i)
    df = df.T
    df.to_csv(f'{path}{keyword}.top_tweets.csv')

#distsribution of hashtags
def distribution_hashtags(df2, keyword):
    hashtags = {}
    for i,row in enumerate(df2['hashtags']):
        for x in df2['hashtags'].iloc[i]:
            if x in hashtags:
                hashtags[x] +=1
            else:
                hashtags[x] =1
    hashes = pd.DataFrame(hashtags.values(), hashtags.keys()).sort_values(0, ascending = False).reset_index().rename(columns = {'index':'hashtags', 0:'count'})
    hashes.to_csv(f'{path}{keyword}.hashtags.csv')

#correlation of topic and sentiment
#correlation of topic and source
