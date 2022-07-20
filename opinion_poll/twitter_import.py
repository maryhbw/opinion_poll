"""script to import tweets and save as csv to data or pass to later
functions as a DataFrame"""
##IMPORT PACKAGES##
# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
##import os
from decouple import config
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
# For Creating a timeseries with which to query twitter
import datetime
import pytz
import dateutil.parser
import unicodedata
from datetime import datetime,  timedelta

def auth():
    '''retrieve Twitter API token saved
    as an enviromental variable'''
    TOKEN = config('TOKEN')
    return TOKEN

def create_headers():
    bearer_token = auth()
    '''create headers from bearer token'''
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(keyword, start_date, end_date, max_results = 100):
    '''make a url for the API call'''
    search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date, #start-time limited to 1 week with Essential Access from Twitter
                    'max_results': max_results, #maximum is 100
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source,context_annotations',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    '''function to connect to twitter API endpoint'''
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def loop_through_query(headers, keyword , start_time , end_time ,n_tweets = 1000, max_results = 100):
    """funtion loops through next_tokens to return n_tweets"""
    '''define '''
    # now we have two json STRINGS
    next_token = None
    url = create_url(keyword, start_time,end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
    df_keep = pd.DataFrame(json_response['data'])
    #df_keep = df[['created_at','source', 'text', 'author_id',  'conversation_id','context_annotations']] #'geo',
    for i in range(round(n_tweets/100)):
        if next_token:
            url = create_url(keyword, start_time,end_time, max_results)
            json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
            if len(json_response) !=0:
                df_new = pd.DataFrame(json_response['data'])
               # df_new = df[['created_at','source', 'text', 'author_id',  'conversation_id','context_annotations']] #'geo',
                df_keep = pd.concat([df_keep, df_new], axis=0) #final_dict = {key: value for (key, value) in (final_dict.items() + dictA.items())}
        if json_response['meta']['result_count'] >= max_results:
            next_token = json_response['meta']['next_token']
        else:#
            next_token = None
    return df_keep

def recent_time_series(days):
    '''creates a list of dates to cycle through for making a times series for 1 week'''
    #current timezone set to 'UTC' as this is the form that the query takes
    tz = pytz.timezone('UTC')
    start_times = []
    end_times = []
    for i in range(days):
        tz = pytz.timezone('UTC')
        today = str(datetime.now(tz))[:26]
        today = today - timedelta(hours=0.20) #offset seach start to buffer query from 10min delay set by Twitter
        end_date = (today - timedelta(days=i)).isoformat() #slice by days, could be hours, minutes, etc.
        start_date = (today - timedelta(days=(i+1))).isoformat()
        start_times.append((start_date[:23]+'Z'))
        end_times.append((end_date[:23]+'Z'))

    return start_times, end_times

def recent_time_by_hour(hours):
    '''creates a list of dates to cycle through for making a times series for hours
    up to one week back max = 168'''
    start_times = []
    end_times = []
    for i in range(hours):
        today = datetime.today() - timedelta(hours=5.05) #timedelta based on diff to UTC, 2.05 from Amsterdam
        end_date = (today - timedelta(hours=i)).isoformat() #slice by days, could be hours, minutes, etc.
        start_date = (today - timedelta(hours=(i+1.05))).isoformat()
        start_times.append((start_date[:23]+'Z'))
        end_times.append((end_date[:23]+'Z'))

    return start_times, end_times

def time_series_api_call(headers,keyword, n_tweets= 1000, hours=None,  days = None):
    '''takes a list of start and end times and returns a concatenated
    DataFrame of the tweets over the specified times'''
    count = 0
    if hours:
        start_times, end_times = recent_time_by_hour(hours)
    if days:
        start_times, end_times = recent_time_series(days)
    keyword = f'{keyword} lang:en'
    for start_time, end_time in zip (start_times,end_times):
        df = loop_through_query(n_tweets = n_tweets, headers = headers, keyword = keyword, start_time = start_time, end_time = end_time, max_results = 100)
        if count ==0:
            df2 = df
            count +=1
        # df2 = df2[['created_at','source', 'text', 'author_id',  'conversation_id']]
        else:
            print(count)
            df2 = pd.concat([df,df2], axis = 0)
    return df2

def save_tweets_csv(df,keyword):
    path = f'/Users/maryward/code/maryhbw/opinion_poll/opinion_poll/data/'#change filename
    df.to_csv(f'{path}{keyword}.csv')

if __name__ in "__main__":
    headers = create_headers()
    df = time_series_api_call(headers, keyword = 'hotdogs', n_tweets= 10, hours=3,  days = None)
    print(df.columns)
