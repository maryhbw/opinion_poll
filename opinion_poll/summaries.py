from clean_label_tweets import load_saved_data
import re
#summarization
import nltk
import sumy
#more modules to import
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.text_rank import TextRankSummarizer

df2 = load_saved_data(keyword='abortion')

#prepare data for summarization pipeline
#requires tweets to be concatenated into a single string

def tweets_to_string(df2):
    '''takes a DataFrame with clean_text and concatenates
    all the clean_text tweets into a single string for
    the summarization pipeline with summy'''
    df_users = df2.groupby('author_id').agg('count').sort_values('created_at', ascending=False)
    #filters tweets for accounts that tweeted at least 2x in time interval
    df_masses = df_users[df_users['created_at']<=2].reset_index()
    full_text_punct = ""
    for i in range(len(df_masses)):
        df = df2[df2['author_id']==df_masses['author_id'][i]]
        df = df['clean_text'].apply(lambda x: re.sub('rt : ', ' ', x, flags=re.MULTILINE))
        df = df.reset_index()
        #separates each tweet by a period punctuation
        for j in range(len(df)):
            full_text_punct = full_text_punct + df['clean_text'].iloc[j] + '. '
    return full_text_punct

def Lex_Rank_summarizer(df2, sentences = 2):
    '''used for titling final product. summarizes based on frequency of sentence in document
    and returns number of sentences specified. Parser is instatiated.'''
    chunks = []
    count = 0
    for i in range(500,len(df2[df2['retweet']==0]), 500):
        document = ''
        for x in df2[df2['retweet']==0]['clean_text'].iloc[count:i].to_list():
            document = document + x +'. '
        chunks.append(document)
        count += 500
    summaries = []
    for i in range(len(chunks)):
        parser = PlaintextParser.from_string(chunks[i], Tokenizer("english"))
        summarizer = LexRankSummarizer(Stemmer("english"))
        summary = summarizer(parser.document, sentences)
        short_summary = [sentence for sentence in summary]
        summaries.append(short_summary)
    return summaries

def Luhn_summarizer(full_text_punct, sentences = 2):
    '''Luhn’s algorithm is an approach based on TF-IDF.
    It selects only the words of higher importance as per their frequency.
    Higher weights are assigned to the words present at the begining of the document.
    Parser is instatiated.'''
    parser = PlaintextParser.from_string(full_text_punct, Tokenizer("english"))
    summarizer_luhn = LuhnSummarizer()
    #Summarize the document with 2 sentences
    summary = summarizer_luhn(parser.document,sentences)
    summary1 = [sentence for sentence in summary]
    return summary1

#luhn vs LSA parse two sides of the story!
def LSA_summarizer(full_text_punct, sentences = 2):
    '''Based on term frequency techniques with singular value decomposition to summarize texts.'''
    parser = PlaintextParser.from_string(full_text_punct, Tokenizer("english"))
    summarizer_lsa = LsaSummarizer(Stemmer("english"))
    summarizer_lsa.stop_words = get_stop_words("english")
    summary =summarizer_lsa(parser.document,sentences)
    summary2 = [sentence for sentence in summary]
    return summary2

def text_rank_summarizer(full_text_punct, sentences = 2):
    '''Text rank is a graph-based summarization technique with keyword extractions in from document.'''
    parser = PlaintextParser.from_string(full_text_punct, Tokenizer("english"))
    summarizer = TextRankSummarizer(Stemmer("english"))
    summary3 =summarizer(parser.document,3)
    return summary3

def full_summary(summaries, summary1, summary2, summary3):
    sum_dict = {'lex_rank': summaries, 'luhn': summary1, 'lsa': summary2, 'text_rank':summary3}
    return sum_dict

if __name__ in "__main__":
    df2 = load_saved_data(keyword='abortion')
    full_text_punct = tweets_to_string(df2)
    summaries = Lex_Rank_summarizer(df2)
    summary1 = Luhn_summarizer(full_text_punct)
    summary2 = LSA_summarizer(full_text_punct, sentences = 2)
    summary3 = text_rank_summarizer(full_text_punct, sentences = 2)
    sum_dict = full_summary(summaries, summary1, summary2, summary3)
    print(len(sum_dict))
