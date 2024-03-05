import pandas as pd
from clean_basis import clean_tweet, remove_duplicate_tweet
from polarity import get_polarity
from KNN import *

import sys

def main(csv_path,csv_clean_path):
    
    print('Starting main_basis.py Twitter Sentiment Analysis :',end='\n\n')

    df = pd.read_csv(csv_path)

    print(f'\t - pandas.DataFrame("{csv_path}") loaded',
            u'\N{check mark}',end='\n')

    df['tweet'] = df['tweet'].apply(clean_tweet)

    print(f'\t - tweets of DataFrame have been cleaned ',
            u'\N{check mark}',end='\n')

    before_len = len(df)

    df = remove_duplicate_tweet(df)

    remove_duplicate = before_len - len(df)

    print(f'\t - Duplicate tweets have been removed : {remove_duplicate} duplicates ',
            u'\N{check mark}',end='\n')

    df['polarity'] = df['tweet'].apply(get_polarity)

    print(f'\t - The polarity of the tweets has been changed',u'\N{check mark}',end='\n')

    df.to_csv(csv_clean_path)

    print(f'\t - DataFrame saved in csv file : {csv_clean_path} ', u'\N{check mark}',end='\n\n')

if __name__ == '__main__':
        main("data/basis_learning/tweets.csv", "data/basis_learning/tweets_clean_annoted.csv")
