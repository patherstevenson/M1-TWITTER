import re
import pandas as pd

def remove_duplicate_tweet(dataframe):
    return dataframe.drop_duplicates(subset=['tweet'])

def clean_tweet(tweet):

    # remove @user tag, #hashtags, url links
    tweet = re.sub('(@[^\s]+)|(#[^\s]+)|(#)|(http[^\s]+)|("|"")','',tweet)

    # replace EOL by space
    tweet = re.sub('\n',' ',tweet)

    # remove emoji
    emoji_pattern = re.compile("["
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F700-\U0001F77F"  # alchemical symbols
                            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                            u"\U00002702-\U000027B0"  # Dingbats
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE).sub(r'', tweet).strip().lower()

    return emoji_pattern

