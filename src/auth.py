import json
import tweepy

def authpy(credentials='../credentials.json'):
    '''
    Authenticate to Twitter and get API object.
    '''
    creds = read_creds(credentials)
    key, secrets = creds['api_key'], creds['api_secrets']
    tk, tk_secrets = creds['access_token'], creds['access_secret']
    tk_bearer = creds['bearer_token']

    # Create the client with keys
    client = tweepy.Client(bearer_token=tk_bearer,
                           consumer_key=key,consumer_secret=secrets,
                           access_token=tk,access_token_secret=tk_secrets)
    return client

def read_creds(filename):
    '''
    Read JSON file to load credentials.
    Store API credentials in a safe place.
    If you use Git, make sure to add the file to .gitignore
    '''
    with open(filename) as f:
        credentials = json.load(f)
    return credentials

#if __name__ == '__main__':
#    client = authpy()
#    client.search_recent_tweets(query="les anneaux de pouvoir",tweet_fields=['context_annotations'], max_results=10)
