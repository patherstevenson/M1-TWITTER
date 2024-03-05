NONE=-1
NEGATIVE=0
NEUTRAL=2
POSITIVE=4

def get_class_keywords(l_class=['positive','negative'],separator=', '):

    d = dict()

    for class_name in l_class:
        with open('data/classes/' + class_name + '.txt','r') as fd:
            d[class_name] = fd.readlines()[0].split(separator)

    return d

def get_polarity(tweet,d_keywords=get_class_keywords()):

    text = tweet.split(' ')

    n, p = 0, 0

    for words in text:
        if words in d_keywords['positive']:
            p += 1
        if words in d_keywords['negative']:
            n += 1
    
    if n > p:
        return NEGATIVE
    elif n == p:
        return NEUTRAL
    elif p > n:
        return POSITIVE
    else:
        return NONE