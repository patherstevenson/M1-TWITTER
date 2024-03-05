import numpy as np

from clean_basis import clean_tweet

def verify_len(s,s_len=3):
    return len(s) > s_len

def N_grammes(n,t_split,combinaison=False):
    if not combinaison:
        j = n
    else:
        j = np.random.randint(1,n+1)
        
    return np.array([' '.join(t_split[i:i+j]) for i in range(0,len(t_split),j)])

def filter_min_word(split_tweet):
    tweet = []
    
    for m in split_tweet:
        if len(m) > 3:
            tweet.append(m)
    return tweet

class BayesClassifier():
    
    def __init__(self,freq=False,n_gram=1,combinaison=False):
        
        self.n_gram = n_gram
        self.combinaison = combinaison
        self.freq = freq
        self.min_word = 3
        self.p_wc_matrix = None

    def get_n_gram(self):
        return self.n_gram
        
    def fit(self,X,y):

        self.classes, counts = np.unique(y,return_counts=True)

        self.p_c = counts / len(y)

        split_tweet = np.char.split(X.astype(str), sep =' ')
        words = np.unique(np.concatenate(split_tweet,axis=0))
        self.all_words = words[np.vectorize(verify_len)(words)]

        matrix = np.zeros((len(self.all_words),3),dtype=float)

        self.sorter = np.argsort(self.all_words)

        for i in range(len(split_tweet)):

            tweet = filter_min_word(split_tweet[i]) 

            if self.n_gram > 1:
                tweet = N_grammes(self.n_gram,tweet,self.combinaison)

            index = np.searchsorted(self.all_words, tweet, sorter=self.sorter)
            index = self.sorter[index[index < len(self.all_words)]]
            
            y_true = int(y[i]/2)

            for i in index:
                matrix[i][y_true] += 1

        self.p_wc_matrix = (matrix+1) / (matrix.sum(axis=0) + len(self.all_words))

        return self

    def _predict(self,tweet): #single private predict
        
        t = filter_min_word(tweet)

        if self.n_gram > 1:
            t = N_grammes(self.n_gram,tweet,self.combinaison)

        index = np.searchsorted(self.all_words,t, sorter=self.sorter)
        index = self.sorter[index[index < len(self.all_words)]]
        
        if not self.freq:
            index = np.unique(index)

        return (np.prod(self.p_wc_matrix[index],axis=0) * self.p_c).argmax() * 2

    def predict(self,X_test): # vector
        split_tweet = (np.char.split((np.vectorize(clean_tweet)(X_test)).astype(str), sep =' '))
        return np.vectorize(self._predict)(split_tweet)
    

    def score(self, X_test, y_test):
        
        unique, counts = np.unique((self.predict(X_test) == y_test), return_counts=True)

        return dict(zip(unique, counts))[True] / len(X_test)