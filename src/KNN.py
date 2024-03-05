import numpy as np
import functools
import Levenshtein

def get_distance(t1,t2):

    t1 = np.array(t1.split(' '))
    t2 = np.array(t2.split(' '))
    nb_common = np.intersect1d(t1,t2).size
    nb_total = t1.size + t2.size
    
    return (nb_total - nb_common) / nb_total

def levenshtein(t1, t2):
    return Levenshtein.distance(t1,t2) / max(len(t1),len(t2))


class KNN():

    def __init__(self,d_func=get_distance,k=19):

        self.k=k
        self.d_func = d_func
        self.y_pred = None
        self.classes = None

    def set_distance_func(self,d_func):
        self.d_func = d_func

    def set_k(self,k):
        self.k = k

    def get_k(self):
        return self.k

    def fit(self,X_train,y_train):

        self.X_train=X_train
        self.y_train=y_train
        self.classes = np.unique(y_train)

        return self

    def nearest_neighbors_predict(self,tweet):

        d = np.vectorize(functools.partial(self.d_func,tweet))(self.X_train)
        
        index = np.argpartition(d, self.get_k() + 1)
        
        return np.bincount(self.y_train[index][:self.get_k()]).argmax()
    
    def predict(self,X_test):
        
        return np.vectorize(self.nearest_neighbors_predict)(X_test)

    def score(self,X_test,y_test):
        
        self.y_pred = self.predict(X_test)

        unique, counts = np.unique((self.y_pred == y_test), return_counts=True)

        return dict(zip(unique, counts))[True] / len(X_test)