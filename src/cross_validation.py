import numpy as np

def cross_val_score(X,y,k,estimator):
    
    indices = list(range(len(X)))

    np.random.shuffle(indices)    
    folds = np.array_split(indices,k)
    
    scores = []
    
    for i in range(len(folds)):

        index_train = folds.copy()
        index_valid = index_train.pop(i)

        index_train = np.concatenate(index_train)

        estimator.fit(X[index_train],y[index_train])

        scores.append((estimator.score(X[index_valid],y[index_valid])))

    return np.average(scores)

