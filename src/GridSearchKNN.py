import matplotlib.pyplot as plt
import seaborn as sns
import seaborn; seaborn.set()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
seaborn.set_style(style='white')

from sklearn.model_selection import train_test_split
import pandas as pd


from KNN import *

def gridsearch(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

    k_values = [3,5,11,19,27,40]

    knn = KNN().fit(X_train,y_train)

    l_func = [get_distance, levenshtein]

    acc = {f : [] for f in l_func}

    for func in l_func:
        
        knn.set_distance_func(func)

        for k in k_values:
        
            knn.set_k(k)
            acc[func].append(knn.score(X_test,y_test))

            ConfusionMatrixDisplay(confusion_matrix(y_test,knn.y_pred,labels=knn.classes),display_labels=knn.classes).plot()

            if func == get_distance:
                plt.title("K = " + str(k) +", get_distance")
                plt.savefig("img/confusion_matrix_getdistance " + str(k) + ".jpg")
            else:
                plt.title("K = " + str(k) +", levenshtein")
                plt.savefig("img/confusion_matrix_levenshtein " + str(k) + ".jpg")

    plt.figure(figsize=(10,6))
    plt.plot(k_values,acc[get_distance],label="get_distance",color = 'blue',linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.plot(k_values,acc[levenshtein],label='levenshtein',color = 'orange',linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K-Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')


    plt.savefig("img/gridsearchKNN_accuracy.jpg")

if __name__ == '__main__':
    df = pd.read_csv("data/basis_learning/tweets_clean_annoted.csv",index_col=0)
    
    X, y = df['tweet'], df['polarity']
    
    gridsearch(X.values,y.values)

