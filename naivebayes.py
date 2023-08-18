# scikit-learn manually installed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import main

def train(df):
    df[main.MOOD].replace({'pos':1, 'neg':0}, inplace=True)
 
    cv = CountVectorizer(analyzer = 'word',ngram_range=(1,3))
    tv = TfidfVectorizer()
    combined = FeatureUnion([("bow", cv), ("tfidf", tv)])
    X = combined.fit_transform(df[main.PREPROCESSED])
    # print(X)
    y = df.iloc[:,-4]
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    # Naive Bayes Classifier
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)

    # accuracy scores
    print("Accuracy: ", accuracy_score(y_test, y_pred_mnb), "\nPrecision: ",precision_score(y_test, y_pred_mnb), "\nRecall: ", recall_score(y_test, y_pred_mnb), "\nF1-score: ", f1_score(y_test, y_pred_mnb))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_mnb)
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mnb.classes_)
    disp.plot()
    # plt.show()

    
    return combined, mnb