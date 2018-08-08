
import pandas as pd
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC


print('started running')
df = pd.read_csv('preprocessed_data.csv')
print('red df')
processed = df['tweet']
y = df['label']
df = df.dropna()

texts = df['tweet'].values
labels = df['label'].values

print(df.head)

processed = texts
y = labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(processed)


X_train, X_test, y_train, y_test = train_test_split(
    X_ngrams,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)
#
# clf = SVC()
# clf.fit(X_train, y_train)
# print('fit done')
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
#

clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.f1_score(y_test, y_pred))

print('SVC Done')

y_pred = clf.predict(X_test)

print(metrics.f1_score(y_test, y_pred))
