
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


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

clf = svm.LinearSVC(loss='hinge', C=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.f1_score(y_test, y_pred))

print(pd.DataFrame(
      metrics.confusion_matrix(y_test, y_pred),
      index=[['actual', 'actual'], ['spam', 'ham']],
      columns=[['predicted', 'predicted'], ['spam', 'ham']]))

# param_grid = [{'C': np.logspace(-4, 4, 15)}]
#
# grid_search = GridSearchCV(
#     estimator=svm.LinearSVC(loss='hinge'),
#     param_grid=param_grid,
#     cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
#     scoring='f1',
#     n_jobs=-1
# )


# grid_search.fit(X_ngrams, y_enc)
# final_clf = svm.LinearSVC(loss='hinge', C=grid_search.best_params_['C'])
# final_clf.fit(X_ngrams, y_enc)
# y_pred = final_clf.predict(X_test)
#
# print('Best Parameter:' + str(grid_search.best_params_['C']))
#
# print(metrics.f1_score(y_test, y_pred))
#
# print(pd.DataFrame(
#       metrics.confusion_matrix(y_test, y_pred),
#       index=[['actual', 'actual'], ['spam', 'ham']],
#       columns=[['predicted', 'predicted'], ['spam', 'ham']]))


print(pd.Series(
      clf.coef_.T.ravel(),
      index=vectorizer.get_feature_names()).sort_values(ascending=False)[:20])

print(pd.Series(
      clf.coef_.T.ravel(),
      index=vectorizer.get_feature_names()).sort_values(ascending=True)[:20])


df_final = pd.DataFrame()

df_input = pd.read_csv('prepocessed_input.csv')

df_list = list()
for index in df_input.index:
    df_tmp = pd.DataFrame()

    username = df_input['username'].loc[index]
    tweet = df_input['tweet'].loc[index]

    tmp = vectorizer.transform([tweet])
    score = clf.decision_function(tmp)

    df_tmp['username'] = username
    df_tmp['score'] = score

    if clf.predict(tmp):
        df_tmp['category'] = 'r'
    else:
        df_tmp['category'] = 'o'

    df_list.append(df_tmp)

df_final = df_final.concat(df_list)

df_final.to_csv('categorized_users.csv')
