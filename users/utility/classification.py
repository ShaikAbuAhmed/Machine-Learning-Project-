import pandas as pd
from django.conf import settings
import os

path = os.path.join(settings.MEDIA_ROOT, 'HotelReviews.csv')
dataset = pd.read_csv(path, nrows=1000)
#Here we will sort the dataframe according to 'Time' feature
#dataset.sort_values(['Time'], ascending=True, inplace=True)
print(dataset.head())

# Make all 'rating' less than 3 equal to -ve class and
# 'rating' greater than 3 equal to +ve class.
dataset.loc[dataset['rating']<3, 'rating'] = 0
dataset.loc[dataset['rating']>3, 'rating'] = 1
print(dataset.head())
#import numpy as np
#from sklearn.model_selection import train_test_split
#train, test = train_test_split(dataset, test_size = 0.3)

total_size=len(dataset)

train_size=int(0.70*total_size)

#training dataset
train=dataset.head(train_size)
#test dataset
test=dataset.tail(total_size - train_size)

train.rating.value_counts()
test.rating.value_counts()
# Removing all rows where 'Score' is equal to 3
train = train[train.rating != 3]
test = test[test.rating != 3]

print(train.shape)
print(test.shape)
train['rating'].value_counts()
test.rating.value_counts()
#Taking the 'Text' & 'Summary' column in seperate list for further
#text preprocessing.
lst_text = train['reviews'].tolist()
lst_summary = train['title'].tolist()

test_text = test['reviews'].tolist()
#Converting the whole list to lower-case.
lst_text = [str(item).lower() for item in lst_text]
lst_summary = [str(item).lower() for item in lst_summary]
test_text = [str(item).lower() for item in test_text]

#Lets now remove all HTML tags from the list of strings.
import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

for i in range(len(lst_text)):
    lst_text[i] = striphtml(lst_text[i])
    lst_summary[i] = striphtml(lst_summary[i])

for i in range(len(test_text)):
    test_text[i] = striphtml(test_text[i])
lst_text[0:5]

#Now we will remove all special characters from the strings.
for i in range(len(lst_text)):
    lst_text[i] = re.sub(r'[^A-Za-z]+', ' ', lst_text[i])
    lst_summary[i] = re.sub(r'[^A-Za-z]+', ' ', lst_summary[i])
for i in range(len(test_text)):
    test_text[i] = re.sub(r'[^A-Za-z]+', ' ', test_text[i])
lst_text[0:5]
#Removing Stop Words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        if not r in stop_words:
            summary_filtered.append(r)
    lst_summary[i] = ' '.join(summary_filtered)

for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    test_text[i] = ' '.join(text_filtered)
#Lets now stem each word.
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        text_filtered.append(str(stemmer.stem(r)))
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        summary_filtered.append(str(stemmer.stem(r)))
    lst_summary[i] = ' '.join(summary_filtered)

for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(str(stemmer.stem(r)))
    test_text[i] = ' '.join(text_filtered)
lst_text[0:5]
test_text[0:5]
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vect = CountVectorizer()
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
X_train_dtm = vect.fit_transform(lst_text)
# Numpy arrays are easy to work with, so convert the result to an
# array
#train_data_features = train_data_features.toarray()
# examine the document-term matrix
# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(test_text)
X_test_dtm


def build_naive_model():
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_dtm, train.rating)
    # make class predictions for X_test_dtm
    y_pred_class_nb = nb.predict(X_test_dtm)
    # calculate accuracy of class predictions
    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_nb)
    precession  = metrics.precision_score(test.rating, y_pred_class_nb)
    recall = metrics.recall_score(test.rating, y_pred_class_nb)
    f1_score = metrics.f1_score(test.rating, y_pred_class_nb)
    print("NB Accuracy" ,accuracy, precession, recall, f1_score)
    return format(accuracy, ".2f"), precession, recall, f1_score

def build_logistic_model():
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression()
    lg.fit(X_train_dtm, train.rating)
    # make class predictions for X_test_dtm
    y_pred_class_lg = lg.predict(X_test_dtm)
    # calculate accuracy of class predictions
    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_lg)
    precession  = metrics.precision_score(test.rating, y_pred_class_lg)
    recall = metrics.recall_score(test.rating, y_pred_class_lg)
    f1_score = metrics.f1_score(test.rating, y_pred_class_lg)
    print("LG Accuracy" ,accuracy, precession, recall, f1_score)
    return accuracy, precession, recall, f1_score


def build_svm_model():
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train_dtm, train.rating)
    # make class predictions for X_test_dtm
    y_pred_class_svm = svm.predict(X_test_dtm)
    # calculate accuracy of class predictions
    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_svm)
    precession  = metrics.precision_score(test.rating, y_pred_class_svm)
    recall = metrics.recall_score(test.rating, y_pred_class_svm)
    f1_score = metrics.f1_score(test.rating, y_pred_class_svm)
    print("SVM Accuracy" ,accuracy, precession, recall, f1_score)
    return accuracy, precession, recall, f1_score



def build_random_forest_model():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train_dtm, train.rating)
    # make class predictions for X_test_dtm
    y_pred_class_rf= rf.predict(X_test_dtm)
    # calculate accuracy of class predictions
    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_rf)
    precession  = metrics.precision_score(test.rating, y_pred_class_rf)
    recall = metrics.recall_score(test.rating, y_pred_class_rf)
    f1_score = metrics.f1_score(test.rating, y_pred_class_rf)
    print("RF Accuracy" ,accuracy, precession, recall, f1_score)
    return accuracy, precession, recall, f1_score

def build_decision_tree_model():
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train_dtm, train.rating)
    y_pred_class_rf= dt.predict(X_test_dtm)

    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_rf)
    precession  = metrics.precision_score(test.rating, y_pred_class_rf)
    recall = metrics.recall_score(test.rating, y_pred_class_rf)
    f1_score = metrics.f1_score(test.rating, y_pred_class_rf)
    print("DT Accuracy" ,accuracy, precession, recall, f1_score)
    return accuracy, precession, recall, f1_score

def build_neural_network_model():
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(random_state=1)
    nn.fit(X_train_dtm, train.rating)
    y_pred_class_rf= nn.predict(X_test_dtm)

    from sklearn import metrics
    accuracy = metrics.accuracy_score(test.rating, y_pred_class_rf)
    precession  = metrics.precision_score(test.rating, y_pred_class_rf)
    recall = metrics.recall_score(test.rating, y_pred_class_rf)
    f1_score = metrics.f1_score(test.rating, y_pred_class_rf)
    print("NN Accuracy" ,accuracy, precession, recall, f1_score)
    return accuracy, precession, recall, f1_score

