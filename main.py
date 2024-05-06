#importing libraries

import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import nltk
import re
import string
from nltk.stem import WordNetLemmatizer        

#reading the data

test_csv = pd.read_csv('/kaggle/input/imdb-movie-reviews-dataset/test_data (1).csv') 
train_csv = pd.read_csv('/kaggle/input/imdb-movie-reviews-dataset/train_data (1).csv') 

train_csv.head()

test_csv.head()
train_X_non = train_csv['0']   # '0' refers to the review text
train_y = train_csv['1']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test_csv['0']
test_y = test_csv['1']

train_X=[]
test_X=[]

#text pre processing
lemmatizer = WordNetLemmatizer()

for i in range(0, len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    train_X.append(review)
    
    
for i in range(0, len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    test_X.append(review)    

train_X[10]
test_X[10]

#tf idf

tf_idf = TfidfVectorizer(stop_words='english')
X_train_tf = tf_idf.fit_transform(train_X)
print(tf_idf.get_feature_names()[0:30])

X_train_tf = tf_idf.transform(train_X)

print(X_train_tf.toarray())

#transforming test data into tf-idf matrix

X_test_tf = tf_idf.transform(test_X)

#naive bayes classifier

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)
y_pred = naive_bayes_classifier.predict(X_test_tf)

from sklearn.metrics import confusion_matrix
conf_matrix = metrics.confusion_matrix(test_y, y_pred)
cm_display = ConfusionMatrixDisplay(conf_matrix).plot()

from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score


precision = 100*precision_score(test_y, y_pred)
recall = 100*recall_score(test_y, y_pred)
f1 = 100*f1_score(test_y, y_pred)
accuracy = 100*accuracy_score(test_y, y_pred)


label = [precision, recall , f1, accuracy]
values = ["Precision", "Recall" , "F1", "Accuracy"]
plt.bar(values, label, color ='blue',
        width = 0.4)


plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

prec, recall, _ = precision_recall_curve(test_y, y_pred)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay


fpr, tpr, _ = roc_curve(test_y, y_pred)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
