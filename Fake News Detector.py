import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake.head()
data_true.head()

data_fake["class"] = 0
data_true["class"] = 1

data_fake.shape, data_true.shape

data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

data_fake = data_fake.iloc[:-10, :]
data_true = data_true.iloc[:-10, :]

data_fake.shape, data_true.shape

data_merge = pd.concat([data_fake, data_true], axis=0)

data_merge.head(10)

data_merge.columns

data = data_merge.drop(['title', 'subject', 'date'], axis=1)

data.isnull().sum()

data = data.sample(frac=1).reset_index(drop=True)

data.head()

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', "", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))