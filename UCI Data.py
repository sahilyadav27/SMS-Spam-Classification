import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#UCI Dataset
dfu = pd.read_table('/home/shlydv/Downloads/train', sep="\t", names=['label', 'sms'])


dfu['label'] = dfu.label.map({'ham':0, 'spam':1})

"""
#IIITD Dataset
dfh = pd.read_table('/home/shlydv/Downloads/ham.txt', names = ['sms'], encoding = "ISO-8859-1")
dfs = pd.read_table('/home/shlydv/Downloads/spam.txt', names = ['sms'], encoding = "ISO-8859-1")

dfh['label'] = 0
dfs['label'] = 1
dfd = dfh.append(dfs, ignore_index=True)


df = dfd.append(dfu, ignore_index=True)
"""
#X_train, y_train = dfd['sms'], dfd['label']
#X_test, y_test = dfu['sms'], dfu['label']
X_train, X_test, y_train, y_test = train_test_split(dfu['sms'], dfu['label'], random_state=1)

#X_train, X_test, y_train, y_test = train_test_split(dfd['sms'], dfd['label'], random_state=1)
#X_test = X_test.append(dfu['sms'], ignore_index = True)
#y_test = y_test.append(dfu['label'], ignore_index = True)
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))


