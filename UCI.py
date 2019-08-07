import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# To download the UCI dataset, follow the link :
#     https://archive.ics.uci.edu/ml/machine-learning-databases/00228/

#UCI Dataset
dfu = pd.read_table('/home/shlydv/Downloads/train', sep="\t", names=['label', 'sms'])

dfu['label'] = dfu.label.map({'ham':0, 'spam':1})


# To download the IIITD dataset, follow the link :
#     http://precog.iiitd.edu.in/requester.php?dataset=smsspam

#IIITD Dataset
dfh = pd.read_table('/home/shlydv/Downloads/ham.txt', names = ['sms'], encoding = "ISO-8859-1")
dfs = pd.read_table('/home/shlydv/Downloads/spam.txt', names = ['sms'], encoding = "ISO-8859-1")

dfh['label'] = 0
dfs['label'] = 1
# concatenating ham and spam
dfd = dfh.append(dfs, ignore_index=True)
# concatenating UCI and IIITD datasets
df = dfd.append(dfu, ignore_index=True)

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(df['sms'], df['label'], random_state=1)

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))


