import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# To download the dataset, follow the link :
#     https://archive.ics.uci.edu/ml/machine-learning-databases/00228/

#UCI Dataset
df = pd.read_table('/home/shlydv/Downloads/train', sep="\t", names=['label', 'sms'])

df['label'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['sms'], df['label'], random_state=1)

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))


