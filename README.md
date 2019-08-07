# SMS-Spam-Classification
This project deals with classification of SMS messages as spam or ham using Naive Bayes classifier

# Datasets
There are two datasets used in this project:
1. UCI SMS Spam Dataset (in English)
2. Precog IIITD SMSAssassin Dataset  (in Hindi)

This project aims to be a stepping stone towards building a universal SMS Spam Classifier regardless of the language used. To accomplish this task, I used two datasets in different languages.
# Testing
Preliminary testing was done on the two datasets individually to set a baseline for comparision.

Accuracies on testing set:

UCI Dataset       98.8%

IIITD Dataset     95.2%

Later on, different combinations were tried out. Their accuracies were as follows:

UCI train + IIITD test                        77.18%

IIITD	train + UCI test                        90.41%

Combined train + Combined	                    97.04%

(Combined - UCI test)train + 	UCI test	      98.49%

(Combined - IIITD test)train +	IIITD	        95.8%

UCI	+ (Combined - UCI train)	                85.99%

IIITD	+ (Combined - IIITD train)              90.72%


