#Naive Bayes ML algorithm for NLP modeling
#Implemented using sklearn module
#Sharan Pai, April 2023


# importing necessary libraries
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
import math

#Read file that will be used for training
#You can find the file
#https://www.kaggle.com/code/shubhamptrivedi/sentiment-analysis-on-imdb-movie-reviews/input
#The link has ~50,000 sentences, in which you can take a sample size out of it

#Then provide path to the training csv and validation csv here with two columns (review, sentiment)
#sentiment should be marked positive or negative
df = pd.read_csv("training_dataset_1000.csv")
df2 = pd.read_csv("validation_dataset.csv")

cv = CountVectorizer(max_features=800)

#Creating list of english stopwords
stopword_list = stopwords.words("english")

#Resetting index
df.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
#Sample dataset size
#print(df.shape)

#Naive Bayes Classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

df["sentiment"].replace({"positive":1, "negative":0}, inplace=True)
df2["sentiment"].replace({"positive":1, "negative":0}, inplace=True)

#Functions to remove noise
#Remove html tags
def clean_html(text):
 clean = re.compile("<.*?>")
 return re.sub(clean, "", text)

#Remove brackets
def remove_brackets(text):
 return re.sub("\[[^]]*\]", "", text)

#Convert all characters in text to lowercase
def lower_cases(text):
 return text.lower()

#Remove special characters
def remove_char(text):
 pattern = r"[^a-zA-z0–9\s]"
 text = re.sub(pattern, "", text)
 return text

#Remove noise(combine above functions)
def remove_noise(text):
 text = clean_html(text)
 text = remove_brackets(text)
 text = lower_cases(text) 
 text = remove_char(text) 
 return text

#Call the function on predictors
df["review"]=df["review"].apply(remove_noise)
df2["review"]=df2["review"].apply(remove_noise)

def stem_words(text):
 ps = PorterStemmer()
 stem_list = [ps.stem(word) for word in text.split()] 
 text = "".join(ps.stem(word) for word in text)
 
 return text

#Removing the stopwords from review
def remove_stopwords(text):
    # list to add filtered words from review
    filtered_text = []
        # verify & append words from the text to filtered_text list
    for word in text.split():
        if word not in stopword_list:
            filtered_text.append(word)
        # add content from filtered_text list to new variable
    clean_review = filtered_text[:]
        # emptying the filtered_text list for new review
    filtered_text.clear()
    return clean_review

#Join back all words as single paragraph
def join_back(text):
    return " ".join(text)

df2["review"] = df2["review"].apply(stem_words)
df2["review"] = df2["review"].apply(remove_stopwords)

df2["review"] = df2["review"].apply(join_back)

start_vectorize_time = time.time()
df["review"] = df["review"].apply(stem_words)

df["review"]=df["review"].apply(remove_stopwords)

df["review"] = df["review"].apply(join_back)
print(df.head())

#Vectorizing words and storing in variable X(predictor)
X = cv.fit_transform(df["review"]).toarray()
y = df.iloc[:,-1].values

#Split the Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

stop_vectorize_time = time.time()

total_vectorize_time = stop_vectorize_time - start_vectorize_time


#Fitting and predicting the model
gnb_start_time = time.time()
gnb.fit(X_train, y_train)
gnb_stop_time = time.time()
gnb_train_time = gnb_stop_time - gnb_start_time

y_pred_gnb = gnb.predict(X_test)

mnb_start_time = time.time()
mnb.fit(X_train, y_train)
mnb_end_time = time.time()
mnb_train_time = mnb_end_time - mnb_start_time

y_pred_mnb = mnb.predict(X_test)

bnb_start_time = time.time()
bnb.fit(X_train, y_train)
bnb_end_time = time.time()
bnb_train_time = bnb_end_time - bnb_start_time

y_pred_bnb = bnb.predict(X_test)
#Accuracy scores
print("Gaussian", accuracy_score(y_test, y_pred_gnb))
print("Multinomial", accuracy_score(y_test, y_pred_mnb))
print("Bernoulli", accuracy_score(y_test, y_pred_bnb))

print(f"\nGaussian total runtime: {gnb_train_time+total_vectorize_time}")
print(f"Multinomial total runtime: {mnb_train_time+total_vectorize_time}")
print(f"Bernoulli total runtime: {bnb_train_time+total_vectorize_time}")

#Gaussian scores
f1_gnb = f1_score(y_test,y_pred_gnb)
precision_gnb = precision_score(y_test,y_pred_gnb)
recall_gnb = recall_score(y_test,y_pred_gnb)

print('\n---GNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_gnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_gnb*100.))
print('       F1 score: {:2.1f}%'.format(f1_gnb*100.))

#Multinomial scores
f1_mnb = f1_score(y_test,y_pred_mnb)
precision_mnb = precision_score(y_test,y_pred_mnb)
recall_mnb = recall_score(y_test,y_pred_mnb)

print('\n---MNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_mnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_mnb*100.))
print('       F1 score: {:2.1f}%'.format(f1_mnb*100.))

#Bernoulli scores
f1_bnb = f1_score(y_test,y_pred_bnb)
precision_bnb = precision_score(y_test,y_pred_bnb)
recall_bnb = recall_score(y_test,y_pred_bnb)

print('\n---BNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_bnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_bnb*100.))
print('       F1 score: {:2.1f}%'.format(f1_bnb*100.))

print('\n\n\n===================================================================\n\n\n')
answer_array = df2['sentiment'].values
X2 = cv.fit_transform(df2["review"]).toarray()
mnb_predict = mnb.predict(X2)
gnb_predict = gnb.predict(X2)
bnb_predict = bnb.predict(X2)

#Gaussian scores
f1_gnb = f1_score(answer_array,gnb_predict)
precision_gnb = precision_score(answer_array,gnb_predict)
recall_gnb = recall_score(answer_array,gnb_predict)

print('\n---GNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_gnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_gnb*100.))
print('       F1 score: {:2.1f}%\n'.format(f1_gnb*100.))

#Multinomial scores
f1_mnb = f1_score(answer_array,mnb_predict)
precision_mnb = precision_score(answer_array,mnb_predict)
recall_mnb = recall_score(answer_array,mnb_predict)

print('\n---MNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_mnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_mnb*100.))
print('       F1 score: {:2.1f}%\n'.format(f1_mnb*100.))

#Bernoulli scores
f1_bnb = f1_score(answer_array,bnb_predict)
precision_bnb = precision_score(answer_array,bnb_predict)
recall_bnb = recall_score(answer_array,bnb_predict)

print('\n---BNB---\n')
print('\nPrecision score: {:2.1f}%'.format(precision_bnb*100.))
print('   Recall score: {:2.1f}%'.format(recall_bnb*100.))
print('       F1 score: {:2.1f}%\n'.format(f1_bnb*100.))


print("Gaussian:", accuracy_score(answer_array, gnb_predict))
print("Multinomial:", accuracy_score(answer_array, mnb_predict))
print("Bernoulli:", accuracy_score(answer_array, bnb_predict))

rmse = math.sqrt(mean_squared_error(answer_array, bnb_predict))
print('       RMSE: {:2.1f}'.format(rmse))

gnb_valacc_score = accuracy_score(answer_array, gnb_predict)
mnb_valacc_score = accuracy_score(answer_array, mnb_predict)
bnb_valacc_score = accuracy_score(answer_array, bnb_predict)
list_scores = [gnb_valacc_score, mnb_valacc_score, bnb_valacc_score]

from sklearn.metrics import confusion_matrix
cm_gnb = confusion_matrix(answer_array, gnb_predict)
cm_mnb = confusion_matrix(answer_array, mnb_predict)
cm_bnb = confusion_matrix(answer_array, bnb_predict)
print('gnb: \n',cm_gnb,'\n')
print('mnb: \n',cm_mnb,'\n')
print('bnb: \n',cm_bnb,'\n')

import seaborn as sns
import matplotlib.pyplot as plt
#cm_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
#cm_labels = np.asarray(cm_labels).reshape(2,2)

cm_matrix_gnb = pd.DataFrame(data=cm_gnb, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predicted Positive:1', 'Predicted Negative:0'])

cm_matrix_mnb = pd.DataFrame(data=cm_mnb, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predicted Positive:1', 'Predicted Negative:0'])

cm_matrix_bnb = pd.DataFrame(data=cm_bnb, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predicted Positive:1', 'Predicted Negative:0'])

if max(list_scores) == gnb_valacc_score:
    sns.heatmap(cm_matrix_gnb, annot=True, fmt='d', cmap='YlGnBu')
elif max(list_scores) == mnb_valacc_score:
    sns.heatmap(cm_matrix_mnb, annot=True, fmt='d', cmap='YlGnBu')
elif max(list_scores) == bnb_valacc_score:
    sns.heatmap(cm_matrix_bnb, annot=True, fmt='d', cmap='YlGnBu')
plt.show()



