#Bi-directional LSTM ML algorithm for NLP modeling
#Implemented using sklearn module, and TensorFlow Keras module
#Sharan Pai, April 2023

#Import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import time
from sklearn.utils import class_weight
from keras.utils.vis_utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt

#Set the number of epochs
epoch_count = 10

#Class CallbackTraining gives a summary of the training accuracy and loss over each epoch in training
class CallbackTraining(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_count = 0
        print("\nStarting the training!\n")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if (self.epoch_count % 5 == 0 or self.epoch_count == 1):
            print(f"\n========================EPOCH {self.epoch_count} OUT OF {epoch_count} SUMMARY========================\n\
Training loss: {logs['loss']}\n\
Training accuracy: {logs['accuracy']}")
        
#Read file that will be used for training
#You can find the file
#https://www.kaggle.com/code/shubhamptrivedi/sentiment-analysis-on-imdb-movie-reviews/input
#The link has ~50,000 sentences, in which you can take a sample size out of it (I chose 10,000)
#Then provide path to the training csv and validation csv here with two columns (review, sentiment)
#sentiment should be marked positive or negative
reviews = pd.read_csv('training_dataset_10000.csv')


reviews['sentiment'] = np.where(reviews['sentiment'] == 'positive', 1, 0)

#Create stopword list using NLTK Stopwords
stopword_list = stopwords.words("english")

#Define initial pre-processing functions

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

#Remove all special characters
def remove_char(text):
    pattern = r"[^a-zA-z0–9\s]"
    text = re.sub(pattern, "", text)
    return text

#Function remove_noise combines all minor pre-processing functions above
def remove_noise(text):
    text = clean_html(text)
    text = remove_brackets(text)
    text = lower_cases(text) 
    text = remove_char(text) 
    return text

#Stem all the words in the text
def stem_words(text):
    ps = PorterStemmer()
    stem_list = [ps.stem(word) for word in text.split()] 
    text = "".join(ps.stem(word) for word in text)
    return text

#Remove the stopwords in the text
def remove_stopwords(text):
    filtered_text = []

    #Verify and append words that are not stopwords into filtered_text
    for word in text.split():
        if word not in stopword_list:
            filtered_text.append(word)
            
    #Add the content from filtered_text to a new variable
    clean_review = filtered_text[:]
    
    #Empty filtered_text for a new review text
    filtered_text.clear()
    return clean_review

#Join back all words as single paragraph
def join_back(text):
    return " ".join(text)

#Apply all Pre-processing functions to the dataset
reviews['review'] = reviews['review'].apply(remove_noise)
reviews['review'] = reviews['review'].apply(stem_words)
reviews['review'] = reviews['review'].apply(remove_stopwords)
reviews['review'] = reviews['review'].apply(join_back)

#Combine the sentiment and the sentences into arrays
sentences = reviews['review'].to_numpy()
labels = reviews['sentiment'].to_numpy()

#Split the Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25)

#Setting oov_token (Out Of Vocabulary words), and a tokenizer
oov_tok = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_tok)

#Fit the training data to get it ready for a transformation into a sequence
tokenizer.fit_on_texts(X_train)

#Convert the sentences of X_train into numbered-sequences for training
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

vocab_size = vocab_size = len(tokenizer.word_index)+1
sequence_length = 560

#Pad the sequences such that all sequences have the same length (truncate if too long)
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')

#Define the model that will be used for training
model = Sequential()
embedding_dim = 32
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))
lstm_out = 64

model.add(SpatialDropout1D(0.25))

#Add the main (Bi-directional) LSTM layers
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2,recurrent_dropout=0.2,
                             activation='tanh')))

model.add(Dropout(0.1))

#Add another Dense layer with activation sigmoid
#Sigmoid takes in real value inputs and outputs a number ranging from 0 to 1
model.add(Dense(1, activation='sigmoid'))

#Compile the loss which is calculated via binary crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train                                                    
                                    )

class_weights_dict = dict(zip(np.unique(y_train), class_weights))

#Train the model
history = model.fit(train_padded, y_train, epochs=epoch_count,
                    validation_data=(test_padded, y_test),
                    batch_size=64, verbose=0, callbacks=CallbackTraining(),
                    class_weight=class_weights_dict)

#Pring the metrics, and the summary
metrics_df = pd.DataFrame(history.history)
print('\n')
print(metrics_df)
print('\n')
print(model.summary())
print('\n')


#Saving the model as LSTM_10k_model in a folder called LSTM_models
model.save('LSTM_models/LSTM_10k_model.h5')


#Predict the test_padded
predict_start_time = time.time()
y_prediction_decimal = model.predict(test_padded, verbose=0)
predict_end_time = time.time()
total_time = predict_end_time-predict_start_time

#Print how long it took to predict the text
print(f'\nTime it took to predict: {total_time}\n')

sentiment_list_predicted = []

#Takes in percentage (as decimal) and checks if output is greater than or less than 0.5
#(1 = positive), (0 = negative)
for decimal_prediction in y_prediction_decimal:
    if decimal_prediction > 0.5:
        sentiment_list_predicted.append(1)
    elif decimal_prediction < 0.5:
        sentiment_list_predicted.append(0)

#Converting answer (array) to list to check accuracy
converted_list_answers = []
for answer in range(len(y_test)):
    converted_list_answers.append(y_test[answer])

#If length of answers is not equal to length of predictions, raise value error
if len(converted_list_answers) != len(sentiment_list_predicted):
    raise ValueError("Both the length of converted_list_answers and the length of \
sentiment_list_predicted must be the same.")

#Convert sentiment_list_predicted into array for plotting
y_prediction = np.array(sentiment_list_predicted)

correct_score = 0
wrong_score = 0
total = len(converted_list_answers)

#Checking accuracy
for sentiment in range(len(converted_list_answers)):
    if str(converted_list_answers[sentiment]) == str(sentiment_list_predicted[sentiment]):
        correct_score += 1
    else:
        wrong_score += 1

#Calculate validation accuracy
validation_accuracy = correct_score/total

#Make plots (Confusion matrices, and matplotlib plots)
cm = confusion_matrix(y_test, y_prediction)


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

f1 = f1_score(y_test,y_prediction)
precision = precision_score(y_test,y_prediction)
recall = recall_score(y_test,y_prediction)

#Print Precision, Recall, and F1 score
print('Precision score: {:2.1f}%'.format(precision*100.))
print('   Recall score: {:2.1f}%'.format(recall*100.))
print('       F1 score: {:2.1f}%'.format(f1*100.))

#Make Matplotlib plots
plt.figure(figsize=(10,5))
plt.plot(metrics_df.index, metrics_df.loss)
plt.plot(metrics_df.index, metrics_df.val_loss)
plt.title('Sentiment Analysis Model Loss over Epochs')
plt.xlabel('epochs')
plt.ylabel('loss - Binary Crossentropy')
plt.legend(['Training Loss', 'Validation Loss'])
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(metrics_df.index, metrics_df.accuracy)
plt.plot(metrics_df.index, metrics_df.val_accuracy)
plt.title('Sentiment Analysis Model Accuracy over Epochs')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(metrics_df.index, metrics_df.val_loss,color='red')
plt.plot(metrics_df.index, metrics_df.val_accuracy,color='green')
plt.title('Sentiment Analysis Model Loss over Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['Validation Loss', 'Validation Accuracy'])
plt.grid()
plt.show()

