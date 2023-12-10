# Pai_NWSE-MS-ML-models-2023
Multi-purpose NLP ML Model that can perform sentiment analysis/text classification  
Purpose of code: To create a model that performs sentiment analysis with a high accuracy, and use it for different use cases  
## Data Sets  
There are 3 data sets in csv files that are used to train the models:  
1. training_dataset_1000.csv --> Contains 1000 "Data"
2. training_dataset_10000.csv --> Larger version of previous file, containing 10000 "Data"
3. validation_dataset.csv --> Contains additional data  
## Models  
I used 3 models for my project. I have listed two here, as the third is too large for my computer to handle:
1. nlp_bi_lstm_model_v1.py --> This code is meant to be executed, and uses the neural network model Bi-directional LSTM (Long Short Term Memory) for training and testing
2. nlp_naive_bayes_model_v1.py --> This code is meant to be executed. It contains 3 different alorithms, Bernoulli, Gaussian, and Multinomial. They are each outputted for comparison
The data used to train these models go through pre-processing, which is specified in the code
Running these models will take a 5-10 minutes, as their size and datasets are massive.
