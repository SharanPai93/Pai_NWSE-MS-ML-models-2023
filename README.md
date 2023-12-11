# Pai_NWSE-MS-ML-models-2023
Multi-purpose NLP ML Model that can perform sentiment analysis/text classification.  
Purpose of code: To create a model that performs sentiment analysis with a high accuracy, and use it for different use cases.  
## Data Sets  
There are 3 data sets in csv files that are used to train the models:  
1. "*training_dataset_1000.csv*" --> Contains 1000 "Data"
2. "*training_dataset_10000.csv*" --> Larger version of previous file, containing 10000 "Data"
3. "*validation_dataset.csv*" --> Contains additional data  
## Models  
I used 3 models for my project. I have listed two here, as the third is too large for my computer to handle:
1. "*nlp_bi_lstm_model_v1.py*" --> This code is meant to be executed, and uses the neural network model Bi-directional LSTM (Long Short Term Memory) for training and testing
2. "*nlp_naive_bayes_model_v1.py*" --> This code is meant to be executed. It contains 3 different alorithms, Bernoulli, Gaussian, and Multinomial. They are each outputted for comparison
The data used to train these models go through pre-processing, which is specified in the code.
  
Running these models will take a 5-10 minutes, as their size and datasets are massive.
## Counseling App
Additionally, I have posted a code that creates an prototype-App using the Python Module Tkinter. This app has two main interfaces --> The **Counselor View**, and the **Student View**:
### Student View
The **Student View** is the first interface that is shown on when the code is executed. It consists of a few textboxts which take in input, and the information gathered is relayed to the counselor via the **Counselor View**. The information is relayed to the counselor via the "Submit" button. The large text box is used for the student to explain their situation such that a counselor can help them accordingly. At the bottom of the **Student View**, there is a button that can be used to switch to the **Counselor View**. This was created to switch between the two interfaces with relative ease as compared to the alternative, where we would need to run two different programs.
### Counselor View
In the **Counselor View**, the interface allows the user to switch between students who have submitted text, and analyze their situations accordingly. Additionally, I have added a language model to analyze the Students' texts and sort them via emotion, and intensity. The following emotions are listed below:  
1. Joy
2. Anger
3. Sad, Sorrow
4. Fear
5. Surprise
6. Love  
   
Of the emotions above, three are classified as *good*, and three are classified as *bad*. Along with the emotions, the language model predicts an intensity of the emotion, allowing for me to create an algorithm to provide the most needy students with a counselor at a higher priority than others.  
### Purpose
The purpose of this app was to conduct an experiment and simulate a real student-counselor situation in which an over-crowded school has to meet the needs of multiple students with few counselors to support them at the same time. Therefore, this app would be used to be an easy, reliable, and efficient way for the counselors to organize themselves, and treat the students accordingly.
## Notes
1. This project was created for research and experimental purposes only. This app was not tested on anyone.
2. This app remains a prototype, as the data can only be accessed locally.
3. The model used is called BERT, which stands for Bi-directional Encoder Representations from Transformers. This model was made by google, and was the third model I used for my project.
