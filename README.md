## Sentiment Analysis with LSTM
Sentiment analysis involves determining the sentiment or emotional tone expressed in a piece of text, typically categorized as positive, negative, or neutral. In this assignment, 
the task is to perform sentiment analysis using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN)

## Overview of the Assignment
The assignment focuses on classifying movie reviews into positive or negative sentiments using a dataset comprising 25,000 highly polar movie reviews for training and an additional 
25,000 for testing. The primary goal is to build a robust LSTM-based model capable of accurately discerning sentiments expressed in these reviews.

## What is LSTM?
LSTMs Long Short-Term Memory is a type of RNNs Recurrent Neural Network that can detain long-term dependencies in sequential data. LSTMs are able to process and analyze 
sequential data, such as time series, text, and speech. They use a memory cell and gates to control the flow of information, allowing them to selectively retain or discard
information as needed and thus avoid the vanishing gradient problem that plagues traditional RNNs. LSTMs are widely used in various applications such as natural language processing, 
speech recognition, and time series forecasting.

## Dataset Organization
The dataset, initially structured for binary sentiment classification, has been reorganized for practicality. Positive and negative reviews have been combined into separate
folders for both training and testing. This organizational choice simplifies the data handling process while retaining the essential characteristics of the sentiment analysis task.
The reviews were preprocessed by removing HTML tags and non-alphabetic characters. 

**Note:**  Due to the complexity of the Sentiment Analysis task and the substantial size of the dataset, the training and execution of the entire code may take a considerable amount of time. 
To expedite the demonstration and showcase the key functionalities, we have included pre-trained models and sample outputs in this documentation.

## Code Execution 
The following steps were performed:
1. Data Preprocessing
2. Tokenization and Padding
3. Lable Encoding
4. LSTM Model Architecture
5. Model Training
6. Model Evaluation
7. Model Evaluation

## Model Performance On Training Dataset
Accuracy: 87.45% <br>
Loss: 0.35

## Test Set
Accuracy: 84.10% <br>
Loss: 0.42

## Model Performance On Test Dataset
Accuracy: 83.00% <br>
Loss: 0.43

Total Predictions: 25000 <br>
Correct Predictions: 20752 <br>
Incorrect Predictions: 4248
