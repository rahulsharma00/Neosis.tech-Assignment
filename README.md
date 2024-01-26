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
1. `Data Preprocessing` - The reviews were tokenized, converted to lowercase, and stop words were removed. The processed data was then split into training and testing sets.
2. `Tokenization and Padding` - The text data was tokenized using the Keras Tokenizer, and sequences were padded to ensure consistent input dimensions for the LSTM model.
3. `Lable Encoding` - This was used to convert categorical labels into numerical format,
4. `LSTM Model Architecture` - The LSTM model was constructed using Keras. It consists of an Embedding layer, SpatialDropout1D layer, LSTM layer, and a Dense layer with a sigmoid 
    activation function for binary classification.
5. `Model Training` - The model was trained on the training set using binary crossentropy loss and the Adam optimizer. The training process involved five epochs.
6. `Model Evaluation` - The model was evaluated on both the training and test sets to assess its performance. The accuracy and loss metrics were calculated, and predictions were made on the test set.
7. `Confusion Matrix and Classification Report` - The confusion matrix and classification report were generated using scikit-learn to provide insights into the model's performance, including precision, recall, and F1-score.

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

## Classification Report
The classification report summarizes the model's performance on a new test set, indicating precision, recall, and F1-score for positive (1) and negative (0) sentiments. With an accuracy of 83%, the model demonstrates balanced effectiveness in sentiment analysis, achieving strong precision, recall, and F1-score for both sentiment classes.
