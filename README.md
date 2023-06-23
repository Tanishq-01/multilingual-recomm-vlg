# Session-Based Recommendation with RNN Baseline

This project implements a session-based recommendation system using a baseline RNN model. It focuses on providing recommendations based on sequential user interactions within sessions.

## Overview

Session-based recommendation systems aim to provide personalized recommendations based on user interactions within a session. In this project, we implement a baseline RNN model to capture the sequential patterns in session data and make item predictions.

## Approach

To train and evaluate the baseline RNN model, we followed the following approach:

1. Dataset Preparation:
   - We obtained the session data, which includes user interactions and corresponding item information.
   - Each session consists of a sequence of user interactions, where each interaction is associated with a product ID.
   - We performed preprocessing on the dataset, including removing irrelevant information, handling missing values, and ensuring the data is in the required format for model training.

2. Mapping Product IDs to Numbers:
   - As the RNN model takes numerical inputs, we created mappings from product IDs to numbers.
   - We assigned a unique number to each distinct product ID in the dataset.
   - This mapping allows us to represent each session as a sequence of numerical values, making it suitable for input to the RNN model.
   - The mapping is created during the dataset preprocessing step and is used during both training and evaluation phases.
  
3. Model Architecture:
   - The baseline model used in this project is an RNN-based model.
   - It takes session sequences as input and predicts the next item in the sequence.
   - The RNN model consists of multiple layers with a specified hidden size and number of layers.
   - We use an additional linear layer to map the RNN output to the predicted item.

4. Training:
   - We split the preprocessed dataset into training and testing sets.
   - During the training phase, we input the session sequences to the RNN model and optimize the model parameters using a specified optimizer and loss function.
   - We iterate over the training dataset for multiple epochs, updating the model weights to minimize the loss and improve the prediction performance.
   - Adjusting the hyperparameters, optimizer settings, and training configurations can be done to optimize the model's performance.  
