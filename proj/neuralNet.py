# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:56:29 2023

@author: hp
"""
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

def neuralNet(option, trainData, trainLabel):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(trainData.shape[1],)))
    if option == 'ANN':
        model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
    elif option == 'DNN':
        model.add(tf.keras.layers.Dense(units = 256, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.1))
        
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.fit(x=trainData, y=trainLabel, batch_size=100, epochs = 10)
    return model

# This functions uses ANN as model
def ANN(train, valid, labels):
    # We want to save the prediction result that we had for the validation part for later analysis.
    # We shall initialize a dataframe to save the prediction, and the index of the frame should be
    # be identical to the validation data's session id (since each user had different input data for each 18 questions)
    # We'd like a frame with lenth of the unique session id list.
    prediction = pd.DataFrame(data = np.zeros((len(valid.index.unique()), 18)), index = valid.index.unique())

    # And we will need a dictiopnary to save models and evaluations for ANN and DNN models
    models = {}
    evaluations = {}
    # In this dataset, after finishing level 0-4, question 1-3 is answered, and
    # thus adopting the same pattern, we use level 5-12 to train question 4-13
    # and 13-22 to train question 14-18
    # and we train 18 models for each question
    for q in range(1, 19):
        if q <= 3:
            group = '0-4'
        elif q <= 13:
            group = '5-12'
        elif q <= 22:
            group = '13-22'
        print(f"===================This is ANN training of question {q} and group {group}===================")
        
        trainX = train.loc[train.level_group == group]
        validX = valid.loc[valid.level_group == group]
        
        trainY = labels.loc[labels.q == q].set_index('session').loc[trainX.index.values]
        validY = labels.loc[labels.q == q].set_index('session').loc[validX.index.values]
        
        # level group is not needed
        trainX = trainX.loc[:, trainX.columns != 'level_group']
        validX = validX.loc[:, validX.columns != 'level_group']
        
        # copy the data
        trainXminmax = trainX.copy()
          
        # apply normalization for level and page
        for column in ['level', 'page']:
            trainXminmax[column] = (trainXminmax[column] - trainXminmax[column].min()) / (trainXminmax[column].max() - trainXminmax[column].min())
        
        trainData = trainXminmax.to_numpy()
        trainLabels = trainY['correct'].to_numpy()
        
        validData = validX.to_numpy()
        validLabels = validY['correct'].to_numpy()
        
        model = neuralNet('ANN', trainData, trainLabels)
        
        models[f'{group}_{q}'] = model
        
        evaluation = model.evaluate(x=validData, y=validLabels, return_dict=True)
        evaluations[q] = []        
        evaluations[q].append(evaluation["accuracy"])
        
        # Use the trained model to make predictions on the validation dataset and 
        # store the predicted values in the `prediction_df` dataframe.
        predict = model.predict(x=validData)
        predict = np.round(predict, 0)
        f1_score = tfa.metrics.F1Score(num_classes=2, average='micro')
        f1_score.update_state(validLabels, predict)
        evaluations[q].append(f1_score.result().numpy())
        prediction.loc[validX.index.values, q-1] = predict.flatten()
    return prediction, models, evaluations

# This functions uses DNN as model
def DNN(train, valid, labels):
    # We want to save the prediction result that we had for the validation part for later analysis.
    # We shall initialize a dataframe to save the prediction, and the index of the frame should be
    # be identical to the validation data's session id (since each user had different input data for each 18 questions)
    # We'd like a frame with lenth of the unique session id list.
    prediction = pd.DataFrame(data = np.zeros((len(valid.index.unique()), 18)), index = valid.index.unique())

    # And we will need a dictiopnary to save models and evaluations for ANN and DNN models
    models = {}
    evaluations = {}
    # In this dataset, after finishing level 0-4, question 1-3 is answered, and
    # thus adopting the same pattern, we use level 5-12 to train question 4-13
    # and 13-22 to train question 14-18
    # and we train 18 models for each question
    for q in range(1, 19):
        if q <= 3:
            group = '0-4'
        elif q <= 13:
            group = '5-12'
        elif q <= 22:
            group = '13-22'
        print(f"===================This is DNN training of question {q} and group {group}===================")
        
        trainX = train.loc[train.level_group == group]
        validX = valid.loc[valid.level_group == group]
        
        trainY = labels.loc[labels.q == q].set_index('session').loc[trainX.index.values]
        validY = labels.loc[labels.q == q].set_index('session').loc[validX.index.values]
        
        # level group is not needed
        trainX = trainX.loc[:, trainX.columns != 'level_group']
        validX = validX.loc[:, validX.columns != 'level_group']
        
        # copy the data
        trainXminmax = trainX.copy()
          
        # apply normalization for level and page
        for column in ['level', 'page']:
            trainXminmax[column] = (trainXminmax[column] - trainXminmax[column].min()) / (trainXminmax[column].max() - trainXminmax[column].min())
        
        trainData = trainXminmax.to_numpy()
        trainLabels = trainY['correct'].to_numpy()
        
        validData = validX.to_numpy()
        validLabels = validY['correct'].to_numpy()
        
        model = neuralNet('DNN', trainData, trainLabels)
        
        models[f'{group}_{q}'] = model
        
        evaluation = model.evaluate(x=validData, y=validLabels, return_dict=True)
        evaluations[q] = []        
        evaluations[q].append(evaluation["accuracy"])
        
        # Use the trained model to make predictions on the validation dataset and 
        # store the predicted values in the `prediction_df` dataframe.
        predict = model.predict(x=validData)
        predict = np.round(predict, 0)
        f1_score = tfa.metrics.F1Score(num_classes=2, average='micro')
        f1_score.update_state(validLabels, predict)
        evaluations[q].append(f1_score.result().numpy())
        prediction.loc[validX.index.values, q-1] = predict.flatten()
    return prediction, models, evaluations