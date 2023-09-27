# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:08:57 2023

@author: Alfred, Kylie
"""
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
from neuralNet import neuralNet

# This functions uses CART as model
def CART(train, valid, labels):
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
        print(f"===================This is cart training of question {q} and group {group}===================")
        
        trainX = train.loc[train.level_group == group]
        validX = valid.loc[valid.level_group == group]
        
        trainY = labels.loc[labels.q == q].set_index('session').loc[trainX.index.values]
        validY = labels.loc[labels.q == q].set_index('session').loc[validX.index.values]
        
        trainX.loc[:, "correct"] = trainY["correct"]
        validX.loc[:, "correct"] = validY["correct"]
        
        # level group is not needed
        trainX = trainX.loc[:, trainX.columns != 'level_group']
        validX = validX.loc[:, validX.columns != 'level_group']
        
        # copy the data
        trainXminmax = trainX.copy()
          
        # apply normalization for level and page
        for column in ['level', 'page']:
            trainXminmax[column] = (trainXminmax[column] - trainXminmax[column].min()) / (trainXminmax[column].max() - trainXminmax[column].min())
    
        trainData = tfdf.keras.pd_dataframe_to_tf_dataset(trainXminmax, label="correct", task = tfdf.keras.Task.CLASSIFICATION)
        validData = tfdf.keras.pd_dataframe_to_tf_dataset(validX, label="correct", task = tfdf.keras.Task.CLASSIFICATION)
        
        model = tfdf.keras.CartModel(verbose=0, task=tfdf.keras.Task.CLASSIFICATION)
        model.compile(metrics=["accuracy"])

        # Train the model.
        model.fit(x=trainData)
        
        models[f'{group}_{q}'] = model
        
        inspector = model.make_inspector()
        inspector.evaluation()
        evaluation = model.evaluate(x=validData, return_dict=True)
        evaluations[q] = evaluation["accuracy"]         

        # Use the trained model to make predictions on the validation dataset and 
        # store the predicted values in the `prediction_df` dataframe.
        predict = model.predict(x=validData)
        prediction.loc[validX.index.values, q-1] = predict.flatten()
    return prediction, models, evaluations

# This functions uses ANN&CART as model
def annCART(train, valid, labels):
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
        print(f"===================This is ANN+CART training of question {q} and group {group}===================")
        
        trainX = train.loc[train.level_group == group]
        validX = valid.loc[valid.level_group == group]
        
        trainY = labels.loc[labels.q == q].set_index('session').loc[trainX.index.values]
        validY = labels.loc[labels.q == q].set_index('session').loc[validX.index.values]
        
        trainX.loc[:, "correct"] = trainY["correct"]
        validX.loc[:, "correct"] = validY["correct"]
        
        # level group is not needed
        trainX = trainX.loc[:, trainX.columns != 'level_group']
        validX = validX.loc[:, validX.columns != 'level_group']
        
        # copy the data
        trainXminmax = trainX.copy()
          
        # apply normalization for level and page
        for column in ['level', 'page']:
            trainXminmax[column] = (trainXminmax[column] - trainXminmax[column].min()) / (trainXminmax[column].max() - trainXminmax[column].min())
        
        # Prepare the validation dataset
        validData = tfdf.keras.pd_dataframe_to_tf_dataset(validX, label="correct", task = tfdf.keras.Task.CLASSIFICATION)
        
        annTrainData = trainXminmax.drop('correct',axis=1).to_numpy()
        annTrainLabels = trainY['correct'].to_numpy()
        
        annModel = neuralNet('ANN', annTrainData, annTrainLabels)
        print("==================ANN TRAINING FINISHED=================")
        annPredictLabels = np.round(annModel.predict(annTrainData),0)
        annPredictLabels = pd.DataFrame(annPredictLabels, columns = ['correct'])
        
        for i in range(len(annPredictLabels['correct'])):
            trainX.loc[i, 'correct'] = annPredictLabels.loc[i, "correct"]

        cartTrainData = tfdf.keras.pd_dataframe_to_tf_dataset(trainX, label="correct")
        
        model = tfdf.keras.CartModel(verbose=0, task=tfdf.keras.Task.CLASSIFICATION)
        model.compile(metrics=["accuracy"])

        # Train the model.
        print("==================COMMENCING CART MODEL=================")
        model.fit(x=cartTrainData)
        print("==================CART TRAINING FINISHED=================")
        models[f'{group}_{q}'] = model
        
        inspector = model.make_inspector()
        inspector.evaluation()
        evaluation = model.evaluate(x=validData, return_dict=True)
        evaluations[q] = evaluation["accuracy"]         

        # Use the trained model to make predictions on the validation dataset and 
        # store the predicted values in the `prediction_df` dataframe.
        predict = model.predict(x=validData)
        prediction.loc[validX.index.values, q-1] = predict.flatten()
    return prediction, models, evaluations

# This functions uses DNN&CART as model
def dnnCART(train, valid, labels):
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
        print(f"===================This is DNN+CART training of question {q} and group {group}===================")
        
        trainX = train.loc[train.level_group == group]
        validX = valid.loc[valid.level_group == group]
        
        trainY = labels.loc[labels.q == q].set_index('session').loc[trainX.index.values]
        validY = labels.loc[labels.q == q].set_index('session').loc[validX.index.values]
        
        trainX.loc[:, "correct"] = trainY["correct"]
        validX.loc[:, "correct"] = validY["correct"]
        
        # level group is not needed
        trainX = trainX.loc[:, trainX.columns != 'level_group']
        validX = validX.loc[:, validX.columns != 'level_group']
        
        # copy the data
        trainXminmax = trainX.copy()
          
        # apply normalization for level and page
        for column in ['level', 'page']:
            trainXminmax[column] = (trainXminmax[column] - trainXminmax[column].min()) / (trainXminmax[column].max() - trainXminmax[column].min())
        
        # Prepare the validation dataset
        validData = tfdf.keras.pd_dataframe_to_tf_dataset(validX, label="correct", task = tfdf.keras.Task.CLASSIFICATION)
        
        annTrainData = trainXminmax.drop('correct',axis=1).to_numpy()
        annTrainLabels = trainY['correct'].to_numpy()
        
        annModel = neuralNet('DNN', annTrainData, annTrainLabels)
        print("==================ANN TRAINING FINISHED=================")
        annPredictLabels = np.round(annModel.predict(annTrainData),0)
        annPredictLabels = pd.DataFrame(annPredictLabels, columns = ['correct'])
        
        for i in range(len(annPredictLabels['correct'])):
            trainX.loc[i, 'correct'] = annPredictLabels.loc[i, "correct"]

        cartTrainData = tfdf.keras.pd_dataframe_to_tf_dataset(trainX, label="correct")
        
        model = tfdf.keras.CartModel(verbose=0, task=tfdf.keras.Task.CLASSIFICATION)
        model.compile(metrics=["accuracy"])

        # Train the model.
        print("==================COMMENCING CART MODEL=================")
        model.fit(x=cartTrainData)
        print("==================CART TRAINING FINISHED=================")
        models[f'{group}_{q}'] = model
        
        evaluation = model.evaluate(x=validData, return_dict=True)
        evaluations[q] = evaluation["accuracy"]         

        # Use the trained model to make predictions on the validation dataset and 
        # store the predicted values in the `prediction_df` dataframe.
        predict = model.predict(x=validData)
        prediction.loc[validX.index.values, q-1] = predict.flatten()
    return prediction, models, evaluations