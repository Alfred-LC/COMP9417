#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:25:36 2023

@author: Alfred Lu
"""
# import all neccessary libraries
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np

"""
By looking into the provided dataset, we can identify following variables:
session_id - the ID of the session the event took place in
index - the index of the event for the session
elapsed_time - how much time has passed (in milliseconds) between the start of the session and when the event was recorded
event_name - the name of the event type
name - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)
level - what level of the game the event occurred in (0 to 22)
page - the page number of the event (only for notebook-related events)
room_coor_x - the coordinates of the click in reference to the in-game room (only for click events)
room_coor_y - the coordinates of the click in reference to the in-game room (only for click events)
screen_coor_x - the coordinates of the click in reference to the player’s screen (only for click events)
screen_coor_y - the coordinates of the click in reference to the player’s screen (only for click events)
hover_duration - how long (in milliseconds) the hover happened for (only for hover events)
text - the text the player sees during this event
fqid - the fully qualified ID of the event
room_fqid - the fully qualified ID of the room the event took place in
text_fqid - the fully qualified ID of the
fullscreen - whether the player is in fullscreen mode
hq - whether the game is in high-quality
music - whether the game music is on or off
level_group - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)
"""

# According to https://www.skytowner.com/explore/reducing_dataframe_memory_size_in_pandas 
# When using panda.read_csv() to import data from csv files, if user does not
# specify data types for each column, the default setting will set all integers
# to 'int 64', float number to 'float 64' and strings to 'object'. However, in
# in most practical cases, not all numericla data would neccessarily take up
# 'int 64' or 'float 64', we can downsize the dataset by redefining the datatype
# according to the max and minimum value. And the websote has suggested that we
# can set string variables to type 'category'

dataTypes={
            'elapsed_time':np.int32,
            'event_name':'category',
            'name':'category',
            'level':np.uint8,
            'room_coor_x':np.float32,
            'room_coor_y':np.float32,
            'screen_coor_x':np.float32,
            'screen_coor_y':np.float32,
            'hover_duration':np.float32,
            'text':'category',
            'fqid':'category',
            'room_fqid':'category',
            'text_fqid':'category',
            'fullscreen':'category',
            'hq':'category',
            'music':'category',
            'level_group':'category'
            }

# Read the data into the dataset
print("=============start to load training data============")
origDataset = pd.read_csv('predict-student-performance-from-game-play/train.csv', dtype=dataTypes)
print("=============Load Complete============")

# Read labels into dataset
print("=============start to load training data============")
labels = pd.read_csv('predict-student-performance-from-game-play/train_labels.csv')
print("=============Load Complete============")

# In the label data, each "session_id" is a combnination of session ID and question number
# we split the session_id into 'session' and 'q'
labels['session'] = labels.session_id.apply(lambda x: int(x.split('_')[0]) )
labels['q'] = labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )

# Now inorder to process the data and train the model, certain transformation should be performed.
# Among these columns of data, we can divided them into two main types, numerical data, categorical data
# And we should perform different data preprocessing on these different inputs.
categorical = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid', 'music', 'fullscreen', 'hq']
numerical = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']

# After dividing the data, we should think about how to process the data, such
# that it is more understandable for later on training.
# We adopt Chris Deotte's method to preprocess the data and 
# Reference: https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook
def feature_engineer(dataset):
    dfs = []
    for c in categorical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    for c in numerical:
        tmp = dataset.groupby(['session_id','level_group'])[c].agg('skew')
        tmp.name = tmp.name + '_skew'
        dfs.append(tmp)
    dataset = pd.concat(dfs,axis=1)
    dataset = dataset.fillna(-1)
    dataset = dataset.reset_index()
    dataset = dataset.set_index('session_id')
    return dataset

dataProcessed = feature_engineer(origDataset)

# Data split, we divide 80% data to be train data and 20% to be the validation part in default
def dataSplit(dataset, ratio=0.20):
    session = dataset.index.unique()
    numSession = int(len(session))
    numTrain =int(numSession * (1 - ratio))
    return dataset.loc[session[:numTrain]], dataset.loc[session[numTrain:]]

trainData, validData = dataSplit(dataProcessed)

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

# This function will print out the accuracy and F1 evalution for NN models,
# For DT models, it will provide average accuracy evaluation. 'NN' for neural netwaork models
# and 'DT' for decision Tree models.
def evaluationSummary(evaluations, model):
    if model == 'NN':
        accuracySum = 0
        f1ScoreSum = 0
        for name, value in evaluations.items():
            print(f"question {name}: accuracy {value[0]:.4f}")
            accuracySum += value[0]
            f1ScoreSum += value[1]
            
        print("\nAverage accuracy", accuracySum/18)
        print("\nAverage F1 score", f1ScoreSum/18)
    elif model == 'DT':
        for name, value in evaluations.items():
            print(f"question {name}: accuracy {value:.4f}")

        print("\nAverage accuracy", sum(evaluations.values())/18)
    else:
        print("Invalid model type")

# This will print out the variables accordong to the importance.
def checkInspector(models):
    ins = models['0-4_1'].make_inspector()
    print("Available variable importances:")
    for importance in ins.variable_importances().keys():
        print("\t", importance)
    for name, model in models.items():
        print(f"=======NUM AS ROOT INFO for model {name}======")
        inspector = model.make_inspector()
        for i in inspector.variable_importances()["NUM_AS_ROOT"]:
            print(i)
            
def cartF1(validData, labels, prediction):
    true = pd.DataFrame(data=np.zeros((len(validData.index.unique()),18)), index=validData.index.unique())
    for i in range(18):
        # Get the true labels.
        temp = labels.loc[labels.q == i+1].set_index('session').loc[validData.index.unique()]
        true[i] = temp.correct.values

    maxScore = 0; bestThresh = 0

    # Loop through threshold values from 0.4 to 0.8 and select the threshold with 
    # the highest `F1 score`.
    for thresh in np.arange(0.3,0.8,0.01):
        metric = tfa.metrics.F1Score(num_classes=2,average="macro",threshold=thresh)
        actual = tf.one_hot(true.values.reshape((-1)), depth=2)
        pred = tf.one_hot((prediction.values.reshape((-1))>thresh).astype('int'), depth=2)
        metric.update_state(actual, pred)
        f1 = metric.result().numpy()
        if f1 > maxScore:
            maxScore = f1
            bestThresh = thresh
    print("Best threshold ", bestThresh, "\tF1 score ", maxScore)
    
# Now we run ANN, DNN, CART, ANNCART, DNNCART
annPredict, annModels, annEvaluations = ANN(trainData, validData, labels)
dnnPredict, dnnModels, dnnEvaluations = DNN(trainData, validData, labels)
cartPredict, cartModels, cartEvaluations = CART(trainData, validData, labels)
annCartPredict, annCartModels, annCartEvaluations = annCART(trainData, validData, labels)
dnnCartPredict, dnnCartModels, dnnCartEvaluations = dnnCART(trainData, validData, labels)

# Show evaluation results
evaluationSummary(cartEvaluations,'DT')
checkInspector(cartModels)
cartF1(validData, labels, cartPredict)

evaluationSummary(annCartEvaluations,'DT')
checkInspector(annCartModels)
cartF1(validData, labels, annCartPredict)

evaluationSummary(dnnCartEvaluations,'DT')
checkInspector(dnnCartModels)
cartF1(validData, labels, dnnCartPredict)

evaluationSummary(annEvaluations,'NN')
evaluationSummary(dnnEvaluations,'NN')
