# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:51:51 2023

@author: Alfred Lu, Kylie
"""
from neuralNet import ANN, DNN
from cartModel import CART, annCART, dnnCART
from evaluation import evaluationSummary, checkInspector, cartF1
from featureEngine import feature_engineer, dataSplit

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

dataProcessed = feature_engineer(origDataset)

trainData, validData = dataSplit(dataProcessed)

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