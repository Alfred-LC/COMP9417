# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:01:40 2023

@author: Alfred, Kylie
"""

import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

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