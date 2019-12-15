#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337)
import itertools

import matplotlib.pyplot as plt
import seaborn as sns


def plot_accuracy(history, width=15, height=15):
    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.figure(figsize=(width, height))
    plt.show()
    

def plot_loss(history, width=15, height=15):
    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.figure(figsize=(width, height))
    plt.show()
    
    
def plot_confusion_matrix(cm, classes,
                          group=None,
                          class_dict=None,
                          show_values=True,
                          show_ticks=True,
                          width=15, height=15,
                          normalize=False,
						  normalize_float='.2f',
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass
    
    if class_dict:
        rev_class_dict = {value: key for key, value in class_dict.items()}
        classes = [rev_class_dict[i] for i in classes]
    
    plt.figure(figsize=(width, height))
    plt.rcParams.update({'font.size': 10})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    #plt.axhline(y, linestyle='--', color='k') # horizontal lines
    #plt.axvline(x, linestyle='--', color='k') # vertical lines
    
    tick_marks = np.arange(len(classes))
    
    if show_ticks:
        if class_dict:
            plt.xticks(tick_marks, classes, rotation=45)
        else:
            plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

    fmt = normalize_float if normalize else 'd'
    thresh = cm.max() / 2.
    if show_values:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    if group:
        for i in range(group, len(classes), group):
            plt.hlines(y=i-0.5, xmin=-0.5, xmax=len(classes)-0.5, linestyle='--', linewidth=1)
            plt.vlines(x=i-0.5, ymin=-0.5, ymax=len(classes)-0.5, linestyle='--', linewidth=1)
            
            #plt.hlines(y=8.5, xmin=-0.5, xmax=len(classes)-0.5, linestyle='--', linewidth=1)
            #plt.hlines(y=17.5, xmin=-0.5, xmax=35.5, linestyle='--', linewidth=1)
            #plt.hlines(y=26.5, xmin=-0.5, xmax=35.5, linestyle='--', linewidth=1)
            #plt.vlines(x=8.5, ymin=-0.5, ymax=len(classes)-0.5, linestyle='--', linewidth=1)
            #plt.vlines(x=17.5, ymin=-0.5, ymax=35.5, linestyle='--', linewidth=1)
            #plt.vlines(x=26.5, ymin=-0.5, ymax=35.5, linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')