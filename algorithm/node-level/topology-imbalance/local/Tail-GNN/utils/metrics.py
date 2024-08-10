import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def micro_f1(output, labels, index):

    label, count = np.unique(labels, return_counts=True)    
    most_freq = np.argmax(count)    
    index = [i for i in index if labels[i] != most_freq]

    preds = output.max(1)[1]

    return f1_score(labels[index], preds[index], average='micro')

def macro_f1(output, labels, index):

    label, count = np.unique(labels, return_counts=True)

    preds = output.max(1)[1]

    return f1_score(labels[index], preds[index], average='macro')

def bacc(output, labels, index):

    label, count = np.unique(labels, return_counts=True)

    preds = output.max(1)[1]

    return balanced_accuracy_score(labels[index], preds[index])

def roc_auc(output, labels, index):

    label, count = np.unique(labels, return_counts=True)
    one_hot_labels = label_binarize(labels, classes=label)
    detach_out = output[index].detach()

    return roc_auc_score(one_hot_labels[index], detach_out, multi_class="ovo")

