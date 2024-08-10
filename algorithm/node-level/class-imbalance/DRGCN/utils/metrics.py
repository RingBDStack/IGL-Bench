import numpy as np

def evaluate(pred_true_labels):
#     print(len(pred_true_labels))
    pred_category = {}
    true_category = {}
    
    for i in pred_true_labels.keys():
        pred_label_i = pred_true_labels[i][0]
#         print(len(pred_label_i))
        true_label_i = pred_true_labels[i][1]
        
        for c1 in pred_label_i:
            if c1 in pred_category:
                pred_category[c1].append(i)
            else:
                pred_category[c1] = [i]
        for c2 in true_label_i:
            if c2 in true_category:
                true_category[c2].append(i)
            else:
                true_category[c2] = [i]
    accuracy_per_classes = dict()
    for key in true_category.keys():
        if key in pred_category.keys():
            total_samples = len(true_category[key])
            correct_samples = 0
            for id in pred_category[key]:
                if id in true_category[key]:
                    correct_samples += 1
            precision = correct_samples / total_samples;
            accuracy_per_classes[key] = float("{0:.2f}".format(precision))
        else:
            accuracy_per_classes[key] = .0    
#     print('true_category=',true_category)       
#     print('pred_category=',pred_category)       
    return accuracy_per_classes