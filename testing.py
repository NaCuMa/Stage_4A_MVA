import torch.nn as nn

from sklearn.metrics import roc_auc_score
from utils import apply

def test(model, test_dataloader, N_test_data, device, text = True):
    
    model.eval()
    y_test, y_pred_test = apply(model, test_dataloader, device)
    if len(y_pred_test[0]) == 2:
         y_pred_test = [prob[1] for prob in y_pred_test]
    roc_auc = roc_auc_score(y_test, y_pred_test, multi_class='ovo')
    
    if text:
        print("Test roc_auc_score: ", roc_auc, ".")
        
    return roc_auc