import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_auc_score
from utils import apply

def train(model, train_dataloader, val_dataloader, nb_epochs, N_training_data, device, text = False, plot = False):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters())

    training_results = []

    model.eval()
    y_val, y_pred_val = apply(model, val_dataloader, device)
    if len(y_pred_val[0]) == 2:
        y_pred_val = [prob[1] for prob in y_pred_val]
    roc_auc = roc_auc_score(y_val, y_pred_val, multi_class='ovo')
    
    if text:
        print('Initial roc_auc_score (validation):', roc_auc)
    
    model.train()

    for epoch in range(nb_epochs):
        epoch_average_loss = 0.0
        
        for X, y in train_dataloader:
            X = X.to(device)
            y_pred_train = model(X)
            loss = criterion(y_pred_train,y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * y.shape[0] / N_training_data
        
        model.eval()
        y_val, y_pred_val = apply(model, val_dataloader, device)
        if len(y_pred_val[0]) == 2:
            y_pred_val = [prob[1] for prob in y_pred_val]
        roc_auc = roc_auc_score(y_val, y_pred_val, multi_class='ovo')
        model.train()

        if ((epoch+1)%1 == 0) and text:
                print('Epoch [{}/{}], Loss_error (train): {:.4f}, Roc_auc_score (validation): {}'
                      .format(epoch+1, nb_epochs, epoch_average_loss, roc_auc))
                
        training_results.append(epoch_average_loss)
    
    if plot:
        plt.plot(training_results)
        plt.show()

    model.eval()
    
    return 