import numpy as np
import torch

from sklearn import datasets, preprocessing
from torch.utils.data import TensorDataset

## Récupération et mise en forme de différents jeux de données (X, y)

def make_classification(n_samples=5000, n_features=2, n_redundant=0, n_informative=2,
                         n_classes=2):

    data = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                    n_redundant=n_redundant, n_informative=n_informative, n_classes=n_classes)

    X = data[0]
    y = data[1]
    
    return X, y

def fetch_covtype():

    fetch_covtype = datasets.fetch_covtype()

    perm = np.random.permutation(len(fetch_covtype.target))
    X = fetch_covtype.data.copy()[perm]
    y = fetch_covtype.target.copy()[perm]-1
    
    return X, y

## Normalisation et découpage en train, validation et test

def create_dataset(X, y, normalize = True, prop_val = 0.7, prop_test = 0.15):

    if normalize:
        mu_X = np.mean(X, axis = 0)
        std_X = np.std(X, axis = 0)
        X = (X-mu_X)/(std_X+1e-10)

    n_data = len(y)
    val_split = int(n_data*prop_val)
    test_split = n_data-int(n_data*prop_test)

    X_train = X[:val_split]
    X_val = X[val_split:test_split]
    X_test = X[test_split:]
    y_train = y[:val_split]
    y_val = y[val_split:test_split]
    y_test = y[test_split:]

    training_data = TensorDataset(torch.from_numpy(np.array(X_train)).float(), 
                            torch.from_numpy(np.array(y_train)).long())
    validation_data = TensorDataset(torch.from_numpy(np.array(X_val)).float(), 
                            torch.from_numpy(np.array(y_val)).long())
    test_data = TensorDataset(torch.from_numpy(np.array(X_test)).float(), 
                            torch.from_numpy(np.array(y_test)).long())

    return training_data, validation_data, test_data