import numpy as np
from sklearn.preprocessing import StandardScaler

def flatten_data(X):
    """
    Convert:
        samples x channels x frequency
    
    into:
        samples x features
    """
    return X.reshape(X.shape[0],-1)

def fit_scaler(X_train):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    return X_train_scaled,scaler

def transform_data(X,scaler):
    return scaler.transform(X)

def convert_to_float32(X):
    return X.astype(np.float32)