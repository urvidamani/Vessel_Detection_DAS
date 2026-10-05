import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix)

def predict_model(model, data_loader, device):
    model.eval()
    probabilities=[]
    predictions=[]
    actuals=[]
    
    with torch.no_grad():
        for X_batch,y_batch in data_loader:
            X_batch=X_batch.to(device)
            outputs=model(X_batch)
            probabilities_batch=torch.sigmoid(outputs)
            predictions_batch=(probabilities_batch>=0.5).int()
            probabilities.extend(probabilities_batch.cpu().numpy())
            predictions.extend(predictions_batch.cpu().numpy())
            actuals.extend(y_batch.numpy())
        
    return(np.array(actuals),np.array(predictions),np.array(probabilities))

def calculate_metrics(y_true,y_pred):
    y_true=np.asarray(y_true).astype(int)
    y_pred=np.asarray(y_pred).astype(int)
    accuracy=accuracy_score(y_true,y_pred)
    overall_f1=f1_score(y_true,y_pred,zero_division=0)
    report=classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    
    near_f1=report.get("1",{}).get("f1-score",0.0)
    far_f1=report.get("0",{}).get("f1-score",0.0)
    
    return{
        "accuracy":accuracy,
        "overall_f1":overall_f1,
        "near_f1":near_f1,
        "far_f1":far_f1
    }

def get_confusion_matrix(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)
            