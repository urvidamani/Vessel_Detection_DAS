import torch
import copy

def train_model(model,train_loader,criterion,optimizer,device,epochs=10,val_loader=None,patience=10):
    history={"train_loss":[],"val_loss":[]}
    best_val,best_state,wait=float("inf"),None,0
    
    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        for X_batch,y_batch in train_loader:
            X_batch,y_batch=X_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            loss=criterion(model(X_batch),y_batch)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        history["train_loss"].append(total_loss/len(train_loader))
        message=f"Epoch {epoch+1}/{epochs} | Train loss: {history['train_loss'][-1]:.4f}"
        
        if val_loader is not None:
            model.eval()
            val_total=0.0
            with torch.no_grad():
                for X_batch,y_batch in val_loader:
                    X_batch,y_batch=X_batch.to(device),y_batch.to(device)
                    val_total+=criterion(model(X_batch),y_batch).item()
            val_loss=val_total/len(val_loader)
            history["val_loss"].append(val_loss)
            message+=f" | Val loss: {val_loss:.4f}"
            
            if val_loss<best_val:
                best_val,best_state,wait=val_loss,copy.deepcopy(model.state_dict()),0
            else:
                wait+=1
                if wait>=patience:
                    print(message)
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(message)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

        