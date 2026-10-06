import torch
def train_model(model,train_loader,criterion,optimizer,device,epochs=10,val_loader=None):
    history={"train_loss":[],"val_loss":[]}
    
    
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
        
        if val_loader is not None:
            model.eval()
            val_total=0.0
            with torch.no_grad():
                for X_batch,y_batch in val_loader:
                    X_batch,y_batch=X_batch.to(device),y_batch.to(device)
                    val_total+=criterion(model(X_batch),y_batch).item()
            history["val_loss"].append(val_total/len(val_loader))
        msg=f"Epoch {epoch+1}/{epochs} | Train: {history['train_loss'][-1]:.4f}"
        if val_loader is not None:
            msg+=f"| val: {history['val_loss'][-1]:.4f}"
        print(msg)
    return history

        