import torch
def train_model(model,train_loader,criterion,optimizer,device,epochs=10):
    history={"train_loss":[]}
    
    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        for X_batch,y_batch in train_loader:
            X_batch=X_batch.to(device)
            y_batch=y_batch.to(device)
            optimizer.zero_grad()
            outputs=model(X_batch)
            loss=criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        average_loss=(total_loss/len(train_loader))
        history["train_loss"].append(average_loss)
        print(f"Epoch {epoch+1}/{epochs}"
              f"| Loss: {average_loss:.4f}")
    return history

        