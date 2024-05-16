# Purpose: This file contains the training and prediction functions for the diacritization task model.
# Prepared for YZV405E Natural Language Processing Istanbul Technical University
# Authors: Muhammet Serdar NAZLI, Hasan Taha BAĞCI


from tqdm import tqdm 
import torch
import numpy as np 


def train_model(dataloader, model, criterion, optimizer, epochs, device='cpu', save_best_model=False):
    min_loss = np.inf
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for poisoned, original in progress_bar:
            original, poisoned = original.to(device), poisoned.to(device)
            optimizer.zero_grad(),
            try:
                output = model(poisoned)
            except:
                print(original.shape)
                raise ValueError(f"")
            loss = criterion(output.transpose(1, 2), original)
            if loss.item() < min_loss and save_best_model: 
                min_loss = loss.item()
                torch.save(model.state_dict(), f"best_model_so_far_{epoch}.pt")
            try:
                loss.backward()
            except:
                print(original.shape)
                raise ValueError(f"")
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})
        if epoch % 10 == 9:
            print(f"Epoch {epoch+1} completed. Saving model...")
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")


def predict(model, dataloader, index_to_char, device='cpu', sos_eos_tokens=False):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for original in tqdm(dataloader, desc="Predicting"):  # Update here to handle single value
            original = original.to(device)
            output = model(original)
            predicted_indices = output.argmax(dim=2)
            for indices in predicted_indices:
                predicted_chars = [index_to_char.get(idx.item(), '') for idx in indices]
                prediction = ''.join(predicted_chars) 
                prediction = prediction.replace("[SOS]", '').replace("[EOS]", '')  if sos_eos_tokens else prediction
                predictions.append(prediction)
    return predictions # Remove SOS and EOS tokens, return
