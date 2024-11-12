import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import Data_cleanup
import HyperParameters
import Utils as U
from GNN_Model import GNN
from Graph_preprocessing_functions import convert_to_data
device = HyperParameters.device
print(device)

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

#LABELS
###Finish loading data###

### HYPER PARAMETERS ###
BATCH_SIZE = HyperParameters.BATCH_SIZE
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.EPOCHS

#group the graphs and labels together for the DataLoader:
training_group = []
testing_group = []

for graph, label in zip(training_data, training_labels):
    #training
    data = convert_to_data(graph, label)
    training_group.append(data)

'''for graph in training_group:
    for label in set(training_labels):'''


print(f"View a graph data object: {training_group[0]}")

for graph, label in zip(testing_data, testing_labels):
    #testing
    data = convert_to_data(graph, label)
    testing_group.append(data)

#Load the data into training batches.
training_batches = DataLoader(training_group, batch_size=BATCH_SIZE, shuffle=True)
testing_batches = DataLoader(testing_group, batch_size=BATCH_SIZE, shuffle=False)

#DECLARE MODEL INSTANCE WITH INPUT DIMENSION
# Before the model call
Model_0 = GNN(input_dim=HyperParameters.input_dim) # -3 color(R,G,B) + 1 Eccentricity + 1 Aspect_ratio + 1 solidity
Model_0.to(device)
#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model_0.parameters(), lr=LEARNING_RATE)

#Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''

# Lists to store loss values
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss as infinity
patience = HyperParameters.PATIENCE  
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch}\n---------")
    Model_0.train()
    training_loss = 0
    for batch_idx, batch_graphs in enumerate(training_batches):
        #Get the batch of features to send to the model
        batch_graphs = batch_graphs.to(device)
        x = batch_graphs.x
        #print(x)
        y = batch_graphs.y.to(device).long()  #Get the labels in y
        edge_index = batch_graphs.edge_index
        batch = batch_graphs.batch
        #1: Get predictions from the model
        y_pred = Model_0(x, edge_index, batch)
        #print(y_pred)
        #2: Calculate the loss on the model's predictions
        loss = loss_fn(y_pred, y) 
        training_loss += loss.item() #Keep track of each batch's loss
        #print(training_loss)
        #3: optimizer zero grad
        optimizer.zero_grad()

        #4: loss back prop
        loss.backward()

        #5: optimizer step:
        optimizer.step()
    #Finish training batch and calculate the average loss:
    training_loss /= len(training_batches)
    train_losses.append(training_loss)

    #Move to testing on the testing data
    print("Testing the Model...")
    testing_loss, test_acc = 0, 0 #Metrics to test how well the model is doing
    Model_0.eval()
    with torch.inference_mode():
        for batch_idx, batch_graphs in enumerate(testing_batches):
            #Get the batch features to send to the model again
            batch_graphs = batch_graphs.to(device)
            x = batch_graphs.x
            y = batch_graphs.y.to(device).long()  #Get the labels in y
            edge_index = batch_graphs.edge_index
            batch = batch_graphs.batch
            #1: Model Prediction
            y_pred = Model_0(x, edge_index, batch)

            #2: Calculate loss
            loss = loss_fn(y_pred, y)
            testing_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    testing_loss /= len(testing_batches)
    test_acc /= len(testing_batches)
    val_losses.append(testing_loss)
    print(f"Train loss: {training_loss:.4f} | Test loss: {testing_loss:.4f} | Test acc: {test_acc:.4f}%")

    # Check if current validation loss is the best so far
    if testing_loss < best_val_loss:
        best_val_loss = testing_loss
        # Save the model's parameters (state_dict) to a file
        torch.save(Model_0.state_dict(), (U.MODEL_FOLDER / HyperParameters.MODEL_NAME + '.pth').resolve())
        print(f'Saved best model with validation loss: {best_val_loss:.4f}')
        epochs_no_improve = 0  # Reset counter if improvement
    else:
        if testing_loss > training_loss:
            epochs_no_improve += 1
            print(f'Num epochs since improvement: {epochs_no_improve}')
        else:
            print("Not the best, although model is not overfitting yet.")
        #stop training if overfitting starts to happen
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
