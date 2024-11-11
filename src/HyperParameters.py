import torch

### SLIC HYPER PARAMETERS ###
n_segments = 300
sigma = 5
show_visualization_stops = False
target_size=(150, 150)

### MODEL HYPER PARAMETERS ###
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
BATCH_SIZE = 32
DROPOUT_RATE = .5
HIDDEN_UNITS = 40
NUM_HEADS = 16
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = .0002
EPOCHS = 200
PATIENCE = 10  # Number of epochs to wait before early stopping
input_dim=3+1+1+1 #Color(3), eccentricity(1), aspect_ratio(1), solidity(1)

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"