import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import find_objects
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torchvision.transforms import ToTensor

import Data_cleanup
import HyperParameters
import Utils as U
from Dataset import SLICDataset

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

    print(training_data)

    # Extract the images, graphs, and edges
    '''normalized_training_images = training_data['images']  # Images (already tensors)
    training_edge_indices = training_data['edge_indices']  # Edge indices
    training_node_features = training_data['node_features']  # Node features

    normalized_testing_images = testing_data['images']  # Images (already tensors)
    testing_edge_indices = testing_data['edge_indices']  # Edge indices
    testing_node_features = testing_data['node_features']  # Node features'''

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

    # Further preprocessing (assuming you generate node features and edges during cleanup)
    # Example: create_edge_index_from_slic(segments) and compute_node_features(image, segments)

#LABELS
###Finish loading data###

def show_image(x, y):
    fig = plt.figure("Superpixels -- %d segments" % (50))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    plt.title(y)
    plt.axis("off")
    plt.show()

'''#CHECK IMAGES
random_index = random.randint(0, len(normalized_training_images)-1)
random_index_2 = random.randint(0, len(normalized_testing_images)-1)
show_image(normalized_training_images[random_index], HyperParameters.CLASSES[training_labels[random_index]])
show_image(normalized_testing_images[random_index_2], HyperParameters.CLASSES[testing_labels[random_index_2]])

# Transpose to (N, C, H, W) format
normalized_training_images = np.transpose(normalized_training_images, (0, 3, 1, 2))  # Shape (N, 3, 128, 128)
normalized_testing_images = np.transpose(normalized_testing_images, (0, 3, 1, 2))  # Shape (N, 3, 128, 128)
#Convert Data to Tensors
normalized_training_images = torch.tensor(normalized_training_images, dtype=torch.float32)  
normalized_testing_images = torch.tensor(normalized_testing_images, dtype=torch.float32)  
print(normalized_training_images.dtype)
print(normalized_testing_images.dtype)

#Check shape of the data
print(normalized_training_images[0].shape) #(3, 128, 128)'''

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.EPOCHS



# Usage example
'''segments = slic(image, n_segments=50, sigma=5)
edge_index = create_edge_index_from_slic(segments)'''

