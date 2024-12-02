import glob
import os

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch_geometric.utils import from_networkx

import Graph_preprocessing_functions
import HyperParameters
import Utils as U
from Utils import training_folders, testing_folders
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T

def apply_augmentations(image):

    transforms = [
        T.RandomResizedCrop(size= HyperParameters.target_size, scale=(0.6, 0.8)),
        T.ColorJitter(brightness=1.5, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomErasing(p=1.0, scale=(0.2, 0.33), ratio=(0.3, 3.3), value=0),
    ]
    selected_transform = random.choice(transforms)
    #print(transforms.index(selected_transform))
    # Convert the image to a PyTorch tensor before applying transformations
    tensor_image = T.ToTensor()(image)
    augmented_image = selected_transform(tensor_image)
    
    # Convert back to PIL for consistency
    return T.ToPILImage()(augmented_image)

def load_and_preprocess_pred_images(folder, target_size=HyperParameters.target_size):
    #Extract images from data_pred folder, turn them into numpy arrays normalized between 0 and 1
    images = []
    print(folder)
    num = 0
    for image_path in glob.glob(os.path.join(folder, "*.jpg")):
            try:
                img = Image.open(image_path).convert("RGB").resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                num+=1

            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
    print(f'Amount: {num}')
    return np.array(images)

def load_and_preprocess_images(folders, target_size=HyperParameters.target_size):
   #Extract images from data_pred folder, turn them into numpy arrays normalized between 0 and 1

    #makes sure only the training data is being augmented
    augment = False
    if(folders == training_folders):
        augment = True

    images = []
    labels = []
    #print(folder)
    num = 0
    for label, folder in enumerate(folders):
        print(folder)
        print(label)
        for image_path in glob.glob(os.path.join(folder, "*.jpg")):
                try:
                    img = Image.open(image_path).convert("RGB").resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)

                    if augment:
                        aug_img = apply_augmentations(img)
                        Aug_img_array = np.array(aug_img, dtype=np.float32)
                        images.append(Aug_img_array)
                        labels.append(label)
                        num+=1
                    num+=1
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
    print(f'Amount: {num}')
    return images, labels

def process_images_to_graphs(images, labels):
    #Takes images and their labels
    processed_graphs = []
    #create random stops for visualization during prepocessing. Turn on/off using show_visualization_stops var in Hyperparameters folder.
    stops = sorted(random.sample(range(len(images)), min(20, len(images)))) 
    print(stops)
    for i, image in enumerate(images):
        #Make the graph
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(image)
        #Add graph to graph list
        processed_graphs.append(graph)
        #Update progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(images)} images")
        if i in stops and HyperParameters.show_visualization_stops:
            #Visualize random Graphs
            segments = slic(image, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(image, labels[i], segments)
            Graph_preprocessing_functions.draw_graph(graph)
    return processed_graphs

def clean_data():
    print(training_folders[0])
    print()

    print("Loading and preprocessing images...")
    training_data, training_labels = load_and_preprocess_images(training_folders)
    testing_data, testing_labels = load_and_preprocess_images(testing_folders)

    print("Processing training data to graphs...")
    processed_training_graphs = process_images_to_graphs(training_data, training_labels)

    print("Processing testing data to graphs...")
    processed_testing_graphs = process_images_to_graphs(testing_data, testing_labels)

    print("Saving processed graphs...")
    training_tensor = [from_networkx(G) for G in processed_training_graphs]  # Convert Graphs to PyTorch Geometric Data objects
    testing_tensor =  [from_networkx(G) for G in processed_testing_graphs]  # Convert Graphs to PyTorch Geometric Data objects
    #print(training_tensor)
    #Save graphs
    torch.save(training_tensor, (U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve()) 
    torch.save(testing_tensor, (U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    
    print("Saving labels...")
    np.save((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve(), training_labels)
    np.save((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve(), testing_labels)

    print("Data cleanup completed successfully.")

if __name__ == "__main__":
    clean_data()

