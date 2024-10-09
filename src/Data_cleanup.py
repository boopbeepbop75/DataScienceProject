import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch_geometric.data import Data

import Graph_preprocessing_functions
import HyperParameters
import Utils as U
from Utils import training_folders, testing_folders
import random


def load_and_preprocess_images(folders, target_size=(128, 128), extensions=("jpg", "jpeg", "png", "gif")):
    images = []
    labels = []
    for label, folder in enumerate(folders):
        print(folder)
        num = 0
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                try:
                    img = Image.open(image_path).convert("RGB").resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    num+=1
                    labels.append(label)
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
        print(f'Amount: {num}')
    return np.array(images), np.array(labels)

def process_images_to_graphs(images, labels):
    processed_graphs = []
    stops = random.sample(range(len(images)), min(20, len(images))) #create random stops for visualization
    print(stops)
    for i, image in enumerate(images):
        #Make the graph
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(image)
        #Add graph to graph list
        processed_graphs.append(graph)
        #Update progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(images)} images")
        if i in stops:
            #Visualize random Graphs
            segments = slic(image, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(image, labels[i], segments)
            Graph_preprocessing_functions.draw_graph(graph)
    return processed_graphs

def clean_data():
    classes = HyperParameters.CLASSES

    print("Loading and preprocessing images...")
    training_data, training_labels = load_and_preprocess_images(training_folders)
    testing_data, testing_labels = load_and_preprocess_images(testing_folders)

    print("Processing training data to graphs...")
    processed_training_graphs = process_images_to_graphs(training_data, training_labels)

    print("Processing testing data to graphs...")
    processed_testing_graphs = process_images_to_graphs(testing_data, testing_labels)

    print("Saving processed graphs...")
    torch.save(processed_training_graphs, (U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    torch.save(processed_testing_graphs, (U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    
    print("Saving labels...")
    np.save((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve(), training_labels)
    np.save((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve(), testing_labels)

    print("Data cleanup completed successfully.")

if __name__ == "__main__":
    clean_data()
