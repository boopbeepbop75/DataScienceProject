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


def load_and_preprocess_images(folders, target_size=(128, 128), extensions=("jpg", "jpeg", "png", "gif")):
    images = []
    labels = []
    for label, folder in enumerate(folders):
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                try:
                    img = Image.open(image_path).convert("RGB").resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
    return np.array(images), np.array(labels)

def process_images_to_graphs(images, n_segments, sigma):
    processed_graphs = []
    for i, image in enumerate(images):
        segments = slic(image, n_segments=n_segments, sigma=sigma)
        edge_index = Graph_preprocessing_functions.create_edge_index_from_slic(segments)
        node_features = Graph_preprocessing_functions.compute_node_features(image, segments)
        graph = Data(x=node_features, edge_index=edge_index)
        processed_graphs.append(graph)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(images)} images")
    return processed_graphs

def clean_data():
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    training_folders = [(U.RAW_DATA_FOLDER / 'seg_train' / 'seg_train' / class_name).resolve() 
                        for class_name in classes]
    testing_folders = [(U.RAW_DATA_FOLDER / 'seg_test' / 'seg_test' / class_name).resolve()
                       for class_name in classes]

    print("Loading and preprocessing images...")
    training_data, training_labels = load_and_preprocess_images(training_folders)
    testing_data, testing_labels = load_and_preprocess_images(testing_folders)

    print("Processing training data to graphs...")
    processed_training_graphs = process_images_to_graphs(training_data, HyperParameters.n_segments, HyperParameters.sigma)

    print("Processing testing data to graphs...")
    processed_testing_graphs = process_images_to_graphs(testing_data, HyperParameters.n_segments, HyperParameters.sigma)

    print("Saving processed graphs...")
    torch.save(processed_training_graphs, (U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    torch.save(processed_testing_graphs, (U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    
    print("Saving labels...")
    np.save((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve(), training_labels)
    np.save((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve(), testing_labels)

    print("Data cleanup completed successfully.")

if __name__ == "__main__":
    clean_data()
