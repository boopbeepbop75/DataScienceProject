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
import Data_cleanup

training_data, training_labels = Data_cleanup.load_and_preprocess_images(training_folders)
testing_data, testing_labels = Data_cleanup.load_and_preprocess_images(testing_folders)

def visualize_data():
    sample_size = 10
    training_indexes = random.sample(range(len(training_data)), min(sample_size, len(training_data)))
    testing_indexes = random.sample(range(len(testing_data)), min(sample_size, len(testing_data)))


    for x in training_indexes:
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(training_data[x])
        segments = slic(training_data[x], n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
        Graph_preprocessing_functions.show_comparison(training_data[x], training_labels[x], segments)
        Graph_preprocessing_functions.draw_graph(graph)

    for x in testing_indexes:
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(testing_data[x])
        segments = slic(testing_data[x], n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
        Graph_preprocessing_functions.show_comparison(testing_data[x], testing_labels[x], segments)
        Graph_preprocessing_functions.draw_graph(graph)


if __name__ == "__main__":
    visualize_data()