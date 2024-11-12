from skimage.segmentation import slic
from torch_geometric.utils import from_networkx

import Graph_preprocessing_functions
import HyperParameters
from Utils import training_folders, testing_folders
import random
import Data_cleanup

n_segments = 50

'''
Easily visualize the graphs using a small number of segments
'''

def visualize_data():
    #Preprocess Dataset images
    training_data, training_labels = Data_cleanup.load_and_preprocess_images(training_folders)
    testing_data, testing_labels = Data_cleanup.load_and_preprocess_images(testing_folders)
    sample_size = 64
    view_size = 5 #Number of training and testing images shown
    training_indexes = random.sample(range(len(training_data)), min(sample_size, len(training_data)))
    testing_indexes = random.sample(range(len(testing_data)), min(sample_size, len(testing_data)))

    input("Press enter to visualize...")
    for i, x in enumerate(training_indexes):
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(training_data[x], n_segments)
        if i < view_size:
            segments = slic(training_data[x], n_segments=n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(training_data[x], training_labels[x], segments)
            Graph_preprocessing_functions.draw_graph(graph, HyperParameters.CLASSES[training_labels[x]])

    for i, x in enumerate(testing_indexes):
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(testing_data[x], n_segments)
        if i < view_size:
            segments = slic(testing_data[x], n_segments=n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(testing_data[x], testing_labels[x], segments)
            Graph_preprocessing_functions.draw_graph(graph, HyperParameters.CLASSES[testing_labels[x]])


if __name__ == "__main__":
    visualize_data()