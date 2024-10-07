import numpy as np
from skimage.color import rgb2lab
from skimage.measure import regionprops
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.ndimage import find_objects
from itertools import combinations

def create_edge_index_from_slic(segments):
    """
    Create edge_index from SLIC segmentation.
    
    :param segments: 2D numpy array of segment labels from SLIC
    :return: edge_index tensor for PyTorch Geometric
    """
    # Find unique segments
    unique_segments = np.unique(segments)
    num_segments = len(unique_segments)
    
    # Create a mapping from segment label to index
    segment_to_index = {seg: idx for idx, seg in enumerate(unique_segments)}
    
    # Find bounding box for each segment
    bounding_boxes = find_objects(segments)
    
    # Function to check if two segments are neighbors
    def are_neighbors(seg1, seg2):
        bb1 = bounding_boxes[segment_to_index[seg1]]
        bb2 = bounding_boxes[segment_to_index[seg2]]
        return (
            (bb1[0].start <= bb2[0].stop and bb2[0].start <= bb1[0].stop) and
            (bb1[1].start <= bb2[1].stop and bb2[1].start <= bb1[1].stop)
        )
    
    # Create edges
    edges = []
    for seg1, seg2 in combinations(unique_segments, 2):
        if are_neighbors(seg1, seg2):
            # Add edges in both directions
            edges.append([segment_to_index[seg1], segment_to_index[seg2]])
            edges.append([segment_to_index[seg2], segment_to_index[seg1]])
    
    # Convert to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return edge_index

def compute_node_features(image, segments):
    # Convert image to LAB color space
    image_lab = rgb2lab(image)
    
    # Get unique segments
    unique_segments = np.unique(segments)
    
    # Initialize feature array
    num_features = 6  # 3 for color, 3 for other properties
    node_features = np.zeros((len(unique_segments), num_features), dtype=np.float32)
    
    for i, segment in enumerate(unique_segments):
        # Create mask for current segment
        mask = segments == segment
        
        # Get region properties
        props = regionprops(mask.astype(int), image_lab)[0]
        
        # Compute features
        mean_color = props.mean_intensity
        area = props.area
        perimeter = props.perimeter
        
        # Ensure mean_color is 3-dimensional
        if mean_color.size == 1:
            mean_color = np.array([mean_color, 0, 0])
        elif mean_color.size == 2:
            mean_color = np.append(mean_color, 0)
        
        # Combine features
        node_features[i] = np.concatenate([mean_color, [area, perimeter, props.eccentricity]])
    
    return torch.from_numpy(node_features)