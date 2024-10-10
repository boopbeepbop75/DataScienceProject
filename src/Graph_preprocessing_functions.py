import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops, label
from HyperParameters import *

# Original image
def show_comparison(image, label, segments):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    ax1.imshow(image)
    ax1.set_title(CLASSES[label])
    ax1.axis('off')

    # Superpixel image with boundaries
    ax2.imshow(mark_boundaries(image, segments))
    ax2.set_title('SLIC Segmentation')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def find_neighbors(segments, superpixel_id):
    """
    Find neighboring superpixels for a given superpixel ID.

    Parameters:
        segments (ndarray): 2D array where each element is the ID of the superpixel.
        superpixel_id (int): The ID of the superpixel to find neighbors for.

    Returns:
        list: A list of neighboring superpixel IDs.
    """
    # Get the indices of the given superpixel
    superpixel_indices = np.argwhere(segments == superpixel_id)
    
    # Initialize a set to hold neighboring superpixels
    neighbors = set()
    
    # Check for neighbors in the 8 surrounding pixels
    for idx in superpixel_indices:
        x, y = idx[0], idx[1]
        
        # Loop through neighboring coordinates
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):  # Skip the center pixel
                    continue
                neighbor_x, neighbor_y = x + dx, y + dy
                
                # Check if the neighbor is within bounds
                if (0 <= neighbor_x < segments.shape[0] and
                    0 <= neighbor_y < segments.shape[1]):
                    neighbor_id = segments[neighbor_x, neighbor_y]
                    if neighbor_id != superpixel_id:  # Avoid adding the same superpixel
                        neighbors.add(neighbor_id)
    
    return list(neighbors)

def average_color_of_superpixel(image, segments, segment_id):
    # Get the mask for the superpixel
    mask = (segments == segment_id)

    # Calculate the average color
    average_color = image[mask].mean(axis=0)

    return average_color

def calculate_eccentricity(segments, segment_id):
    mask = (segments == segment_id)
    labeled_superpixel = label(mask)
    properties = regionprops(labeled_superpixel)
    region = properties[0]
    
    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
    return region.eccentricity, aspect_ratio

def make_graph_for_image_slic(slic_image):
    segments = slic(slic_image, n_segments=n_segments, sigma=sigma)
    
    # Create the graph
    G = nx.Graph()

    # Step 2: Loop over each segment (superpixel)
    for segment_id in np.unique(segments):
        average_color = average_color_of_superpixel(slic_image, segments, segment_id) #Get the average color of the superpixel
        eccentricity, aspect_ratio = calculate_eccentricity(segments, segment_id) #Calculate how close the superpixel is to a circular shape
        neighbors = find_neighbors(segments, segment_id) #Find the neighboring superpixels

        #print(f'Eccentricity: {eccentricity}')
        G.add_node(segment_id, color=average_color, eccentricity=eccentricity, aspect_ratio=aspect_ratio) #Create the node, detailing the shape and average color
        #print(neighbors)
        for neighbor in neighbors:
            G.add_edge(segment_id, neighbor) #Add the neighboring superpixels as edges connected the neighbor nodes to the current superpixel

    return G

def draw_graph(G):
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    # Extract eccentricity values for node sizes
    eccentricity_values = [data['eccentricity'] for _, data in G.nodes(data=True)]
    # Normalize the values to scale them for node sizes
    #normalized_sizes = [500 * (ecc / max(eccentricity_values)) for ecc in eccentricity_values]  # Scale for visibility
    #Shape the nodes based on their aspect ratios
    node_shapes = ["o" if G.nodes[node]['aspect_ratio'] < 1.5 else "s" for node in G.nodes]

    # Normalize the eccentricity values to scale them for node sizes
    normalized_sizes = [500 * (ecc / max(eccentricity_values)) for ecc in eccentricity_values]  # Scale for visibility
    #Draw the graph
    # Generate positions for nodes
    pos = nx.spring_layout(G)

    # Draw each group of nodes based on their shape
    for shape in ["o", "s"]:
        # Select nodes and filter the corresponding sizes
        selected_nodes = [node for node, node_shape in zip(G.nodes, node_shapes) if node_shape == shape]
        selected_sizes = [normalized_sizes[i] for i, node in enumerate(G.nodes) if node in selected_nodes]
        selected_colors = [node_colors[i] for i, node in enumerate(G.nodes) if node in selected_nodes]
        
        nx.draw_networkx_nodes(G, pos, node_shape=shape, nodelist=selected_nodes, 
                            node_color=selected_colors, node_size=selected_sizes,
                            edgecolors="black", linewidths=2)

    # Draw edges and labels as usual
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_color="white")

    plt.show()