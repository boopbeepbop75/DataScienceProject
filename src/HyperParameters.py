### SLIC HYPER PARAMETERS ###
n_segments = 150
sigma = 5
show_visualization_stops = True

### MODEL HYPER PARAMETERS ###
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
BATCH_SIZE = 32
HIDDEN_UNITS = 24
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = .0001
EPOCHS = 500
