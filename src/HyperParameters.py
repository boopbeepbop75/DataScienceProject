### SLIC HYPER PARAMETERS ###
n_segments = 75
sigma = 5

### MODEL HYPER PARAMETERS ###
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
BATCH_SIZE = 32
HIDDEN_UNITS = 10
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = .001
EPOCHS = 100
