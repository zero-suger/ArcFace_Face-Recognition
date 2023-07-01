# -*- coding: utf-8 -*-
"""Angular_distance.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eHAibK_0wIM1AoBo-JAg5lCaQpim1kff
"""

import numpy as np

# Assuming you have two .npy files containing embeddings: embedding1.npy and embedding2.npy

# Load the embeddings from the .npy files
embedding1 = np.load('embedding1.npy')
embedding2 = np.load('embedding2.npy')

# Normalize the embeddings
embedding1 = embedding1 / np.linalg.norm(embedding1)
embedding2 = embedding2 / np.linalg.norm(embedding2)

# Compute the angular distance using the ArcFace loss formula
angular_distance = np.arccos(np.dot(embedding1, embedding2.T))

# Print the angular distance
print("Angular Distance:", angular_distance)