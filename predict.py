from time import time

import numpy as np
import torch
import torch_neuron
from torchvision import models

from preprocess import preprocess

# Get a sample image
image = preprocess()

# Load the compiled Neuron model
model = torch.jit.load("resnet50_neuron.pt")

# Run inference using the Neuron model
output = model(image)

# Verify that the CPU and Neuron predictions are the same by comparing
# the top-5 results
top5_results = output[0].sort()[1][-5:]
print(f"top-5 labels: {top5_results}")

num_infer = 100

latency = []
for _ in range(num_infer):
    start_time = time()
    results = model(image)
    delta_time = time() - start_time
    latency.append(delta_time)

print(f"Average latency: {np.mean(latency)}")
