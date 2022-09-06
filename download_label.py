import json
import os
from urllib import request

# Create an image directory containing a sample image of a small kitten
os.makedirs("./torch_neuron_test/images", exist_ok=True)
request.urlretrieve(
    "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
    "./torch_neuron_test/images/kitten_small.jpg",
)

# Fetch labels to output the top classifications
request.urlretrieve(
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    "imagenet_class_index.json",
)
idx2label = []

# Read the labels and create a list to hold them for classification
with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
