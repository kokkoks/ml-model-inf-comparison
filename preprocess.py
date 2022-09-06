import os

import numpy as np
import torch
from torchvision import datasets, transforms


def preprocess(batch_size=1, num_neuron_cores=1):
    # Define a normalization function using the ImageNet mean and standard deviation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Resize the sample image to [1, 3, 224, 224], normalize it, and turn it into a tensor
    eval_dataset = datasets.ImageFolder(
        os.path.dirname("./torch_neuron_test/"),
        transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    image, _ = eval_dataset[0]
    image = torch.tensor(image.numpy()[np.newaxis, ...])

    # Create a "batched" image with enough images to go on each of the available NeuronCores
    # batch_size is the per-core batch size
    # num_neuron_cores is the number of NeuronCores being used
    batch_image = image
    for i in range(batch_size * num_neuron_cores - 1):
        batch_image = torch.cat([batch_image, image], 0)

    return batch_image
