import torch
import torch_neuron
from torchvision import models

# Create an example input for compilation
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Tell the model we are using it for evaluation (not training)
model.eval()

# Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model(model, example_inputs=[image])

# Compile the model using torch.neuron.trace to create a Neuron model
# that that is optimized for the Inferentia hardware
model_neuron = torch.neuron.trace(model, example_inputs=[image])

# Save the compiled model
model_neuron.save("resnet50_neuron.pt")
