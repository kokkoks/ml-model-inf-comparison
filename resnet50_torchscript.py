import torch
from torchvision import models

# Create an example input for compilation
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Tell the model we are using it for evaluation (not training)
model.eval()

# Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model(model, example_inputs=[image])

# Compile the model using torch.jit.trace to create a TorchScript model
model_torchscript = torch.jit.trace(model, example_inputs=[image])

# Save the compiled model
model_torchscript.save("resnet50_torchscript.pt")
