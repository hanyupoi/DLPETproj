"""Training model to identify digits from their synogram"""

"""2. Define ODL Radon Transform"""
import odl
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the ODL Radon Transform
def create_radon_transform(image_shape, num_angles):
    space = odl.uniform_discr(
        min_pt=[-1, -1], max_pt=[1, 1], shape=image_shape, dtype='float32'
    )
    geometry = odl.tomo.parallel_beam_geometry(space, num_angles=num_angles)
    radon_transform = odl.tomo.RayTransform(space, geometry)
    return radon_transform

# Apply the Radon Transform to an image
def apply_radon(image, radon_transform):
    image = np.asarray(image.numpy().squeeze(), dtype='float32')  # Convert Tensor to NumPy
    sinogram = radon_transform(image)
    return np.asarray(sinogram, dtype='float32')

print("Im here")

"""3. Transform MNIST Dataset
Create a dataset class that applies the Radon Transform to MNIST images:"""
class ODLRadonTransformDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, radon_transform):
        self.mnist_dataset = mnist_dataset
        self.radon_transform = radon_transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        sinogram = apply_radon(image, self.radon_transform)  # Apply Radon Transform
        sinogram_tensor = torch.tensor(sinogram, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        return sinogram_tensor, label
    
    
"""4. Load and Transform Data
Now, apply the transform to the MNIST dataset:"""
# Define the Radon Transform
image_shape = (28, 28)  # MNIST image size
num_angles = 45  # Number of projection angles
radon_transform = create_radon_transform(image_shape, num_angles)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Apply the Radon Transform
train_dataset = ODLRadonTransformDataset(mnist_train, radon_transform)
test_dataset = ODLRadonTransformDataset(mnist_test, radon_transform)

# DataLoaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


"""5. Define and Train the Neural Network
The neural network and training loop remain the same as before:"""
class SinogramNN(nn.Module):
    def __init__(self, sinogram_size):
        super(SinogramNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(sinogram_size, 128)  # Adjust input size here
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

sinogram_size =  41* num_angles # 41 is nbr of detectors
model = SinogramNN(sinogram_size)

# Model, loss, and optimizer
#model = SinogramNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("im here 2")


# Training loop (same as before)
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing loop (same as before)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt

# Put the model in evaluation mode
model.eval()

# Get a batch of data from the test loader
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# Ensure the data is on CPU
example_data = example_data.to('cpu')
example_targets = example_targets.to('cpu')

# Forward pass to get predictions
with torch.no_grad():
    outputs = model(example_data)
    _, predicted = outputs.max(1)

# Visualization of the first 6 sinograms in the batch
for i in range(6):  # Visualize 6 examples
    plt.subplot(2, 3, i + 1)  # Create a grid of 2 rows and 3 columns
    sinogram = example_data[i].squeeze()  # Remove channel dimension
    plt.imshow(sinogram, cmap='gray')  # Display the sinogram
    plt.title(f"True: {example_targets[i].item()}\nPred: {predicted[i].item()}")
    plt.axis('off')  # Turn off axis ticks and labels

plt.tight_layout()
plt.show()
