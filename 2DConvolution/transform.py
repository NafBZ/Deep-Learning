import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# Load CIFAR10 dataset
dataset = torchvision.datasets.CIFAR10(root='cifar10')

# Visualize random images from the dataset
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in ax.flatten():
    # Get a random image
    img, label = dataset[np.random.randint(0, 50000)]
    class_name = dataset.classes[label]
    # Visualize the image
    i.imshow(img)
    i.set_title(f'{class_name}')
    i.axis('off')
plt.tight_layout()

# Define transformations for the dataset
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32 * 4, antialias=True),
    transforms.Grayscale(num_output_channels=1),
])

# Apply transformations to a specific image
transfored_data = transformation(dataset.data[123, :, :, :])

# Visualize the original and transformed image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(dataset.data[123, :, :, :])
ax[0].set_title('Original image')
ax[1].imshow(torch.squeeze(transfored_data), cmap='gray')

plt.show()
