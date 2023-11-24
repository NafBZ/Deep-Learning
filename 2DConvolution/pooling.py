import torch

# Define the device to be used for computations
mps_device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# Define the size of the pooling window and the stride
pool_size = 3
stride = 3

# Create MaxPool2d and MaxPool3d instances
pool2D = torch.nn.MaxPool2d(pool_size, stride=stride)
pool3D = torch.nn.MaxPool3d(pool_size, stride=stride)

# Print the pooling layers
print(pool2D)
print(pool3D)

# Create a random 2D image tensor
img_2D = torch.rand(1, 1, 30, 30)

# Create a random 3D image tensor
img_3D = torch.rand(1, 3, 30, 30)

# Apply MaxPool2d to the 2D image tensor
img2D_pool2D = pool2D(img_2D)
print(f'2D image and 2D pool shape: {img2D_pool2D.shape}')

# Apply MaxPool2d to the 3D image tensor
img3D_pool2D = pool2D(img_3D)
print(f'3D image and 2D pool shape: {img3D_pool2D.shape}')

# Apply MaxPool3d to the 3D image tensor
img3D_pool3D = pool3D(img_3D)
print(f'3D image and 3D pool shape: {img3D_pool3D.shape}')