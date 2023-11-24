import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

convolution_instance = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=0)

# create a random image
imgsize = (1,3,64,64)
img = torch.rand(imgsize)

# permute the dimensions of the image to fit in the matplotlib format
img2view = img.permute(2,3,1,0).numpy()
plt.imshow(np.squeeze(img2view));

# apply the convolution
convolution = convolution_instance(img)


fig,ax = plt.subplots(3,4,figsize=(10,5))

for i,j in enumerate(ax.flatten()):
    # get the i-th filter
    I = torch.squeeze(convolution[0,i,:,:]).detach()

    # visualize the convolution result on the first channel
    j.imshow(I,cmap='Purples')
    j.set_title('Conv. w/ filter %s'%i)
    j.axis('off')

plt.tight_layout()
plt.show()