#Randomly initialising a 5 hidden layers NN with 200 neurons on each layer.
#Statistics of non linear functions such as tanh and relu.

import numpy as np
import matplotlib.pyplot as plt

design_matrix = np.random.randn(1000, 200)
hidden_layers = [200]*5
nonlinearities = ['tanh']*len(hidden_layers)

activations = {'relu':lambda x:np.max(0,x),
               'tanh':lambda x:np.tanh(x)}

feature_output = {}

for i in range(len(hidden_layers)):
    X = design_matrix if i == 0 else feature_output[i-1]
    layer_input = X.shape[1]
    layer_output = hidden_layers[i]
    W = np.random.randn(layer_input, layer_output)
    H = np.dot(X,W)
    H = activations[nonlinearities[i]](H)
    feature_output[i] = H


print(f'input layer had mean {np.mean(design_matrix)} and std {np.std(design_matrix)}')

layer_mean = [np.mean(H) for _, H in feature_output.items()]
layer_std = [np.std(H) for _, H in feature_output.items()]

for i in feature_output:
    print(f'hidden layers {i+1} had mean {layer_mean[i]} and std {layer_std[i]}')
    
    
plt.figure()
plt.subplot(121)
plt.plot(feature_output.keys(), layer_mean, 'ob-')
plt.title("Layer Mean")

plt.subplot(122)
plt.plot(feature_output.keys(), layer_std, 'ob-')
plt.title("Layer Std")


plt.figure()
for i, H in feature_output.items():
    plt.subplot(1, len(feature_output), i+1)
    plt.hist(H.reshape(-1), 30, range=(-1,1))

plt.show()


