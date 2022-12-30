import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """

    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################
        self.fc1_W = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.fc2_W = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.fc1_b = np.zeros(hidden_dim,)
        self.fc2_b = np.zeros(num_classes,)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        params = {
            "fc1_W": self.fc1_W,
            "fc2_W": self.fc2_W,
            "fc1_b": self.fc1_b,
            "fc2_b": self.fc2_b,
        }
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return params

    def forward(self, X):
        fc_1_out, cache1 = fc_forward(X, self.fc1_W, self.fc1_b)
        relu_out, cache_relu = relu_forward(fc_1_out)
        scores, cache2 = fc_forward(relu_out, self.fc2_W, self.fc2_b)
        cache = [cache1, cache_relu, cache2]
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return scores, cache

    def backward(self, grad_scores, cache):
        grad_fc2, grad_fc2_W, grad_fc2_b = fc_backward(
            grad_scores, cache[2])
        grad_relu=relu_backward(grad_fc2,cache[1])
        grad_x, grad_fc1_W, grad_fc1_b = fc_backward(grad_relu, cache[0])
        grads = {
            "fc2_W": grad_fc2_W,
            "fc2_b": grad_fc2_b,
            "fc1_W": grad_fc1_W,
            "fc1_b": grad_fc1_b,
        }
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads
