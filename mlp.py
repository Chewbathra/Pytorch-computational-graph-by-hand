import nn as nn
from functional import F
class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        #######################################################################
        # TODO: define 2 linear layers, one that takes the inputs and outputs
        # values with hidden_size
        # and the second one that takes the values from the first layer and
        # outputs the scores.
        # implement Linear in nn.py before, you need it here.
        #######################################################################
        self.firstLayer = nn.Linear(in_features, hidden_size)
        self.secondLayer = nn.Linear(hidden_size, out_features)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

    def forward(self, X):
        output = None
        #######################################################################
        # TODO: define your forward pass as follow
        #    1) y = linear(inputs)
        #    2) y_nl = relu(y)
        #    3) output = linear(y_nl)
        # softmax not needed because it's already in cross entropy
        #######################################################################
        y = self.firstLayer(X)
        y_nl = F.relu(y)
        output = self.secondLayer(y_nl)
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################
        return output