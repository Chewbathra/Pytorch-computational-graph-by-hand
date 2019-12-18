import numpy as np

from functional import F
from variable import Variable


class Parameters:
    """Parameters is a class that wraps all the parameters of the model.

    This class is used in the optimizer.
    """

    def __init__(self, model):
        self.params = {}
        self.model = model
        var_list = vars(model)
        num = 1
        for key, val in var_list.items():
            if isinstance(val, Linear):
                self.params["{}_W".format(key)] = val.W
                if val.bias:
                    self.params["{}_b".format(key)] = val.b


    def get_mode(self):
        """Get the mode of the model."""
        return self.model.mode

    def zero_grad(self):
        """Clear all parameters variables."""
        for key in self.params.keys():
            self.params[key].set_defaults()


class Module:
    """Module is an abstract class that all the models must inherit.
    Contains basic methods for all type of models.
    """

    def parameters(self):
        """Create the wrapper for the parameters of the model and return it."""
        params = Parameters(self)
        return params

    def train(self):
        """Change the mode of the model to train. 

        The optimizer use it to know if it can update the weights:
        mode == train -> it can update.
        """
        self.mode = "train"

    def eval(self):
        """Change the mode of the model to eval. 

        The optimizer use it to know if it can update the weights:
        mode == eval -> it can not update.
        """
        self.mode = "eval"

    def __call__(self, X):
        """Enable the call of the class."""
        return self.forward(X)


class Linear:
    """Applies a linear transformation to the incoming data: y = XW^T + b.

    Pytorch: https://pytorch.org/docs/stable/nn.html#linear 

    Shapes:
        - Input: (N, H_{in}) where H_{in} = in_features
        - Output: (N, H_{out} where H_{out} = out_features

    Attributes:
        weight: the learnable weights of the module of shape 
            (out_features, in_features). The values are initialized from 
            Uniform(-sqrt{k}, sqrt{k}), where k = 1/in_features.
        bias:   the learnable bias of the module of shape (1, out_features).
                If bias is True, the values are initialized from
                Uniform(-sqrt{k}, sqrt{k}), where k = 1/in_features.
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        #######################################################################
        # TODO: Initialize the weights accordind to the description above.
        # Ton't forget to wrap the data into a Variable.
        #######################################################################
        k = 1 / in_features
        self.W = Variable(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_features, in_features)))
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################

        if self.bias:
            #######################################################################
            # TODO: Initialize the bias accordind to the description above.
            # Ton't forget to wrap the data into a Variable.
            #######################################################################
            self.b = Variable(np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(1, out_features)))
            #######################################################################
            # --------------------------- END OF YOUR CODE ------------------------
            #######################################################################

    def __call__(self, X):
        """Computes the forward pass."""
        y = None
        #######################################################################
        # TODO: Use the functional module to compute the first part of the
        # linear transfomation -> y = XW.T
        #######################################################################
        y = F.matmul(X, self.W.t())
        #######################################################################
        # --------------------------- END OF YOUR CODE ------------------------
        #######################################################################
        if self.bias:
            #######################################################################
            # TODO: If the bias is true add the bias.
            #######################################################################
            y = F.add(y, self.b)
            #######################################################################
            # --------------------------- END OF YOUR CODE ------------------------
            #######################################################################
        return y


class CrossEntropyLoss:
    """Cross Entropy as in Pytorch with (log) softmax."""

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def _forward(self, X, y):
        """Compute the forward of this loss, it includes the softmax and the 
        cross entropy itself.

        Formula based of the CrossEntropyLoss of Pytorch:
        https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        """
        result = F.log(F.exp(X).sum(1)) - X[range(X.shape[0]), np.ravel(y.data)]
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'none':
            return result
        else:
            raise RuntimeError("Reduction not known")

    def __call__(self, X, y):
        """Call the forward pass.

        There is a problem during the backpropagation, with this function!
        This function provides a workaround by copying the output of the 
        network X and backpropagte trough it, than copying the gradients back to 
        the real X and finaly by changing the grad_fn and the grads of the 
        result to be the ones of X. It's equivalent of propagating from the 
        loss to the scores.
        """
        X_detach = Variable(X.data)
        result = self._forward(X_detach, y)
        result.backward()
        X.grad = X_detach.grad
        result.grad = X_detach.grad
        result.grad_fn = X.grad_fn
        return result
