
class Optimizer:
    """Abstract class for all the optimizers, store the parameters wrapper and
    has a method to clear out the parameters.
    """

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Clear the parameters.

        Ususally call this before a new iteration.
        """
        self.parameters.zero_grad()


class SGD(Optimizer):
    """
    Applies the SGD update to the weights W = lr * W.grad.
    """

    def __init__(self, parameters, lr=1e-3):
        super().__init__(parameters, lr)

    def step(self):
        """If the model is in train mode update the weights by SGD."""
        if self.parameters.get_mode() == "train":
            #######################################################################
            # TODO: Implement the SGD update mechanism.
            # to acces the data of parametes Variables:
            #   - self.parameters.params[key].data
            #######################################################################
            for key in self.parameters.params.keys():
                self.parameters.params[key].data = self.parameters.params[key].data - self.lr * self.parameters.params[key].grad

            #######################################################################
            # --------------------------- END OF YOUR CODE ------------------------
            #######################################################################
